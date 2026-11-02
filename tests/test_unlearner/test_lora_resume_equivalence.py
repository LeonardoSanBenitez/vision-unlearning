"""Does a low-rank adapter run that stops and resumes land exactly where an unbroken run lands?

A run that must stop part-way (a time limit on a shared machine, a crash) is resumed from the training
state it wrote. That is only safe if the resumed run is the SAME run: every trained tensor equal, bit
for bit, to the run that was never interrupted. Anything less is a different model that nobody can
tell apart from the right one, because it trains, its loss looks plausible and it never raises.

What has to come back for that to hold: the adapter, the optimiser moments, the learning-rate
schedule, every random generator, and -- the part the accelerator does not know about -- the retain
loader's place in its pass. The trainer advances the retain loader by hand, one held iterator across
steps and epochs, so at an epoch boundary it is usually part-way through a pass. Restarting it on a
fresh pass gives the resumed run different retain batches, a different shuffle and different random
flips from then on.

The arms:
  A. N epochs, unbroken;
  B. N/2 epochs with the training state written at the end, then a FRESH unlearner that resumes from
     that state and trains to N -- in another output directory, through the public resume path, as a
     second job would.

The denoiser is a tiny real ``UNet2DConditionModel`` with a real adapter, dropout and all; the data
are a few fixed tensors whose random flip is drawn while loading, as the real transform does. The
trajectory is under test, not the realism of the model, and keeping it on a processor in seconds is
what makes it runnable at all. The forget set gives 3 batches per epoch and the retain set 5 per pass,
so at the end of epoch 2 the retain loader is one batch into its second pass: the case that breaks.

    python -m pytest tests/test_unlearner/test_lora_resume_equivalence.py -v
"""
from __future__ import annotations

import logging
import os
import pathlib
import random
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from accelerate.utils import set_seed
from peft.utils import get_peft_model_state_dict

from vision_unlearning.unlearner.lora import RETAIN_STREAM_FILE, UnlearnerLora
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

LATENT_SIDE = 8
CROSS_DIM = 8
SEQUENCE = 3
SEED = 42
TOTAL_EPOCHS = 4
HALF_EPOCHS = 2


class _FlippingSet(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Fixed latents that are flipped at random when loaded, from the generator of whoever loads them."""

    def __init__(self, size: int, offset: float) -> None:
        self.size = size
        self.offset = offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base = torch.linspace(-1.0, 1.0, 4 * LATENT_SIDE * LATENT_SIDE).view(4, LATENT_SIDE, LATENT_SIDE)
        pixel_values = base * (index + 1) * 0.1 + self.offset
        if torch.rand(()) < 0.5:
            pixel_values = torch.flip(pixel_values, dims=[2])
        return {"pixel_values": pixel_values, "input_ids": torch.full((SEQUENCE,), index % CROSS_DIM)}


class _TinyLora(UnlearnerLora):
    """The base trainer's loop around a step shaped like SPARE's: random noise and timesteps from the
    global generators, the adapter's dropout, both batches in the loss, one optimiser step per
    accumulation boundary."""

    def _prepare_dataloaders(self) -> Tuple[Any, Any]:
        def loader(dataset: _FlippingSet) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
            return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.per_device_train_batch_size,
                                               num_workers=self.dataloader_num_workers)
        return loader(_FlippingSet(6, 0.0)), loader(_FlippingSet(10, 0.3))

    def _train_one_batch(self, batch_forget: Dict[str, Any], batch_retain: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._unet is not None and self._accelerator is not None and self._optimizer is not None

        def loss_of(batch: Dict[str, Any]) -> torch.Tensor:
            latents = batch["pixel_values"]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],))
            context = torch.nn.functional.one_hot(batch["input_ids"], CROSS_DIM).float()
            prediction = self._unet(latents + noise, timesteps, context).sample
            return torch.nn.functional.mse_loss(prediction, noise)

        loss_forget = loss_of(batch_forget)
        loss_retain = loss_of(batch_retain)
        self._accelerator.backward(loss_forget + loss_retain)
        if self._accelerator.sync_gradients:
            self._accelerator.clip_grad_norm_(self._lora_layers, self.max_grad_norm)
        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad()
        return loss_forget.detach(), loss_retain.detach()


def _tiny_unet() -> torch.nn.Module:
    from diffusers import UNet2DConditionModel

    torch.manual_seed(5)  # identical base weights in every arm, or nothing below means anything
    unet = UNet2DConditionModel(
        sample_size=LATENT_SIDE, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=CROSS_DIM,
        attention_head_dim=4, norm_num_groups=8,
    )
    unet.requires_grad_(False)
    return unet


def _fit(output_dir: pathlib.Path, epochs: int, num_workers: int = 0, **overrides: Any) -> _TinyLora:
    """Build a fresh trainer and run what `train` runs around `_fit`, without loading Stable Diffusion."""
    settings: Dict[str, Any] = {
        "model_name_or_path": "unused",
        "dataset_forget_name": "unused",
        "dataset_retain_name": "unused",
        "output_dir": str(output_dir),
        "gradient_weighting_method": GradientWeightingMethodSimple(),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-2,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "max_grad_norm": 5.0,
        "lora_r": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.2,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "mixed_precision": "no",
        "checkpointing_steps": 10000,
        "dataloader_num_workers": num_workers,
        "seed": SEED,
    }
    settings.update(overrides)
    trainer = _TinyLora(**settings)
    os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    trainer._accelerator = trainer._get_accelerator()
    trainer._unet = _tiny_unet()
    trainer._train_forget_dataloader, trainer._train_retain_dataloader = trainer._prepare_dataloaders()
    trainer._fit()
    return trainer


def _adapter(trainer: _TinyLora) -> Dict[str, torch.Tensor]:
    assert trainer._unet is not None
    unet = trainer._accelerator.unwrap_model(trainer._unet)  # type: ignore[union-attr]
    return {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(unet).items()}


def _differing(first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor]) -> List[str]:
    assert first.keys() == second.keys()
    return [name for name in first if not torch.equal(first[name], second[name])]


def _checkpoints(directory: pathlib.Path) -> List[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("checkpoint-"))


class _Arms:
    """The unbroken run, the first half with its training state written, and the resumed second half."""

    def __init__(self, root: pathlib.Path, num_workers: int) -> None:
        self.root = root
        self.unbroken = _fit(root / "unbroken", TOTAL_EPOCHS, num_workers)
        # Written at epoch 1 as well, so the scheduling of the writes is checked on the same run.
        self.first = _fit(root / "first", HALF_EPOCHS, num_workers, checkpoint_at_epochs=[1, HALF_EPOCHS])
        self.checkpoint = root / "first" / "checkpoint-4"  # 2 optimiser steps per epoch
        self.resumed = _fit(root / "second", TOTAL_EPOCHS, num_workers, resume_from_checkpoint=str(self.checkpoint))


@pytest.fixture(scope="module", params=[0, 2], ids=["main-process loading", "two loader workers"])
def arms(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> _Arms:
    # Trained once per module: each arm costs seconds on a processor, and four tests read the same three runs.
    return _Arms(tmp_path_factory.mktemp(f"workers{request.param}"), request.param)


def test_a_resumed_run_lands_bit_for_bit_where_an_unbroken_one_does(arms: _Arms) -> None:
    """Fails on the code before: it has no ``checkpoint_at_epochs``, and its resume restarts the retain loader
    (the control below is that behaviour, and it does not land here)."""
    assert _differing(_adapter(arms.unbroken), _adapter(arms.first)), (
        "the half-way adapter already equals the finished one, so this comparison cannot tell a resume from a no-op"
    )
    different = _differing(_adapter(arms.unbroken), _adapter(arms.resumed))
    assert not different, f"{len(different)} adapter tensors differ between the unbroken and the resumed run: {different[:3]}"
    assert arms.resumed._loss_history == arms.unbroken._loss_history, "the resumed run's loss history is not the unbroken run's"


def test_a_checkpoint_is_written_only_at_the_requested_epochs(arms: _Arms) -> None:
    """Opt-in: the unbroken run asked for nothing and wrote nothing; the first half asked for epochs 1 and 2."""
    assert _checkpoints(arms.root / "unbroken") == []
    assert _checkpoints(arms.root / "first") == ["checkpoint-2", "checkpoint-4"]
    assert (arms.checkpoint / RETAIN_STREAM_FILE).is_file()


def test_without_the_retain_position_the_resume_still_runs_but_is_not_exact(
        arms: _Arms, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
    """The control, and backward compatibility in one: a checkpoint written by the code before has no retain
    position. It still loads, with a warning, and the retain loader starts a fresh pass as it always did --
    which must NOT land on the unbroken run, or the exactness test above could pass on a trainer that
    ignores the retain batches."""
    if arms.first.dataloader_num_workers:
        pytest.skip("one control is enough; it does not depend on the loader workers")
    old_style = tmp_path / "checkpoint-4"
    shutil.copytree(arms.checkpoint, old_style)
    (old_style / RETAIN_STREAM_FILE).unlink()
    with caplog.at_level(logging.WARNING):
        resumed = _fit(tmp_path / "second", TOTAL_EPOCHS, resume_from_checkpoint=str(old_style))
    assert any(RETAIN_STREAM_FILE in record.getMessage() for record in caplog.records)
    assert resumed._loss_history[-1]["step"] == 8
    assert _differing(_adapter(arms.unbroken), _adapter(resumed)), (
        "a resume that restarts the retain loader still matched the unbroken run, so the retain loader's "
        "position is not what the exactness test is testing"
    )


def test_an_epoch_beyond_training_is_a_configuration_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_at_epochs"):
        _fit(tmp_path / "beyond", HALF_EPOCHS, checkpoint_at_epochs=[HALF_EPOCHS + 1])


def test_a_missing_checkpoint_path_raises_instead_of_training_from_scratch(tmp_path: pathlib.Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        _fit(tmp_path / "second", TOTAL_EPOCHS, resume_from_checkpoint=str(tmp_path / "nowhere" / "checkpoint-4"))
