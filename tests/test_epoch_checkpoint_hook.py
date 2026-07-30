"""Tests for the optional per-epoch LoRA checkpoint hook on UnlearnerLora.

Covers the scheduling helper, the field validator, the train-start range guard, and the
non-mutating intermediate saver. The scheduler, validator and guard need no GPU; the saver
test builds a tiny CPU LoRA module (no Stable Diffusion download) and asserts the save does
not perturb training state.
"""
import os

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from pydantic import ValidationError

from vision_unlearning.unlearner.lora import UnlearnerLora, epochs_to_save
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple


class _MinimalUnlearnerLora(UnlearnerLora):
    """Smallest concrete UnlearnerLora: implements the two abstract methods so the base
    class (which carries the checkpoint hook) can be instantiated for unit tests."""

    def _prepare_dataloaders(self):  # type: ignore[override]
        raise NotImplementedError

    def _train_one_batch(self, batch_forget, batch_retain):  # type: ignore[override]
        raise NotImplementedError


def _make(**overrides):
    kwargs = dict(
        model_name_or_path="dummy",
        dataset_forget_name="forget",
        dataset_retain_name="retain",
        gradient_weighting_method=GradientWeightingMethodSimple(),
    )
    kwargs.update(overrides)
    return _MinimalUnlearnerLora(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Scheduling helper (no GPU)                                                   #
# --------------------------------------------------------------------------- #
def test_epochs_to_save_none_never_fires():
    assert epochs_to_save(None, 1) is False
    assert epochs_to_save(None, 50) is False


def test_epochs_to_save_is_one_based():
    requested = [1, 3, 5]
    assert epochs_to_save(requested, 0) is False  # 0-based epoch index never matches
    assert epochs_to_save(requested, 1) is True
    assert epochs_to_save(requested, 2) is False
    assert epochs_to_save(requested, 3) is True
    assert epochs_to_save(requested, 5) is True
    assert epochs_to_save(requested, 6) is False


# --------------------------------------------------------------------------- #
# Field validator (no GPU)                                                     #
# --------------------------------------------------------------------------- #
def test_validator_default_is_none():
    assert _make().save_lora_at_epochs is None


def test_validator_dedup_and_sort():
    unlearner = _make(save_lora_at_epochs=[2, 1, 2, 5])
    assert unlearner.save_lora_at_epochs == [1, 2, 5]


@pytest.mark.parametrize("bad", [
    [0],          # below 1
    [-1],         # negative
    [3, 0],       # one below 1
    [True],       # bool is not an int here
    [False],
    [1.5],        # float
    [2.0],        # whole-valued float still rejected
    ["1"],        # numeric string
    [1, "2"],
    "notalist",   # not a list
    5,            # not a list
])
def test_validator_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        _make(save_lora_at_epochs=bad)


# --------------------------------------------------------------------------- #
# Train-start range guard (no GPU)                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("epochs,num_train_epochs,should_raise", [
    ([1, 5], 5, False),
    ([1, 5], 4, True),      # 5 > 4
    ([50], 50, False),
    ([51], 50, True),
    (None, 3, False),
])
def test_assert_save_epochs_within_training(epochs, num_train_epochs, should_raise):
    unlearner = _make(save_lora_at_epochs=epochs, num_train_epochs=num_train_epochs)
    if should_raise:
        with pytest.raises(ValueError):
            unlearner._assert_save_epochs_within_training()
    else:
        unlearner._assert_save_epochs_within_training()


# --------------------------------------------------------------------------- #
# Non-mutating intermediate saver (CPU torch, no Stable Diffusion)             #
# --------------------------------------------------------------------------- #
class _TinyLoraModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)

    def forward(self, x):
        return self.to_q(x)


def _tiny_peft_model():
    base = _TinyLoraModule()
    config = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"])
    return get_peft_model(base, config)


def test_save_lora_layers_to_writes_file_and_does_not_mutate(tmp_path):
    unlearner = _make()
    unlearner._unet = _tiny_peft_model()  # type: ignore[assignment]
    unlearner._accelerator = Accelerator()

    # Snapshot training-relevant state before the save.
    a_lora_param = next(p for n, p in unlearner._unet.named_parameters() if "lora_A" in n)
    before_tensor = a_lora_param.detach().clone()
    before_dtype = a_lora_param.dtype
    before_mode = unlearner._unet.training
    before_rng = torch.get_rng_state()

    target_dir = str(tmp_path / "epoch-1")
    unlearner._save_lora_layers_to(target_dir)

    assert os.path.isfile(os.path.join(target_dir, "pytorch_lora_weights.safetensors"))

    # The intermediate save must not perturb subsequent training.
    after_param = next(p for n, p in unlearner._unet.named_parameters() if "lora_A" in n)
    assert after_param.dtype == before_dtype
    assert unlearner._unet.training == before_mode
    assert torch.equal(after_param.detach(), before_tensor)
    assert torch.equal(torch.get_rng_state(), before_rng)
