"""Does a resumed ESD run produce the SAME result as an uninterrupted one?

``test_esd_training_state.py`` proves the checkpoint round-trips: weights, optimiser moments and
sampler position come back. That verifies the mechanism EXISTS. It does not verify that it
DISCRIMINATES -- that a run which was interrupted and resumed lands where one that was never
interrupted lands. Those are different claims, and for a session whose purpose is reproducing a
published number the second is the one carrying the value.

The false pass this file exists to prevent: "the resume ran, training continued, the loss looked
plausible". That is exactly how a resume which dropped the optimiser moments, or restarted the depth
sampler, would look. It would be wrong by a small amount, in a direction nobody can see, and it would
never raise.

So the assertion is bit-for-bit equality of the trained tensors between:
  A. N steps, uninterrupted;
  B. N/2 steps, state written, a FRESH unlearner built from that state and run on to N.

Arm B constructs a second ``ESD`` and comes back through the public resume path, so it exercises what
a supervisor restart actually does rather than continuing an object that never died.

The denoiser is a stand-in with the module naming ``_select_parameter_names`` selects on, plus a real
scheduler. The trajectory is under test, not the realism of the model, and a stand-in keeps this on a
processor in seconds -- the only reason a test like this gets run at all.

    python -m pytest tests/test_unlearner/test_esd_resume_equivalence.py -v
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Tuple

import torch
from diffusers import DDIMScheduler

from vision_unlearning.unlearner.esd import ESD

LATENT_CHANNELS = 4
LATENT_SIDE = 8
EMBED_DIM = 16


class _TinyCrossAttention(torch.nn.Module):
    """Named ``attn2`` so ``esd-x`` selects it, exactly as it selects the real cross-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.to_k = torch.nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.to_v = torch.nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = _TinyCrossAttention()


class _TinyUnetConfig:
    in_channels = LATENT_CHANNELS


class _TinyUnet(torch.nn.Module):
    """Deterministic stand-in denoiser whose output depends on the latent, the timestep and the
    trained tensors, so an optimiser step genuinely changes the next prediction."""

    config = _TinyUnetConfig()

    def __init__(self) -> None:
        super().__init__()
        self.block = _TinyBlock()
        self.project = torch.nn.Linear(EMBED_DIM, LATENT_CHANNELS * LATENT_SIDE * LATENT_SIDE, bias=False)
        for parameter in self.project.parameters():
            parameter.requires_grad_(False)

    def forward(self, latent: torch.Tensor, timestep: Any, encoder_hidden_states: torch.Tensor,
                return_dict: bool = True) -> Tuple[torch.Tensor]:
        batch = latent.shape[0]
        context = encoder_hidden_states[:batch].mean(dim=1)
        keyed = self.block.attn2.to_k(context) + self.block.attn2.to_v(context)
        projected = self.project(keyed).view(batch, LATENT_CHANNELS, LATENT_SIDE, LATENT_SIDE)
        scale = timestep.float().mean() / 1000.0 if torch.is_tensor(timestep) else float(timestep) / 1000.0
        return (latent * 0.5 + projected * scale,)


class _StubPipeline:
    """The members ``_fit`` touches, and nothing else."""

    def __init__(self) -> None:
        self.unet = _TinyUnet()
        self.vae = None
        self.text_encoder = None
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.vae_scale_factor = 8

    def encode_prompt(self, *arguments: Any, **keywords: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fixed rather than random: the conditioning must be identical across arms, or the
        # comparison measures the prompts instead of the resume.
        base = torch.linspace(-1.0, 1.0, EMBED_DIM).view(1, 1, EMBED_DIM).repeat(1, 2, 1)
        return base.clone(), (base * -0.5).clone()

    def set_progress_bar_config(self, *arguments: Any, **keywords: Any) -> None:
        return None


def _unlearner(output_dir: str, total_steps: int, **overrides: Any) -> ESD:
    settings: Dict[str, Any] = {
        "erase_concept": "Van Gogh",
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 42,
        "learning_rate": 1e-2,
        "num_train_epochs": total_steps,
        "num_inference_steps": 6,
        "resolution": LATENT_SIDE * 8,
        "train_method": "esd-x",
    }
    settings.update(overrides)
    return ESD(**settings)


def _trained_tensors(unlearner: ESD) -> Dict[str, torch.Tensor]:
    pipeline = _StubPipeline()
    # Identical starting weights in every arm, or nothing below means anything.
    torch.manual_seed(1234)
    for parameter in pipeline.unet.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    return unlearner._fit(pipeline)  # type: ignore[arg-type]


class _Killed(Exception):
    """Stands in for the process being killed, the instant a checkpoint has been written."""


def test_a_resumed_run_lands_bit_for_bit_where_an_uninterrupted_one_does() -> None:
    total, half = 8, 4
    with tempfile.TemporaryDirectory() as straight_dir, tempfile.TemporaryDirectory() as resumed_dir:
        straight = _trained_tensors(_unlearner(straight_dir, total))

        # Arm B is killed AT the checkpoint, the worst moment for resume correctness and the one
        # actually observed in production. total_steps stays at `total`: a real kill does not change
        # the configuration, and an earlier version of this test that shortened it was correctly
        # REFUSED by the resume guard -- the guard working, not the resume failing.
        first_half = _unlearner(resumed_dir, total, checkpoint_every_steps=half)
        original_save = first_half._save_training_state

        def save_then_die(*arguments: Any, **keywords: Any) -> None:
            original_save(*arguments, **keywords)
            raise _Killed

        setattr(first_half, "_save_training_state", save_then_die)
        try:
            _trained_tensors(first_half)
        except _Killed:
            pass
        else:
            raise AssertionError("arm B was never interrupted, so it is not testing a resume")
        assert os.path.exists(os.path.join(resumed_dir, "esd_training_state.pt")), \
            "arm B wrote no checkpoint, so it is not testing a resume at all"

        second_half = _unlearner(resumed_dir, total, checkpoint_every_steps=half,
                                 resume_from_training_state=True)
        resumed = _trained_tensors(second_half)

    assert set(straight) == set(resumed)
    for name in straight:
        assert torch.equal(straight[name], resumed[name]), (
            f"{name} differs after a resume: max |difference| "
            f"{(straight[name] - resumed[name]).abs().max().item():.3e}. The resume runs but does not "
            f"reproduce, which is the silent failure this test exists for."
        )


def test_the_comparison_can_actually_fail() -> None:
    """The control. If two genuinely different trajectories compared equal, the test above is empty."""
    with tempfile.TemporaryDirectory() as one_dir, tempfile.TemporaryDirectory() as other_dir:
        eight = _trained_tensors(_unlearner(one_dir, 8))
        seven = _trained_tensors(_unlearner(other_dir, 7))
    assert any(not torch.equal(eight[name], seven[name]) for name in eight), \
        "7 steps and 8 steps gave identical tensors, so this comparison cannot detect anything"
