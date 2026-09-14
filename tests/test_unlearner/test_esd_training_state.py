"""The ESD trainer's periodic training state, and that a resume continues the same run.

A paper-configuration ESD session is 1000 optimizer steps and about twelve hours on one card, and
until this state existed it wrote nothing until the very end: a session killed at step 759 lost all
of it. These tests cover the mechanism that fixes that, and specifically the part that would be
silently wrong rather than loudly broken -- a resume that restores the weights but not the optimizer
moments or the depth sampler would carry on from the right place along a different trajectory.

Not collected by the repository's pytest configuration if that is scoped elsewhere; run directly::

    python -m pytest tests/test_esd_training_state.py -v
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import pytest
import torch

from vision_unlearning.unlearner.esd import ESD


def _unlearner(output_dir: str, **overrides: Any) -> ESD:
    settings: Dict[str, Any] = {
        "erase_concept": "Van Gogh",
        "output_dir": output_dir,
        "seed": 42,
        "learning_rate": 1e-5,
        "num_train_epochs": 100,
        "resume_from_training_state": True,
    }
    settings.update(overrides)
    return ESD(**settings)


def _trainable() -> Dict[str, torch.nn.Parameter]:
    torch.manual_seed(0)
    return {
        "down.attn.to_k.weight": torch.nn.Parameter(torch.randn(4, 3)),
        "down.attn.to_v.weight": torch.nn.Parameter(torch.randn(2)),
    }


def _advance(optimizer: torch.optim.Optimizer, trainable: Dict[str, torch.nn.Parameter],
             steps: int) -> None:
    """Take real optimizer steps, so Adam accumulates moments there is something to restore."""
    for _ in range(steps):
        optimizer.zero_grad()
        loss = sum((parameter ** 2).sum() for parameter in trainable.values())
        loss.backward()
        optimizer.step()


@pytest.fixture()
def saved_state(tmp_path: Any) -> Dict[str, Any]:
    """A state written after 7 completed steps, plus what the run would have done next."""
    output_dir = str(tmp_path)
    unlearner = _unlearner(output_dir, checkpoint_every_steps=5)

    trainable = _trainable()
    optimizer = torch.optim.Adam(list(trainable.values()), lr=1e-5)
    _advance(optimizer, trainable, 2)

    sampler = random.Random(42)
    for _ in range(7):
        sampler.randint(0, 49)
    # What the uninterrupted run would draw next, captured before saving.
    continuation = random.Random(42)
    for _ in range(7):
        continuation.randint(0, 49)
    expected_next = [continuation.randint(0, 49) for _ in range(5)]

    unlearner._save_training_state(7, trainable, optimizer, sampler)
    return {
        "output_dir": output_dir,
        "trainable": trainable,
        "optimizer": optimizer,
        "expected_next": expected_next,
        "path": unlearner._training_state_path(),
    }


def test_state_is_written_atomically(saved_state: Dict[str, Any]) -> None:
    """The file exists under its final name and no temporary is left behind.

    The temporary-and-rename is the whole point: this file is written precisely because the process
    is killed without warning, and a half-written state would be loaded on the next start.
    """
    assert os.path.exists(saved_state["path"])
    assert not os.path.exists(saved_state["path"] + ".partial")


def test_resume_restores_weights_moments_and_sampler(saved_state: Dict[str, Any]) -> None:
    """All four pieces of the run come back: step, weights, Adam moments, sampler position."""
    trainable = _trainable()
    for parameter in trainable.values():
        torch.nn.init.zeros_(parameter)
    optimizer = torch.optim.Adam(list(trainable.values()), lr=1e-5)
    sampler = random.Random(0)

    first_step = _unlearner(saved_state["output_dir"])._load_training_state(
        trainable, optimizer, sampler,
    )

    assert first_step == 7
    for name, parameter in saved_state["trainable"].items():
        assert torch.equal(parameter.detach(), trainable[name].detach()), name

    reference = saved_state["optimizer"].state_dict()["state"]
    restored = optimizer.state_dict()["state"]
    assert reference.keys() == restored.keys()
    for key in reference:
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(reference[key][moment], restored[key][moment]), moment

    # The sampler draws both the denoising depth and the sampling noise, so continuing it is what
    # makes the remaining steps the steps this run would have taken.
    assert [sampler.randint(0, 49) for _ in range(5)] == saved_state["expected_next"]


@pytest.mark.parametrize("overrides,reason", [
    ({"seed": 7}, "seed"),
    ({"learning_rate": 9e-9}, "learning rate"),
    ({"num_train_epochs": 55}, "total steps"),
])
def test_resume_refuses_across_a_changed_configuration(
    saved_state: Dict[str, Any], overrides: Dict[str, Any], reason: str,
) -> None:
    """Resuming across a changed trajectory setting would match neither configuration."""
    unlearner = _unlearner(saved_state["output_dir"], **overrides)
    with pytest.raises(ValueError, match="Refusing to resume"):
        unlearner._load_training_state(
            _trainable(), torch.optim.Adam(list(_trainable().values()), lr=1e-5), random.Random(0),
        )


def test_resume_refuses_when_the_selected_tensors_differ(saved_state: Dict[str, Any]) -> None:
    """A changed train_method or base checkpoint selects different tensors; that is not a resume."""
    different = {"up.attn.to_q.weight": torch.nn.Parameter(torch.randn(3))}
    with pytest.raises(ValueError, match="Refusing to resume"):
        _unlearner(saved_state["output_dir"])._load_training_state(
            different, torch.optim.Adam(list(different.values()), lr=1e-5), random.Random(0),
        )


def test_disabled_resume_ignores_an_existing_state(saved_state: Dict[str, Any]) -> None:
    """The flag is off by default, and off means start from zero even with a state file present."""
    trainable = _trainable()
    first_step = _unlearner(
        saved_state["output_dir"], resume_from_training_state=False,
    )._load_training_state(
        trainable, torch.optim.Adam(list(trainable.values()), lr=1e-5), random.Random(0),
    )
    assert first_step == 0


def test_no_state_file_is_not_an_error(tmp_path: Any) -> None:
    """The ordinary case: nothing to resume, so training starts at step 0."""
    trainable = _trainable()
    first_step = _unlearner(str(tmp_path))._load_training_state(
        trainable, torch.optim.Adam(list(trainable.values()), lr=1e-5), random.Random(0),
    )
    assert first_step == 0
