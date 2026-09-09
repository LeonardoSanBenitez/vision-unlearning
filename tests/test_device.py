"""Regressions for the guarded device instrumentation.

Heavy tier: torch only, no checkpoint, no network, no graphics card. Each test states the mutation
that must fail it, because a guard that has never been shown to do anything is indistinguishable
from a guard that does nothing.
"""
from typing import Any

import pytest
import torch

from vision_unlearning.utils import device as device_utils


def _pretend_context_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state this whole module exists for: a visible card whose context does not exist yet.

    `is_available()` returns True and `is_initialized()` returns False, which is what a fresh
    interpreter on the ROCm build reports before anything has touched the device. The guarded calls
    below must all be inert in that state.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def test_the_peak_memory_reset_is_skipped_while_the_context_is_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """`reset_peak_memory_stats` raises in that state, and it is the first device call on the
    training path -- so an unguarded one aborts training before the first step.

    The mutation that must fail this test is calling `torch.cuda.reset_peak_memory_stats()` directly
    instead of the guarded wrapper.
    """
    _pretend_context_is_missing(monkeypatch)

    def _explode() -> None:
        raise RuntimeError("Invalid device argument")

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *arguments: _explode())

    device_utils.reset_peak_memory_stats()


def test_the_memory_reading_is_zero_rather_than_an_error_without_a_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-step peak reading has to return a number the caller can take a maximum of."""
    _pretend_context_is_missing(monkeypatch)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *arguments: 1 / 0)

    assert device_utils.max_memory_allocated() == 0
    device_utils.synchronize()
    device_utils.empty_cache()


def test_availability_alone_is_not_treated_as_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the module: `is_available()` is True in the state that raises.

    A guard written as `if torch.cuda.is_available():` -- the obvious one, and the one the plan
    originally specified -- passes here and lets the call through. That is the mutation.
    """
    _pretend_context_is_missing(monkeypatch)

    assert torch.cuda.is_available() is True
    assert device_utils.is_context_ready() is False


def test_a_live_context_is_used_rather_than_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard that is always false would satisfy every test above and instrument nothing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    calls: list = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *arguments: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *arguments: calls.append("empty"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *arguments: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *arguments: 4096)

    device_utils.reset_peak_memory_stats()
    device_utils.empty_cache()
    device_utils.synchronize()
    reading: Any = device_utils.max_memory_allocated()

    assert calls == ["reset", "empty", "sync"]
    assert reading == 4096
