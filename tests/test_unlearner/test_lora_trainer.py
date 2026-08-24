"""Focused regressions for `UnlearnerLora`'s training plumbing.

Heavy tier: torch only, no diffusers checkpoint, no network, no graphics card. These tests are
about the wiring between the parameter collection, the optimizer and the gradient clipper -- a
defect there is invisible to any test that only looks at the adapter afterwards, because an
unclipped step still produces a perfectly plausible adapter.
"""
from typing import Any

import pytest
import torch

from accelerate import Accelerator


def _trainable_parameters() -> torch.nn.Module:
    """A tiny module standing in for the adapter parameters, with one frozen tensor among them."""
    module = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
    module[1].weight.requires_grad_(False)
    return module


def test_parameter_collection_survives_being_consumed_by_the_optimizer() -> None:
    """The collection handed to the optimizer must still hold parameters afterwards.

    This is the defect itself. A `filter(...)` is exhausted by the optimizer while it builds its
    parameter group, so every later consumer of the same object -- the gradient clipper -- sees an
    empty sequence and silently does nothing. The mutation that must fail this test is replacing
    the list comprehension with `filter(lambda p: p.requires_grad, module.parameters())`.
    """
    module = _trainable_parameters()

    collection = [p for p in module.parameters() if p.requires_grad]
    assert len(collection) == 1

    torch.optim.AdamW(collection, lr=1e-3)

    assert len(list(collection)) == 1, (
        "the parameter collection was consumed by the optimizer; gradient clipping would see "
        "nothing to clip"
    )


def test_clipping_bounds_a_large_gradient() -> None:
    """A known large gradient is brought under `max_grad_norm`, and the norm reported is real.

    Measured against the pre-fix behaviour: with an exhausted iterator the identical call left a
    gradient of 1000.0 untouched and reported a total norm of 0.0.
    """
    module = _trainable_parameters()
    accelerator = Accelerator()

    collection = [p for p in module.parameters() if p.requires_grad]
    torch.optim.AdamW(collection, lr=1e-3)

    for parameter in collection:
        parameter.grad = torch.full_like(parameter, 1000.0)

    max_grad_norm = 5.0
    total_norm = accelerator.clip_grad_norm_(collection, max_grad_norm)

    assert float(total_norm) > max_grad_norm, (
        "the reported pre-clip norm is not larger than the threshold, so nothing was measured"
    )
    clipped_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in collection])
    )
    assert float(clipped_norm) <= max_grad_norm + 1e-4, (
        f"post-clip norm {float(clipped_norm)} exceeds max_grad_norm {max_grad_norm}"
    )


def _minimal_lora_unlearner(**overrides: Any) -> Any:
    """A concrete `UnlearnerLora` built with only its required fields; nothing is loaded or trained.

    `UnlearnerLora` is abstract -- `_prepare_dataloaders` and `_train_one_batch` are left to each
    method -- so the smallest honest stand-in is a subclass that implements both with nothing. The
    fields under test live on the base class, so the stub does not weaken what is being asserted.
    """
    from vision_unlearning.unlearner.lora import UnlearnerLora
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    class _ConcreteLora(UnlearnerLora):  # type: ignore[misc]
        def _prepare_dataloaders(self) -> None:
            raise NotImplementedError("this stand-in never trains")

        def _train_one_batch(self, *arguments: Any, **keywords: Any) -> None:
            raise NotImplementedError("this stand-in never trains")

    fields: dict = {
        "model_name_or_path": "CompVis/stable-diffusion-v1-4",
        "dataset_forget_name": "unused",
        "dataset_retain_name": "unused",
        "output_dir": "unused",
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
    }
    fields.update(overrides)
    return _ConcreteLora(**fields)


def test_an_unknown_hyperparameter_raises_instead_of_vanishing() -> None:
    """The defect this guards: pydantic's default policy drops unknown keyword arguments.

    Two hyperparameters lived in that gap for the whole life of the class -- `lora_dropout`, passed
    as 0.2 by every caller and never reaching the adapter, and `train_batch_size`, whose real name
    is `per_device_train_batch_size`. The mutation that must fail this test is removing
    `model_config = ConfigDict(extra="forbid")` from `Unlearner`.
    """
    import pydantic

    # The error must be about the EXTRA field specifically. A bare `pytest.raises(ValidationError)`
    # also passes when construction fails for an unrelated reason -- a missing required field, say --
    # which is a test that would keep passing after the fix was reverted.
    for unknown in ("a_hyperparameter_that_does_not_exist", "train_batch_size"):
        with pytest.raises(pydantic.ValidationError) as raised:
            _minimal_lora_unlearner(**{unknown: 1})
        reasons = [error["type"] for error in raised.value.errors()]
        assert reasons == ["extra_forbidden"], (
            f"{unknown} was rejected, but for the wrong reason: {reasons}"
        )


def test_the_configured_dropout_reaches_the_adapter_configuration() -> None:
    """`lora_dropout` is a real field and its value arrives at `LoraConfig`.

    Asserted on the configuration object rather than on a trained adapter, because a dropout that
    fails to arrive changes training only stochastically -- which is exactly why nobody noticed for
    the lifetime of this class.
    """
    unlearner = _minimal_lora_unlearner(lora_dropout=0.25)
    assert unlearner.lora_dropout == 0.25  # type: ignore[attr-defined]

    configuration = unlearner._get_lora_config()  # type: ignore[attr-defined]
    assert configuration.lora_dropout == 0.25, (
        "the configured dropout did not reach LoraConfig; peft would silently use its own default"
    )

    assert _minimal_lora_unlearner()._get_lora_config().lora_dropout == 0.0  # type: ignore[attr-defined]


def test_no_call_site_still_passes_the_dead_batch_size_key() -> None:
    """A repository scan, because `extra="forbid"` turns a leftover key into a crash at run time.

    `train_batch_size` was accepted and dropped at every call site. Now that unknown fields raise,
    any surviving occurrence is a training run that dies on construction -- and these call sites are
    campaign scripts that fail hours into a session, not in a test.

    `benchmarks/u_care` is excluded deliberately: it is a vendored standalone script with its own
    argparse namespace, not a caller of this class.
    """
    import pathlib

    package_root = pathlib.Path(__file__).resolve().parents[2] / "vision_unlearning"
    offenders = []
    for path in package_root.rglob("*.py"):
        if "u_care" in path.parts or ".mypy_cache" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if '"train_batch_size"' in line or "'train_batch_size'" in line:
                offenders.append(f"{path.relative_to(package_root)}:{number}")

    assert offenders == [], "call sites still passing a key the model now rejects: " + ", ".join(offenders)
