"""Focused regressions for `UnlearnerLora`'s training plumbing.

Heavy tier: torch only, no diffusers checkpoint, no network, no graphics card. These tests are
about the wiring between the parameter collection, the optimizer and the gradient clipper -- a
defect there is invisible to any test that only looks at the adapter afterwards, because an
unclipped step still produces a perfectly plausible adapter.
"""
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
