"""Executable evidence for three claims about the current SPARE trainer.

The SDXL work rests on a comparison between ``UnlearnerSpare`` (this repository's SPARE
implementation, on Stable Diffusion 1.4) and a proof-of-concept SDXL training script. Three of the
differences that comparison found are claims about what our own code *does at run time*, not about
what it says, and each one would be a reasonable thing to get wrong from reading alone. This script
reproduces each mechanism in isolation, on the processor, in a few seconds, so the comparison cites a
measurement instead of an interpretation.

The three claims:

1. **Gradient clipping is a no-op.** ``UnlearnerLora.train`` assigns a lazy ``filter`` object to
   ``self._lora_layers`` and then hands that same object to the optimizer, which iterates it to
   collect the parameter group. The training step later clips ``self._lora_layers`` -- by then an
   exhausted iterator, so clipping sees no parameters and ``max_grad_norm`` never takes effect.
2. **The retain set is sampled, not iterated.** The step loop calls ``next(iter(retain_loader))``,
   which builds a new iterator on every call, so it always yields the first batch of a freshly
   shuffled pass. Over an epoch the retain images are drawn with replacement in batch-sized groups
   rather than each being visited once.
3. **Unknown hyperparameters are silently accepted.** ``Unlearner`` is a plain pydantic model with no
   ``extra`` configuration, so a mistyped or non-existent field passed by a caller is dropped without
   an error.

Run with the interpreter that has torch (no need for the package to be installed)::

    "$PY" check_implementation_differences.py

It writes ``assets/check_implementation_differences.json`` and prints ``CHECKS_OK`` only when every
claim came out as stated; a claim that comes out the other way is printed as ``UNEXPECTED`` and the
exit code is non-zero, because that would mean the comparison needs correcting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import torch
from pydantic import BaseModel

_OUT = Path(__file__).resolve().parent / "assets"


def claim_gradient_clipping_is_a_no_op() -> Dict[str, Any]:
    """Reproduce the exhausted-iterator pattern of ``UnlearnerLora.train`` exactly."""
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    # Sequential.__getitem__ is typed as returning Module, which has no weight or bias.
    first = cast(torch.nn.Linear, module[0])
    # Stand-in for the adapter parameters: the only trainable tensors.
    trainable = list(first.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)

    # The two lines under test, transcribed from lora.py: a lazy filter, then the optimizer.
    lora_layers = filter(lambda p: p.requires_grad, module.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=1e-4)
    parameters_seen_by_optimizer = sum(len(group["params"]) for group in optimizer.param_groups)

    # And the line in fade.py's training step: params_to_clip = self._lora_layers
    remaining = list(lora_layers)
    first.weight.grad = torch.full_like(first.weight, 1000.0)
    first.bias.grad = torch.full_like(first.bias, 1000.0)
    total_norm_reported = float(torch.nn.utils.clip_grad_norm_(remaining, max_norm=1.0))
    largest_gradient_after_clipping = float(first.weight.grad.abs().max())

    # The same clipping against a list, which is what the proof of concept does.
    total_norm_with_list = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    largest_gradient_after_list_clipping = float(first.weight.grad.abs().max())

    return {
        "claim": "gradient clipping over the exhausted filter clips nothing",
        "parameters_seen_by_optimizer": parameters_seen_by_optimizer,
        "parameters_remaining_for_clipping": len(remaining),
        "reported_total_norm_over_exhausted_iterator": total_norm_reported,
        "largest_gradient_after_clipping_exhausted": largest_gradient_after_clipping,
        "reported_total_norm_over_list": total_norm_with_list,
        "largest_gradient_after_clipping_list": largest_gradient_after_list_clipping,
        "as_stated": (
            parameters_seen_by_optimizer == 2
            and len(remaining) == 0
            and largest_gradient_after_clipping == 1000.0
            and largest_gradient_after_list_clipping < 1000.0
        ),
    }


def claim_retain_set_is_sampled_not_iterated() -> Dict[str, Any]:
    """``next(iter(loader))`` in a loop returns first batches of fresh passes, never advances."""
    data = torch.arange(20).unsqueeze(1).float()
    # A tensor is an indexable, sized object, so DataLoader accepts it directly; the annotation
    # only tells mypy what the loader yields, since the Dataset protocol is what it expects.
    loader: "torch.utils.data.DataLoader[torch.Tensor]" = torch.utils.data.DataLoader(
        cast(Any, data), batch_size=4, shuffle=True,
    )

    torch.manual_seed(0)
    fresh_iterator_batches: List[List[int]] = []
    for _ in range(5):
        batch = next(iter(loader))                      # the pattern in lora.py's step loop
        fresh_iterator_batches.append([int(v) for v in batch.flatten()])

    torch.manual_seed(0)
    single_iterator_batches: List[List[int]] = []
    iterator = iter(loader)                             # what iterating the loader once looks like
    for _ in range(5):
        batch = next(iterator)
        single_iterator_batches.append([int(v) for v in batch.flatten()])

    flat_fresh = [value for batch in fresh_iterator_batches for value in batch]
    flat_single = [value for batch in single_iterator_batches for value in batch]
    return {
        "claim": "next(iter(loader)) draws with replacement; iterating the loader visits each item once",
        "batches_from_fresh_iterators": fresh_iterator_batches,
        "batches_from_one_iterator": single_iterator_batches,
        "distinct_items_seen_fresh_iterators": len(set(flat_fresh)),
        "distinct_items_seen_one_iterator": len(set(flat_single)),
        "dataset_size": len(data),
        "as_stated": len(set(flat_single)) == 20 and len(set(flat_fresh)) < 20,
    }


def claim_unknown_hyperparameters_are_ignored() -> Dict[str, Any]:
    """A plain pydantic model drops unknown keyword arguments instead of raising."""
    class Minimal(BaseModel):
        learning_rate: float = 1e-4

    instance = Minimal(learning_rate=6e-4, lora_dropout=0.2, train_batch_size=4)  # type: ignore[call-arg]
    fields = sorted(instance.model_dump().keys())
    return {
        "claim": "pydantic's default extra policy silently ignores unknown fields",
        "constructed_without_error": True,
        "fields_kept": fields,
        "lora_dropout_kept": "lora_dropout" in fields,
        "as_stated": fields == ["learning_rate"],
    }


def main() -> int:
    checks = [
        claim_gradient_clipping_is_a_no_op(),
        claim_retain_set_is_sampled_not_iterated(),
        claim_unknown_hyperparameters_are_ignored(),
    ]
    payload = {"torch": torch.__version__, "checks": checks}
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / "check_implementation_differences.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"WROTE {path}")

    failed = [check["claim"] for check in checks if not check["as_stated"]]
    if failed:
        for claim in failed:
            print(f"UNEXPECTED: {claim}")
        return 1
    print("CHECKS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
