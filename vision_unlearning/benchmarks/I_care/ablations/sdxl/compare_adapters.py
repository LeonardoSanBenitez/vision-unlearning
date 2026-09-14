'''
Compares two saved LoRA adapters tensor by tensor and prints how far apart they are.

A whole-file hash answers "identical or not". On this machine that answer is always "not", because
the training step is not bit-reproducible on the GPU. This script answers the useful question
instead: *how far apart*, in absolute and relative terms. Two uses:

- **Measuring the oracle's noise floor.** Run the training step twice with unchanged code and
  compare the two adapters. Whatever difference appears is what the hardware contributes, and it is
  the threshold any real regression has to beat to be visible.
- **Checking a refactor.** Compare a post-refactor adapter against a pre-refactor one. A difference
  at the noise floor means the refactor changed nothing that matters; a difference orders of
  magnitude above it means it did.

    python compare_adapters.py A.safetensors B.safetensors

It prints the comparison, never a verdict: per-tensor and overall maxima, the relative scale they
should be read against, and the count of tensors that differ at all.
'''
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file


def load_adapter(path: Path) -> Dict[str, torch.Tensor]:
    tensors = load_file(str(path))
    return {name: value.to(torch.float32) for name, value in tensors.items()}


def compare(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> List[Tuple[str, float, float, float]]:
    '''
    Returns one row per tensor: name, max absolute difference, max absolute value, and the
    difference expressed as a fraction of that magnitude.

    The third and fourth columns matter because an absolute difference is meaningless without the
    scale of the thing it is a difference of: 1e-7 is noise on a weight of 0.1 and a rewrite on a
    weight of 1e-7.
    '''
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))
        only_right = sorted(set(right) - set(left))
        raise SystemExit(
            f"adapters have different tensors, which is a structural difference rather than a "
            f"numerical one.\nonly in first: {only_left}\nonly in second: {only_right}"
        )

    rows = []
    for name in sorted(left):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            raise SystemExit(f"tensor {name} has shape {tuple(a.shape)} against {tuple(b.shape)}")
        difference = float((a - b).abs().max())
        magnitude = float(a.abs().max())
        relative = difference / magnitude if magnitude > 0 else 0.0
        rows.append((name, difference, magnitude, relative))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two LoRA adapters tensor by tensor.")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--show", type=int, default=5, help="how many of the most-different tensors to list")
    args = parser.parse_args()

    left = load_adapter(Path(args.first))
    right = load_adapter(Path(args.second))
    rows = compare(left, right)

    n_tensors = len(rows)
    n_differing = sum(1 for _, difference, _, _ in rows if difference > 0)
    max_absolute = max(difference for _, difference, _, _ in rows)
    max_relative = max(relative for _, _, _, relative in rows)

    print("")
    print(f"first           : {args.first}")
    print(f"second          : {args.second}")
    print(f"tensors         : {n_tensors}")
    print(f"tensors differing: {n_differing} of {n_tensors}")
    print(f"max absolute difference : {max_absolute:.6e}")
    print(f"max relative difference : {max_relative:.6e}")
    print("")
    print(f"the {args.show} most-different tensors (name, absolute, magnitude, relative):")
    for name, difference, magnitude, relative in sorted(rows, key=lambda row: -row[3])[:args.show]:
        print(f"  {name}  {difference:.6e}  {magnitude:.6e}  {relative:.6e}")


if __name__ == "__main__":
    main()
