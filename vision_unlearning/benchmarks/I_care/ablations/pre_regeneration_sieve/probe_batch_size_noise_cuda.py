"""Does the batch size change the initial noise, on an NVIDIA card?

The benchmark's generation advances one seeded generator across the whole prompt list, and the
denoiser draws the initial latents once per batch: a call of batch size N consumes a single
tensor of shape ``(N, 4, 64, 64)``. Whether the resulting noise depends on N is a property of
the generator's fill order, not of the model, so it can be measured in isolation.

It matters because the benchmark's two sides were produced at different batch sizes: the shared
baseline at 1, the per-entity images at 25 or 50. If the draw is not decomposable, those two
sides do not share initial noise, and every metric that compares an image with its baseline
pixel for pixel is comparing two different noise realisations.

The control runs first and must pass: two identical draws of the same shape from the same seed
must be identical, otherwise nothing below means anything.

Run on the machine whose card is in question::

    CUDA_VISIBLE_DEVICES=3 python probe_batch_size_noise_cuda.py
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import torch

SHAPE = (4, 64, 64)


def draw(total: int, batch: int, seed: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """``total`` latents drawn from one generator in consecutive calls of ``batch`` each."""
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks: List[torch.Tensor] = []
    drawn = 0
    while drawn < total:
        size = min(batch, total - drawn)
        chunks.append(torch.randn((size, *SHAPE), generator=generator, device=device, dtype=dtype))
        drawn += size
    return torch.cat(chunks, dim=0)


def compare(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    difference = (left.float() - right.float()).abs()
    return {
        'max_abs': float(difference.max()),
        'mean_abs': float(difference.mean()),
        'fraction_differing': float((difference > 0).float().mean()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total', type=int, default=100, help='Latents to draw (entities per task).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 25, 50, 100])
    parser.add_argument('--output', default='probe_batch_size_noise_cuda.json')
    args = parser.parse_args(argv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    report: Dict[str, Any] = {
        'device': device,
        'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'processor',
        'torch': torch.__version__,
        'dtype': str(dtype),
        'total': args.total,
        'seed': args.seed,
        'comparisons': {},
    }
    print(f"device {report['device_name']}  torch {report['torch']}  dtype {dtype}")

    reference = draw(args.total, args.batches[0], args.seed, device, dtype)

    control = compare(reference, draw(args.total, args.batches[0], args.seed, device, dtype))
    report['control_same_batch_twice'] = control
    print(f"CONTROL  batch {args.batches[0]} drawn twice: "
          f"max_abs {control['max_abs']:.6g}  differing {control['fraction_differing'] * 100:.2f}%")
    if control['max_abs'] != 0.0:
        print('CONTROL FAILED -- the generator is not reproducible here; nothing below is meaningful.')
        return 1

    for batch in args.batches[1:]:
        result = compare(reference, draw(args.total, batch, args.seed, device, dtype))
        report['comparisons'][f'{args.batches[0]}_vs_{batch}'] = result
        verdict = 'IDENTICAL' if result['max_abs'] == 0.0 else 'DIFFERENT'
        print(f"batch {args.batches[0]:>3} vs batch {batch:>3}: {verdict:<9} "
              f"max_abs {result['max_abs']:.6g}  mean_abs {result['mean_abs']:.6g}  "
              f"differing {result['fraction_differing'] * 100:.2f}%")

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'BATCH_NOISE_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
