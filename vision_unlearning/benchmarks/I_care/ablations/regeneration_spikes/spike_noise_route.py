"""SPIKE -- where should the initial noise of a regenerated image come from? Not evidence.

A planning probe for the full regeneration of the I-CARE corpus. Nothing written by this script may
be cited as the result of any stage; it only tells the plan which design to adopt. The real campaign
re-measures whatever it relies on.

Three questions, all about the initial latent that Stable Diffusion 1.4 starts denoising from:

1. **Does the batch size still change an image once the noise is drawn on the processor?** The
   benchmark currently draws its latents on the graphics card, where a batch of 8 and a batch of 16
   fill the noise tensor differently, so the batch size decides the picture. On the processor the
   draw decomposes. What is not known is whether the denoiser itself, run in half precision with
   deterministic kernels, returns the same pixels for one image computed alone and the same image
   computed inside a batch.
2. **Does the card model still change an image once the noise is drawn on the processor?** An A40
   and a TITAN RTX given the same stored prompt list at the same seed and batch size disagree on
   half of every batch today, because a card draw depends on the launch configuration. With the
   draw moved to the processor only the denoiser's arithmetic can differ; this measures by how much.
3. **What does a larger batch buy on an A40?** Throughput was measured at batch 1, 4 and 8 only.

Two noise schemes are generated, both with a processor-side ``torch.Generator``:

* ``per_image`` -- a fresh generator seeded to the same value for every image, so every entity at a
  given seed starts from the same latent (the scheme under consideration for the new corpus).
* ``stream`` -- one generator per seed, advanced across the prompt list in order (the benchmark's
  current convention, moved to the processor).

Usage, from the repository root, one card at a time (``CUBLAS_WORKSPACE_CONFIG=:4096:8`` must be in
the environment before Python starts)::

    PYTHONPATH=. python .../spike_noise_route.py generate --label a40 --out DIR
    PYTHONPATH=. python .../spike_noise_route.py generate --label titan --out DIR
    PYTHONPATH=. python .../spike_noise_route.py throughput --label a40 --out DIR
    python .../spike_noise_route.py compare --out DIR --cards a40 titan

``generate`` writes ``DIR/images_<label>.npz`` (uint8 images, one array per run key) and
``DIR/timing_<label>.json``. ``throughput`` writes ``DIR/throughput_<label>.json``. ``compare``
reads the ``.npz`` files of the named cards and writes ``DIR/noise_route_summary.json``: for every
comparison, how many of the images are pixel-identical and the mean and largest absolute pixel
difference on the 0-255 scale, beside two floors computed from the same images -- two different
prompts at one seed, and one prompt at two seeds -- which say how large a difference has to be to
mean "a different picture".

Exit codes: 0 finished and wrote its file; 2 a usage or data error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

MODEL_ID = 'CompVis/stable-diffusion-v1-4'
PROMPTS_PER_TASK = 8
TASKS: Tuple[Literal['people'], Literal['breeds']] = ('people', 'breeds')


def load_prompts(base_folder: Path) -> List[str]:
    """The first eight generation prompts of people and of breeds, in metadata order."""
    from vision_unlearning.datasets.entity_names import generation_prompt

    prompts: List[str] = []
    for task in TASKS:
        with (base_folder / f'metadata_{task}_2_enriched_filtered.json').open(encoding='utf-8') as handle:
            metadata = json.load(handle)
        prompts.extend(generation_prompt(task, entry['name']) for entry in metadata[:PROMPTS_PER_TASK])
    return prompts


def load_pipeline() -> Any:
    """Stable Diffusion 1.4 exactly as the benchmark's generation builds it."""
    import torch
    from diffusers import AutoPipelineForText2Image

    pipeline = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipeline = pipeline.to('cuda')
    pipeline.set_progress_bar_config(disable=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return pipeline


def generate_run(pipeline: Any, prompts: List[str], seed: int, batch_size: int, scheme: str) -> Tuple[np.ndarray, float]:
    """All prompts at one seed, in order, in batches. Returns uint8 images and seconds spent."""
    import torch

    stream = torch.Generator(device='cpu').manual_seed(seed)
    images: List[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        generator: Any
        if scheme == 'per_image':
            generator = [torch.Generator(device='cpu').manual_seed(seed) for _ in chunk]
        elif scheme == 'stream':
            generator = stream
        else:
            raise ValueError(f'unknown scheme {scheme!r}')
        output = pipeline(chunk, generator=generator).images
        images.extend(np.asarray(image, dtype=np.uint8) for image in output)
    torch.cuda.synchronize()
    return np.stack(images), time.perf_counter() - started


RUNS: List[Tuple[str, int, int]] = [
    # (scheme, seed, batch size)
    ('per_image', 42, 1),
    ('per_image', 42, 8),
    ('per_image', 42, 16),
    ('per_image', 43, 8),
    ('stream', 42, 1),
    ('stream', 42, 8),
    ('stream', 42, 16),
]


def run_key(scheme: str, seed: int, batch_size: int) -> str:
    return f'{scheme}_s{seed}_b{batch_size}'


def command_generate(args: argparse.Namespace) -> int:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.base_folder))
    pipeline = load_pipeline()
    arrays: Dict[str, np.ndarray] = {}
    timing: Dict[str, Any] = {'card': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'prompts': prompts, 'runs': {}}
    for scheme, seed, batch_size in RUNS:
        key = run_key(scheme, seed, batch_size)
        images, seconds = generate_run(pipeline, prompts, seed, batch_size, scheme)
        arrays[key] = images
        timing['runs'][key] = {'seconds': round(seconds, 3), 'images': int(images.shape[0])}
        print(f'SPIKE {args.label} {key}: {images.shape[0]} images in {seconds:.1f} s', flush=True)
    timing['peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    np.savez_compressed(out / f'images_{args.label}.npz', **arrays)
    (out / f'timing_{args.label}.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
    print(f'SPIKE_GENERATE_DONE {args.label}', flush=True)
    return 0


def command_throughput(args: argparse.Namespace) -> int:
    """Marginal seconds per image: a warm-up batch is generated and discarded before timing."""
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.base_folder))
    pipeline = load_pipeline()
    result: Dict[str, Any] = {'card': torch.cuda.get_device_name(0), 'images_timed': args.images, 'batches': {}}
    for batch_size in args.batch_sizes:
        repeated = (prompts * (args.images // len(prompts) + 1))[:args.images]
        generate_run(pipeline, repeated[:batch_size], 42, batch_size, 'per_image')  # warm-up, discarded
        torch.cuda.reset_peak_memory_stats()
        _, seconds = generate_run(pipeline, repeated, 42, batch_size, 'per_image')
        result['batches'][str(batch_size)] = {
            'seconds_per_image': round(seconds / len(repeated), 4),
            'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
        print(f'SPIKE {args.label} throughput batch {batch_size}: {seconds / len(repeated):.3f} s/image', flush=True)
    (out / f'throughput_{args.label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'SPIKE_THROUGHPUT_DONE {args.label}', flush=True)
    return 0


def difference(left: np.ndarray, right: np.ndarray) -> Dict[str, Any]:
    """Per-image comparison of two equally shaped uint8 stacks, summarised."""
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    per_image_mean = delta.reshape(delta.shape[0], -1).mean(axis=1)
    per_image_max = delta.reshape(delta.shape[0], -1).max(axis=1)
    return {
        'images': int(delta.shape[0]),
        'identical': int((per_image_max == 0).sum()),
        'mean_abs': round(float(per_image_mean.mean()), 4),
        'mean_abs_per_image': [round(float(v), 4) for v in per_image_mean],
        'max_abs': int(per_image_max.max()),
    }


def command_compare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    stacks: Dict[str, Dict[str, np.ndarray]] = {}
    for card in args.cards:
        path = out / f'images_{card}.npz'
        if not path.exists():
            print(f'missing {path}', file=sys.stderr)
            return 2
        with np.load(path) as loaded:
            stacks[card] = {key: loaded[key] for key in loaded.files}
    summary: Dict[str, Any] = {'within_card': {}, 'across_cards': {}, 'floors': {}}
    for card, runs in stacks.items():
        within: Dict[str, Any] = {}
        for scheme in ('per_image', 'stream'):
            b1, b8, b16 = (runs[run_key(scheme, 42, b)] for b in (1, 8, 16))
            within[f'{scheme}: batch 1 vs 8'] = difference(b1, b8)
            within[f'{scheme}: batch 8 vs 16'] = difference(b8, b16)
            within[f'{scheme}: batch 1 vs 16'] = difference(b1, b16)
        within['per_image vs stream, batch 8 (sanity: must differ except where noise coincides)'] = difference(
            runs[run_key('per_image', 42, 8)], runs[run_key('stream', 42, 8)])
        summary['within_card'][card] = within
        reference = runs[run_key('per_image', 42, 8)]
        summary['floors'][card] = {
            'different prompt, same seed (image i against image i+1, within a task)': difference(
                np.concatenate([reference[0:7], reference[8:15]]), np.concatenate([reference[1:8], reference[9:16]])),
            'same prompt, seed 42 against seed 43': difference(reference, runs[run_key('per_image', 43, 8)]),
        }
    cards = list(stacks)
    for i, left in enumerate(cards):
        for right in cards[i + 1:]:
            for key in stacks[left]:
                if key in stacks[right]:
                    summary['across_cards'][f'{left} vs {right}: {key}'] = difference(stacks[left][key], stacks[right][key])
    (out / 'noise_route_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SPIKE_COMPARE_DONE', flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'throughput'):
        command = sub.add_parser(name, help=f'{name} on the visible card')
        command.add_argument('--label', required=True, help='Card label used in the output file names, e.g. a40.')
        command.add_argument('--out', required=True, help='Output directory.')
        command.add_argument('--base-folder', default='vision_unlearning/benchmarks/I_care/assets',
                             help='The I-CARE assets folder holding the task metadata files.')
        if name == 'throughput':
            command.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32])
            command.add_argument('--images', type=int, default=64, help='Images timed per batch size, after warm-up.')
    compare = sub.add_parser('compare', help='compare the images written by generate, no card needed')
    compare.add_argument('--out', required=True, help='Directory holding images_<card>.npz.')
    compare.add_argument('--cards', nargs='+', required=True, help='Labels given to generate.')
    args = parser.parse_args(argv)
    if args.command in ('generate', 'throughput') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
        print('CUBLAS_WORKSPACE_CONFIG must be set before Python starts, e.g. :4096:8', file=sys.stderr)
        return 2
    if args.command == 'generate':
        return command_generate(args)
    if args.command == 'throughput':
        return command_throughput(args)
    return command_compare(args)


if __name__ == '__main__':
    sys.exit(main())
