"""SPIKE -- what does a training step cost on an A40, and what do the precision settings buy? Not evidence.

A planning probe for the full regeneration of the I-CARE corpus, where 600 of the 900 unlearning
sessions train a network and training is the largest single cost. The only per-session timings that
exist were taken on a TITAN RTX at full single precision. This measures, on the visible card, the
seconds per optimizer step of the two training methods under the precision settings their trainers
already expose, so the plan can decide whether to change them before equalization fixes everything
else. Nothing written here may be cited as the result of any stage.

Each variant is a real session configuration, built by ``session_config.session_hyperparameters``
exactly as ``pipeline_03_unlearn_model.py`` builds it, then cut to ``--steps`` optimizer steps and
given empty final evaluation prompt lists so that no evaluation images are generated. The trainer's
own ``Runtime training seconds`` record is read back; it starts after the models are loaded and the
data prepared, so it is the marginal training cost plus the first steps' warm-up. Two step counts
per variant separate the two: the per-step cost is the difference of the two runtimes over the
difference of the two step counts.

Variants:

* SPARE (``distil``), breeds entity 14 -- the slowest recorded session -- at batch 4, accumulation 1,
  the setting ``pipeline_03`` picks on a card with over 20 GB free:
  ``fp32`` (today's configuration), ``tf32`` (TensorFloat-32 matrix products), ``bf16`` (bfloat16
  mixed precision, TensorFloat-32 on).
* SalUn (``salun``), people entity 0: ``fp32`` and ``tf32``. The SalUn trainer has no precision
  setting of its own, so ``tf32`` is set globally before the trainer is built.

Usage, from the repository root on the card to be measured, with the assets folder on disk. **One
process per variant**: Accelerate keeps its mixed-precision setting in process-global state, so a
second variant with another precision cannot be built in the same interpreter::

    PYTHONPATH=. python .../spike_training_throughput.py --out DIR --variant distil:bf16

Writes ``DIR/training_throughput_<card>_<method>_<precision>.json`` and prints ``SPIKE_TRAINING_DONE``
last.
Exit codes: 0 finished; 2 a usage or data error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASSETS = 'vision_unlearning/benchmarks/I_care/assets'
SPLIT_FOLDERS = {'people': 'lfw_splits_filtered', 'breeds': 'taras_breeds_splits_filtered', 'scenes': 'SUN_splits_filtered'}
MODEL_ID = 'CompVis/stable-diffusion-v1-4'

VARIANTS: List[Tuple[str, str, int, str]] = [
    # (method, task, entity index, precision)
    ('distil', 'breeds', 14, 'fp32'),
    ('distil', 'breeds', 14, 'tf32'),
    ('distil', 'breeds', 14, 'bf16'),
    ('salun', 'people', 0, 'fp32'),
    ('salun', 'people', 0, 'tf32'),
]


def run_variant(method: str, task: str, index: int, precision: str, steps: int, out: Path, assets: Path) -> Dict[str, Any]:
    import torch
    from vision_unlearning.benchmarks.I_care.session_config import session_hyperparameters
    from vision_unlearning.unlearner import SalUn, UnlearnerSpare
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    with (assets / f'metadata_{task}_2_enriched_filtered.json').open(encoding='utf-8') as handle:
        target = json.load(handle)[index]['name']
    split = assets / 'datasets' / SPLIT_FOLDERS[task] / target
    output_dir = out / 'sessions' / f'{method}_{precision}_{steps}'
    hyperparameters = session_hyperparameters(
        task, method, target,  # type: ignore[arg-type]
        output_dir=str(output_dir),
        dataset_forget_name=str(split / 'train_forget'),
        dataset_retain_name=str(split / 'train_retain'),
        model_base_name=MODEL_ID,
        device='cuda',
        num_train_epochs=100,
        hub_model_id=None,
    )
    hyperparameters.update({'max_train_steps': steps, 'final_eval_prompts_forget': [], 'final_eval_prompts_retain': []})
    torch.backends.cuda.matmul.allow_tf32 = precision in ('tf32', 'bf16')
    torch.backends.cudnn.allow_tf32 = precision in ('tf32', 'bf16')
    unlearner: Any
    if method == 'distil':
        hyperparameters.update({
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 1,
            'allow_tf32': precision in ('tf32', 'bf16'),
            'mixed_precision': 'bf16' if precision == 'bf16' else 'no',
            'gradient_weighting_method': GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        })
        unlearner = UnlearnerSpare(**hyperparameters)
    else:
        unlearner = SalUn(**hyperparameters)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = unlearner.train()
    wall = time.perf_counter() - started
    runtimes = {r.metric_name: float(r.metric_value) for r in records if str(r.metric_name).startswith('Runtime')}
    training = next(v for k, v in runtimes.items() if k.startswith('Runtime training seconds'))
    del unlearner
    torch.cuda.empty_cache()
    return {
        'method': method, 'task': task, 'index': index, 'precision': precision, 'steps': steps,
        'runtime_training_seconds': round(training, 2), 'wall_seconds': round(wall, 2),
        'runtimes': {k: round(v, 2) for k, v in runtimes.items()},
        'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Output directory.')
    parser.add_argument('--assets', default=ASSETS, help='The I-CARE assets folder.')
    parser.add_argument('--steps', type=int, nargs=2, default=[20, 60], help='The two step counts per variant.')
    parser.add_argument('--variant', required=True, choices=[f'{m}:{p}' for m, _, _, p in VARIANTS],
                        help='method:precision, one per process.')
    args = parser.parse_args(argv)
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets)
    if not (assets / 'datasets').is_dir():
        print(f'no datasets under {assets}', file=sys.stderr)
        return 2
    card = torch.cuda.get_device_name(0).replace(' ', '_')
    rows: List[Dict[str, Any]] = []
    selected = [v for v in VARIANTS if f'{v[0]}:{v[3]}' == args.variant]
    for method, task, index, precision in selected:
        for steps in args.steps:
            row = run_variant(method, task, index, precision, steps, out, assets)
            rows.append(row)
            print(f'SPIKE training {method} {precision} {steps} steps: {row["runtime_training_seconds"]} s', flush=True)
    summary: Dict[str, Any] = {}
    low, high = args.steps
    for method, task, index, precision in selected:
        pair = {r['steps']: r['runtime_training_seconds'] for r in rows if r['method'] == method and r['precision'] == precision}
        summary[f'{method} {precision}'] = {
            'seconds_per_step': round((pair[high] - pair[low]) / (high - low), 4),
            'fixed_seconds': round(pair[low] - low * (pair[high] - pair[low]) / (high - low), 2),
        }
    result = {'card': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'rows': rows, 'per_step': summary,
              'environment': {'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    (out / f'training_throughput_{card}_{args.variant.replace(":", "_")}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SPIKE_TRAINING_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
