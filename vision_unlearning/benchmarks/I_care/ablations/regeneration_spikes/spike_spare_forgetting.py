"""SPIKE -- does SPARE erase its target under a given configuration? Not evidence.

A diagnostic probe for the equalization of the I-CARE regeneration. At its published configuration
(learning rate 1e-4, LoRA rank 4, alpha 8, gradient clipping 1.0, batch 4) SPARE left a people
calibration target recognisable at every value of its epoch ladder, up to 3 200 epochs, while the
first corpus's configuration (rate 6e-4, rank 16, alpha 4, clipping 5.0, batch 2 accumulated twice)
had erased it at 400. This trains one session per configuration variant and reads back the trainer's
own seeded final evaluation, so the plan can decide which setting to change before the ladder is
measured again. The ladder, through the campaign's own read-out, is the evidence; nothing written
here may be cited as the result of any stage.

Each variant is the session configuration ``session_config.session_hyperparameters`` builds, exactly
as ``pipeline_03_unlearn_model.py`` builds it, with the overrides given by ``--set`` applied on top.
Training runs under the same determinism regime and global seeding as ``pipeline_03`` (plan D27), with
the same loss weights (forget 0.3, retain 1.0).

Usage, from the repository root, with an assets folder holding the task metadata and its splits::

    PYTHONPATH=. python .../spike_spare_forgetting.py --assets ASSETS --out DIR --name old_config \\
        --task people --index 8 --epochs 400 --set learning_rate=6e-4 lora_r=16 lora_alpha=4

``--set`` values are read as JSON when they parse (numbers, ``null``, ``true``), otherwise as strings;
``forget_concept=null`` restores conditioning the forget side on the captions stored on disk.

Writes ``DIR/spare_forgetting_<name>.json``: the variant, its overrides, every setting the trainer
received that is a plain value, and every record of the trainer's final evaluation. The record to
read is ``ForgetSet clip score difference between original and unlearned mean``: how much the CLIP
score of the target's own prompts fell, positive meaning erasure (the first corpus's model of the
same entity recorded 12.58). Prints ``SPIKE_SPARE_DONE <name> <that value>`` last.
Exit codes: 0 finished; 2 a usage or data error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

MODEL_ID = 'CompVis/stable-diffusion-v1-4'
SPLIT_FOLDERS = {'people': 'lfw_splits_filtered', 'breeds': 'taras_breeds_splits_filtered', 'scenes': 'SUN_splits_filtered'}
FORGET_RECORD = 'ForgetSet clip score difference between original and unlearned mean'


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """``key=value`` pairs to a dict; each value is JSON when it parses, else the raw string."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition('=')
        if not separator or not key:
            raise ValueError(f'--set expects key=value, got {pair!r}')
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def run(assets: str, out: str, name: str, task: str, index: int, epochs: int, overrides: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from vision_unlearning.benchmarks.I_care.session_config import TRAINING_SEED, assert_split_captions_ready, session_hyperparameters
    from vision_unlearning.unlearner import UnlearnerSpare
    from vision_unlearning.utils.determinism import seed_training
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    with open(os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        target = json.load(handle)[index]['name']
    split = os.path.join(assets, 'datasets', SPLIT_FOLDERS[task], target)
    forget, retain = os.path.join(split, 'train_forget'), os.path.join(split, 'train_retain')
    assert_split_captions_ready(task, 'distil', forget, retain)  # type: ignore[arg-type]
    hyperparameters = session_hyperparameters(
        task, 'distil', target,  # type: ignore[arg-type]
        output_dir=os.path.join(out, 'models', name),
        dataset_forget_name=forget,
        dataset_retain_name=retain,
        model_base_name=MODEL_ID,
        device='cuda',
        num_train_epochs=epochs,
    )
    unknown = sorted(set(overrides) - set(hyperparameters) - {'gradient_accumulation_steps', 'per_device_train_batch_size'})
    if unknown:
        raise ValueError(f'--set names settings the session does not have: {unknown}')
    hyperparameters.update(overrides)
    plain = {k: v for k, v in hyperparameters.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    hyperparameters['gradient_weighting_method'] = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0)

    seed_training(TRAINING_SEED)
    started = time.perf_counter()
    records = UnlearnerSpare(**hyperparameters).train()
    wall = time.perf_counter() - started
    evaluation = {str(r.metric_name): float(r.metric_value) for r in records}
    return {
        'name': name, 'task': task, 'index': index, 'entity': target, 'epochs': epochs,
        'overrides': overrides, 'configuration': plain, 'wall_seconds': round(wall, 1),
        'card': torch.cuda.get_device_name(0), 'evaluation': evaluation,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', required=True, help='An I-CARE assets folder with the task metadata and its splits.')
    parser.add_argument('--out', required=True, help='Output directory.')
    parser.add_argument('--name', required=True, help='The variant\'s name, used in the file names.')
    parser.add_argument('--task', default='people', choices=sorted(SPLIT_FOLDERS))
    parser.add_argument('--index', type=int, default=8, help='Entity index in the task metadata (8: Serena_Williams).')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Settings to override.')
    args = parser.parse_args(argv)
    try:
        overrides = parse_overrides(args.set)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    result = run(args.assets, args.out, args.name, args.task, args.index, args.epochs, overrides)
    path = os.path.join(args.out, f'spare_forgetting_{args.name}.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(path)
    # The trainer's record names end in a direction marker such as " (↑)", so match by prefix.
    forget = next((v for k, v in result['evaluation'].items() if k.startswith(FORGET_RECORD)), None)
    print(f"SPIKE_SPARE_DONE {args.name} {forget}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
