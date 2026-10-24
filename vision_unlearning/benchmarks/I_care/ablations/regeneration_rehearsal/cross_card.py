"""Rehearsal: how far one session's outputs move between two runs, for the reproduction tolerances.

Compares a session's outputs in two assets folders -- typically one written on an A40 and one written on
another card model -- and reports, per method:

* **images**, with ``compare_folders.compare``: byte-identical count, mean and worst-image mean absolute
  pixel difference on 0-255, and whether the recorded initial-latent hashes agree;
* **the per-pair file**, receiver by receiver, for every metric of ``metadata.PER_PAIR_METRICS``: how many
  receivers hold exactly the same value, the largest and the mean absolute difference, and the Pearson
  correlation over receivers;
* **the per-seed file**, value by value, for every metric and for both CLIP components (``clip_off`` and
  ``clip_on``): how many values are equal and the largest absolute difference.

Writes the result as JSON and prints one ``CROSS_CARD <method> ...`` line per method, then
``CROSS_CARD_DONE``. Read-only on both folders.

Usage::

    python cross_card.py --a ASSETS_A --b ASSETS_B --task people --index 0 \\
        --sessions uce:0 distil:400 salun:100 --output RESULT.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

from vision_unlearning.benchmarks.I_care.ablations.regeneration_rehearsal.compare_folders import compare
from vision_unlearning.benchmarks.I_care.metadata import PER_PAIR_METRICS, get_interference_per_pair_path, per_seed_path_of
from vision_unlearning.datasets.entity_names import canonical_entity, type_entity_task
from vision_unlearning.datasets.testbed import get_generated_dataset_folder


def _load(path: str) -> Any:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float('nan')
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def per_pair_differences(file_a: str, file_b: str) -> Dict[str, Any]:
    """Receiver by receiver, per metric: equal count, largest and mean absolute difference, Pearson."""
    rows_a, rows_b = _load(file_a), _load(file_b)
    receivers = sorted(set(rows_a) & set(rows_b))
    out: Dict[str, Any] = {'receivers': len(receivers)}
    for metric in PER_PAIR_METRICS:
        a = [float(rows_a[r][metric]) for r in receivers]
        b = [float(rows_b[r][metric]) for r in receivers]
        d = [abs(x - y) for x, y in zip(a, b)]
        out[metric] = {'equal': sum(1 for x in d if x == 0.0), 'max_abs': max(d), 'mean_abs': sum(d) / len(d),
                       'pearson': _pearson(a, b)}
    return out


def per_seed_differences(file_a: str, file_b: str) -> Dict[str, Any]:
    """Value by value over every receiver and seed: equal count and largest absolute difference."""
    seeds_a, seeds_b = _load(file_a)['receivers'], _load(file_b)['receivers']
    receivers = sorted(set(seeds_a) & set(seeds_b))
    out: Dict[str, Any] = {}
    for metric in list(PER_PAIR_METRICS) + ['clip_off', 'clip_on']:
        d = [abs(float(x) - float(y)) for r in receivers for x, y in zip(seeds_a[r][metric], seeds_b[r][metric])]
        out[metric] = {'values': len(d), 'equal': sum(1 for x in d if x == 0.0), 'max_abs': max(d)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True, help='the reference assets folder')
    parser.add_argument('--b', required=True, help='the candidate assets folder')
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--index', type=int, required=True, help="the target's index in the task metadata")
    parser.add_argument('--sessions', nargs='+', required=True, help='METHOD:EPOCHS for each session to compare')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    task: type_entity_task = args.task
    with open(os.path.join(args.a, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        entity = json.load(handle)[args.index]['name']
    result: Dict[str, Any] = {'a': args.a, 'b': args.b, 'task': task, 'index': args.index, 'sessions': {}}
    for spec in args.sessions:
        method, epochs_text = spec.split(':')
        epochs = int(epochs_text)
        folders = [get_generated_dataset_folder(task, method, epochs, canonical_entity(task, entity), base)  # type: ignore[arg-type]
                   for base in (args.a, args.b)]
        per_pair = [get_interference_per_pair_path(task, args.index, method, epochs, base_folder=base)  # type: ignore[arg-type]
                    for base in (args.a, args.b)]
        images = compare(folders[0], folders[1])
        session = {
            'images': {k: v for k, v in images.items() if k != 'per_image'},
            'per_pair': per_pair_differences(per_pair[0], per_pair[1]),
            'per_seed': per_seed_differences(per_seed_path_of(per_pair[0]), per_seed_path_of(per_pair[1])),
        }
        result['sessions'][spec] = session
        clip, dino = session['per_pair']['clip_diff'], session['per_pair']['dino_diff']
        print(f"CROSS_CARD {spec} images={json.dumps(session['images'])} clip_diff_max_abs={clip['max_abs']:.6g} "
              f"clip_diff_pearson={clip['pearson']:.6f} dino_diff_max_abs={dino['max_abs']:.6g}")
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print('CROSS_CARD_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
