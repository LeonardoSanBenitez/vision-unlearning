"""Read-only audit of how per-pair interference is aggregated into per-entity interference.

A per-pair file holds one entry for every entity of the task **including the emitter itself**,
whose entry is the forget effect rather than an interference. Some aggregations in
``metrics.py`` iterate the whole dictionary and some exclude the emitter, so two per-entity
values that read like the same statistic are computed over different populations.

This script measures the size of that difference on the files already on disk, for every
(task, method) combination: how often the emitter is its own worst-interfered entity, and what
each affected aggregate becomes once the emitter's own entry is excluded.

It reads and prints; it writes one JSON into its own ignored ``assets/`` folder and nothing
else. It reimplements no aggregation -- it calls the same helpers the pipeline calls.

Run (no torch needed)::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_aggregation.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vision_unlearning.benchmarks.I_care.metrics import (  # noqa: E402
    average_metric,
    find_worst_interfered,
    number_of_interfered_worse_than_threshold,
)

DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))

#: (task, method, epochs) -- the epoch counts of ``configuration.unlearning_algorithm_to_epochs``.
COMBOS: Tuple[Tuple[str, str, int], ...] = (
    ('people', 'distil', 400), ('people', 'munba', 200), ('people', 'uce', 0),
    ('breeds', 'distil', 100), ('breeds', 'munba', 50), ('breeds', 'uce', 0),
    ('scenes', 'distil', 100), ('scenes', 'munba', 100), ('scenes', 'uce', 0),
)

#: The per-pair metrics and whether "worse" means a bigger number, from ``mp_to_direction``.
METRICS: Tuple[Tuple[str, bool], ...] = (
    ('clip_diff', False),
    ('dino_diff', False),
    ('ssim', False),
    ('rmse', True),
    ('brisque_diff', True),
)


def _per_pair_path(assets: str, task: str, index: int, method: str, epochs: int) -> str:
    return os.path.join(
        assets, 'datasets',
        f'interferences_caused_by_{task}_{index}_{method}_{epochs}.json',
    )


def _load_names(assets: str, task: str) -> List[str]:
    path = os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json')
    with open(path, 'r', encoding='utf-8') as handle:
        return [entry['name'] for entry in json.load(handle)]


def audit_combo(assets: str, task: str, method: str, epochs: int) -> Dict[str, Any]:
    """One (task, method) combination: what changes when the emitter's own entry is dropped."""
    names = _load_names(assets, task)
    per_metric: Dict[str, Dict[str, Any]] = {
        metric: {
            'sessions': 0,
            'emitter_is_own_worst': 0,
            'worst_with_self': [],
            'worst_without_self': [],
            'average_with_self': [],
            'average_without_self': [],
        }
        for metric, _ in METRICS
    }
    threshold_with_self: List[int] = []
    threshold_without_self: List[int] = []
    sessions = 0

    for index, name in enumerate(names):
        path = _per_pair_path(assets, task, index, method, epochs)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as handle:
            per_pair: Dict[str, Dict[str, float]] = json.load(handle)
        if name not in per_pair:
            continue
        sessions += 1
        without_self = {k: v for k, v in per_pair.items() if k != name}
        if not without_self:
            continue

        for metric, worst_is_biggest in METRICS:
            if any(metric not in row for row in per_pair.values()):
                continue
            entry = per_metric[metric]
            entry['sessions'] += 1
            worst_name, worst_value = find_worst_interfered(per_pair, metric, worst_is_biggest)
            if worst_name == name:
                entry['emitter_is_own_worst'] += 1
            entry['worst_with_self'].append(worst_value)
            entry['worst_without_self'].append(
                find_worst_interfered(without_self, metric, worst_is_biggest)[1]
            )
            entry['average_with_self'].append(average_metric(per_pair, metric))
            entry['average_without_self'].append(average_metric(without_self, metric))

        if all('clip_diff' in row for row in per_pair.values()):
            threshold_with_self.append(
                number_of_interfered_worse_than_threshold(per_pair, 'clip_diff', False, 0.0)
            )
            threshold_without_self.append(
                number_of_interfered_worse_than_threshold(without_self, 'clip_diff', False, 0.0)
            )

    summary: Dict[str, Any] = {'sessions_found': sessions, 'metrics': {}}
    for metric, entry in per_metric.items():
        if not entry['sessions']:
            continue
        summary['metrics'][metric] = {
            'sessions': entry['sessions'],
            'emitter_is_own_worst': entry['emitter_is_own_worst'],
            'worst_mean_with_self': statistics.fmean(entry['worst_with_self']),
            'worst_mean_without_self': statistics.fmean(entry['worst_without_self']),
            'average_mean_with_self': statistics.fmean(entry['average_with_self']),
            'average_mean_without_self': statistics.fmean(entry['average_without_self']),
        }
    if threshold_with_self:
        summary['count_below_zero_clip_diff'] = {
            'mean_with_self': statistics.fmean(threshold_with_self),
            'mean_without_self': statistics.fmean(threshold_without_self),
        }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_aggregation.json'))
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {'assets': args.assets, 'combos': {}}
    for task, method, epochs in COMBOS:
        key = f'{task}/{method}/{epochs}'
        report['combos'][key] = audit_combo(args.assets, task, method, epochs)

    for key, summary in report['combos'].items():
        print(f'=== {key}  sessions on disk: {summary["sessions_found"]}')
        for metric, entry in summary.get('metrics', {}).items():
            print(f'    {metric:<13} emitter is its own worst-interfered in '
                  f'{entry["emitter_is_own_worst"]}/{entry["sessions"]} sessions')
            print(f'        worst   with self {entry["worst_mean_with_self"]:+9.4f}   '
                  f'without self {entry["worst_mean_without_self"]:+9.4f}')
            print(f'        average with self {entry["average_mean_with_self"]:+9.4f}   '
                  f'without self {entry["average_mean_without_self"]:+9.4f}')
        below = summary.get('count_below_zero_clip_diff')
        if below:
            print(f'    count(clip_diff < 0)  with self {below["mean_with_self"]:.2f}   '
                  f'without self {below["mean_without_self"]:.2f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nAGGREGATION_SIEVE_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
