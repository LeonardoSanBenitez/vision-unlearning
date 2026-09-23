"""Did moving the embedding join key from ``prompted_entity`` to ``prompt`` move any number?

It must not. Before the change both sides of the join carried the same wrong key, so the buckets
came out the same; the repair makes the join correct, not different. That is an argument, and this
script is the measurement: it recomputes ``embedding_specificity_ratio`` for every scenes entity and
every method from the repaired files, and compares against the values stored in
``assets/interference_per_entity_scenes.json``, which were computed before the change.

It is a check of *equivalence*, not of correctness: it cannot show the new join key is the right
one -- the adversarial fixtures in ``tests/test_pipeline_07_compute_interference_per_entity.py`` do
that, by construction, because they are the only place where the two keys give different answers.

Usage::

    PYTHONPATH=. python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/verify_specificity_ratio_unchanged.py

Prints one line per method with the counted numerator (equal / NaN on both sides / different) and
exits non-zero if any value moved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple, cast

from vision_unlearning.benchmarks.I_care.pipeline_07_compute_interference_per_entity import (
    _compute_specificity_ratio,
    _mean_embeddings_per_entity,
)
from vision_unlearning.datasets.entity_names import canonical_entity, type_entity_task


#: The combinations the stored artifact holds, as (method, num_train_epochs).
COMBINATIONS: List[Tuple[str, int]] = [('uce', 0), ('munba', 100), ('distil', 100)]

RTOL = 1e-9
ATOL = 1e-12


def compare(task: str, assets_folder: str) -> int:
    datasets_folder = os.path.join(assets_folder, 'datasets')
    stored_path = os.path.join(assets_folder, f'interference_per_entity_{task}.json')
    with open(stored_path, encoding='utf-8') as handle:
        stored_rows = json.load(handle)
    print(f'stored rows: {len(stored_rows)} from {stored_path}')

    baseline_path = os.path.join(datasets_folder, f'embeddings_{task}_original.json')
    with open(baseline_path, encoding='utf-8') as handle:
        baseline_mean = _mean_embeddings_per_entity(json.load(handle))
    print(f'baseline entities: {len(baseline_mean)} from {baseline_path}')

    n_different_total = 0
    for method, epochs in COMBINATIONS:
        column = f'metric_{method}_{epochs}_embedding_specificity_ratio (↑)'
        counts: Dict[str, int] = {'equal': 0, 'nan_both': 0, 'different': 0, 'no_stored_value': 0, 'file_missing': 0}
        examples: List[str] = []

        for row in stored_rows:
            target_hf_name = canonical_entity(cast(type_entity_task, task), row['name'])
            target_emb_path = os.path.join(
                datasets_folder, f'embeddings_{task}_{target_hf_name}_{method}_{epochs:03d}.json',
            )
            if not os.path.exists(target_emb_path):
                counts['file_missing'] += 1
                continue
            if column not in row:
                counts['no_stored_value'] += 1
                continue

            recomputed = _compute_specificity_ratio(
                target_hf_name=target_hf_name,
                task=task,  # type: ignore[arg-type]
                target_emb_path=target_emb_path,
                baseline_mean=baseline_mean,
            )
            stored = row[column]
            if stored is None or (isinstance(stored, float) and math.isnan(stored)):
                counts['nan_both' if math.isnan(recomputed) else 'different'] += 1
            elif math.isnan(recomputed):
                counts['different'] += 1
            elif math.isclose(recomputed, float(stored), rel_tol=RTOL, abs_tol=ATOL):
                counts['equal'] += 1
            else:
                counts['different'] += 1
                if len(examples) < 5:
                    examples.append(f"{row['name']}: stored={stored!r} recomputed={recomputed!r}")

        n_different_total += counts['different']
        print(f'{method}_{epochs}: ' + ' '.join(f'{key}={value}' for key, value in counts.items()))
        for example in examples:
            print(f'    {example}')

    return n_different_total


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_assets = os.path.abspath(os.path.join(here, '..', '..', 'assets'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default='scenes', help='task to check (default: %(default)s)')
    parser.add_argument('--assets-folder', default=default_assets, help='I-CARE assets folder (default: %(default)s)')
    args = parser.parse_args()

    n_different = compare(args.task, args.assets_folder)
    print('VERDICT: values unchanged' if n_different == 0 else f'VERDICT: {n_different} value(s) moved')
    return 1 if n_different else 0


if __name__ == '__main__':
    sys.exit(main())
