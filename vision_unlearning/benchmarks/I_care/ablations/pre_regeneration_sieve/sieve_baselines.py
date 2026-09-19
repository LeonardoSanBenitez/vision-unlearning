"""Read-only comparison of the two sets of baseline images that exist on disk for breeds.

A baseline image is the base model's output for a prompt and a seed, with no unlearning
applied, so the same (seed, prompt) must give the same picture whichever code path produced
it. Two sets exist: the shared task-level folder ``generated_{task}_baseline/`` and a legacy
per-entity folder that still holds its own ``off_*`` copies. Every per-pair interference value
is a comparison between an ``on`` image and one of these, so whether they agree decides whether
those numbers mean what they claim.

This script pairs the files by name and reports, per pair, whether the bytes are identical and
what the mean and maximum absolute pixel difference are. It reads and prints; it writes one
JSON into its own ignored ``assets/`` folder.

Run (needs pillow and numpy, not torch)::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_baselines.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))

#: (shared baseline folder, legacy folder that also holds off_* images).
PAIRS: Tuple[Tuple[str, str], ...] = (
    ('generated_breeds_baseline', 'generated_breeds_a dogo argentino_distil_100'),
)


def _digest(path: str) -> str:
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def compare_folders(datasets: str, left: str, right: str, sample: Optional[int]) -> Dict[str, Any]:
    left_dir = os.path.join(datasets, left)
    right_dir = os.path.join(datasets, right)
    for folder in (left_dir, right_dir):
        if not os.path.isdir(folder):
            return {'error': f'missing folder {folder}'}

    left_off = {f for f in os.listdir(left_dir) if f.startswith('off_')}
    right_off = {f for f in os.listdir(right_dir) if f.startswith('off_')}
    common = sorted(left_off & right_off)
    if sample is not None:
        common = common[:sample]

    identical_bytes = 0
    mean_diffs: List[float] = []
    max_diffs: List[float] = []
    examples: List[Dict[str, Any]] = []

    for name in common:
        left_path = os.path.join(left_dir, name)
        right_path = os.path.join(right_dir, name)
        if _digest(left_path) == _digest(right_path):
            identical_bytes += 1
            mean_diffs.append(0.0)
            max_diffs.append(0.0)
            continue
        left_array = np.asarray(Image.open(left_path).convert('RGB'), dtype=np.int16)
        right_array = np.asarray(Image.open(right_path).convert('RGB'), dtype=np.int16)
        if left_array.shape != right_array.shape:
            examples.append({'file': name, 'note': f'shape {left_array.shape} vs {right_array.shape}'})
            continue
        difference = np.abs(left_array - right_array)
        mean_diffs.append(float(difference.mean()))
        max_diffs.append(float(difference.max()))
        if len(examples) < 5:
            examples.append({
                'file': name,
                'mean_abs_difference': float(difference.mean()),
                'max_abs_difference': float(difference.max()),
                'fraction_of_pixels_differing': float((difference > 0).mean()),
            })

    return {
        'left': left,
        'right': right,
        'left_off_count': len(left_off),
        'right_off_count': len(right_off),
        'common_compared': len(common),
        'only_in_left': len(left_off - right_off),
        'only_in_right': len(right_off - left_off),
        'byte_identical': identical_bytes,
        'mean_abs_difference_over_pairs': statistics.fmean(mean_diffs) if mean_diffs else None,
        'worst_max_abs_difference': max(max_diffs) if max_diffs else None,
        'examples': examples,
    }


def pair_on_against_baselines(datasets: str, entity_folder: str, shared_folder: str,
                              sample: Optional[int]) -> Dict[str, Any]:
    """Which baseline an ``on`` image shares its initial noise with.

    An ``on`` image and its ``off`` counterpart are drawn from the same initial latent only if
    both were produced under the same seeding regime. When they are, an untouched receiver's two
    images are near-identical and the difference is the unlearning effect; when they are not, the
    two are unrelated pictures and every paired metric between them measures the noise draw.

    The discriminator is the size of the difference: a matched pair differs by a few units of 255,
    an unmatched pair by tens.
    """
    entity_dir = os.path.join(datasets, entity_folder)
    shared_dir = os.path.join(datasets, shared_folder)
    if not (os.path.isdir(entity_dir) and os.path.isdir(shared_dir)):
        return {'error': f'missing {entity_dir} or {shared_dir}'}

    on_files = sorted(f for f in os.listdir(entity_dir) if f.startswith('on_'))
    if sample is not None:
        on_files = on_files[:sample]

    against_shared: List[float] = []
    against_legacy: List[float] = []
    compared = 0
    for name in on_files:
        off_name = 'off_' + name[len('on_'):]
        shared_path = os.path.join(shared_dir, off_name)
        legacy_path = os.path.join(entity_dir, off_name)
        if not os.path.exists(shared_path):
            continue
        on_array = np.asarray(Image.open(os.path.join(entity_dir, name)).convert('RGB'), dtype=np.int16)
        shared_array = np.asarray(Image.open(shared_path).convert('RGB'), dtype=np.int16)
        against_shared.append(float(np.abs(on_array - shared_array).mean()))
        if os.path.exists(legacy_path):
            legacy_array = np.asarray(Image.open(legacy_path).convert('RGB'), dtype=np.int16)
            against_legacy.append(float(np.abs(on_array - legacy_array).mean()))
        compared += 1

    return {
        'entity_folder': entity_folder,
        'shared_folder': shared_folder,
        'on_images_compared': compared,
        'mean_abs_difference_against_shared_baseline': (
            statistics.fmean(against_shared) if against_shared else None),
        'mean_abs_difference_against_legacy_off': (
            statistics.fmean(against_legacy) if against_legacy else None),
        'n_against_legacy': len(against_legacy),
    }


#: (entity folder holding on_* images, the shared baseline folder of its task).
ON_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('generated_breeds_a dogo argentino_distil_100', 'generated_breeds_baseline'),
    ('generated_breeds_a dogo argentino_munba_050', 'generated_breeds_baseline'),
    ('generated_people_Atal Bihari Vajpayee_distil_400', 'generated_people_baseline'),
    ('generated_people_Atal Bihari Vajpayee_uce_000', 'generated_people_baseline'),
    ('generated_scenes_an abbey scene_distil_100', 'generated_scenes_baseline'),
    ('generated_scenes_an abbey scene_uce_000', 'generated_scenes_baseline'),
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--sample', type=int, default=None,
                        help='Compare only the first N common files (default: all).')
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_baselines.json'))
    args = parser.parse_args(argv)

    datasets = os.path.join(args.assets, 'datasets')
    report: Dict[str, Any] = {'datasets': datasets, 'comparisons': []}
    for left, right in PAIRS:
        result = compare_folders(datasets, left, right, args.sample)
        report['comparisons'].append(result)
        print(f"=== {left}  vs  {right}")
        if 'error' in result:
            print(f"    {result['error']}")
            continue
        print(f"    off_* files: {result['left_off_count']} and {result['right_off_count']}; "
              f"{result['common_compared']} compared, "
              f"{result['only_in_left']} only left, {result['only_in_right']} only right")
        print(f"    byte identical: {result['byte_identical']}/{result['common_compared']}")
        print(f"    mean absolute pixel difference over pairs: "
              f"{result['mean_abs_difference_over_pairs']}")
        print(f"    worst maximum absolute pixel difference:   {result['worst_max_abs_difference']}")
        for example in result['examples']:
            print(f"        {example}")

    report['on_pairing'] = []
    print('\n=== which baseline each on_* image is paired with '
          '(mean absolute pixel difference, 0-255) ===')
    for entity_folder, shared_folder in ON_PAIRS:
        result = pair_on_against_baselines(datasets, entity_folder, shared_folder, args.sample)
        report['on_pairing'].append(result)
        if 'error' in result:
            print(f"    {result['error']}")
            continue
        print(f"    {entity_folder}")
        print(f"        {result['on_images_compared']} on_* images; "
              f"against shared baseline {result['mean_abs_difference_against_shared_baseline']}; "
              f"against legacy off in the same folder "
              f"{result['mean_abs_difference_against_legacy_off']} "
              f"(n={result['n_against_legacy']})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nBASELINE_SIEVE_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
