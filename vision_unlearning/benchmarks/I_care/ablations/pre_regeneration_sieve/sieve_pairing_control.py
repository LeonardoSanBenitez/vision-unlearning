"""Do an ``on`` image and its ``off`` image start from the same initial noise?

The question cannot be answered by looking at one difference, because there is no absolute
scale for "different". It is answered by comparing three distributions measured on the same
files with the same statistic:

* **self**            -- an image against itself. Zero by construction; proves the comparison works.
* **unrelated**       -- two baseline images of *different* prompts. This is what "no shared
  noise" looks like, measured rather than assumed.
* **on against off**  -- the pairing every per-pair interference value is computed from.

If the third distribution sits on top of the second, the two images of a pair are independent
draws and every paired metric between them is measuring the noise, not the unlearning. If it
sits well below, the noise is shared and the difference is the unlearning effect.

The same three distributions are also scored with the project's own ``rmse`` and ``ssim`` at
the library default and at the 8-bit maximum, because those two metrics take a ``max_p``
argument whose default is 4095.

It reads and prints; it writes one JSON into its own ignored ``assets/``.

Run::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_pairing_control.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))

#: (entity folder, its task's shared baseline folder, a one-line description).
CASES: Tuple[Tuple[str, str, str], ...] = (
    ('generated_breeds_a dogo argentino_distil_100', 'generated_breeds_baseline',
     'adapter method, 400 on images'),
    ('generated_people_Atal Bihari Vajpayee_uce_000', 'generated_people_baseline',
     'closed-form edit: receivers should barely move if the noise is shared'),
    ('generated_scenes_an abbey scene_uce_000', 'generated_scenes_baseline',
     'closed-form edit, scenes'),
)


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _rmse(a: np.ndarray, b: np.ndarray, max_p: float) -> float:
    """The project's rmse, spelled out: per-band root mean square of the scaled difference."""
    diff = (a - b) / max_p
    return float(np.sqrt(np.mean(np.square(diff), axis=(0, 1))).mean())


def run_case(datasets: str, entity_folder: str, shared_folder: str, limit: int) -> Dict[str, Any]:
    entity_dir = os.path.join(datasets, entity_folder)
    shared_dir = os.path.join(datasets, shared_folder)
    if not (os.path.isdir(entity_dir) and os.path.isdir(shared_dir)):
        return {'error': f'missing {entity_dir} or {shared_dir}'}

    on_names = sorted(f for f in os.listdir(entity_dir) if f.startswith('on_'))[:limit]
    paired: List[Tuple[np.ndarray, np.ndarray]] = []
    for name in on_names:
        off_path = os.path.join(shared_dir, 'off_' + name[len('on_'):])
        if not os.path.exists(off_path):
            continue
        paired.append((_load(os.path.join(entity_dir, name)), _load(off_path)))
    if len(paired) < 2:
        return {'error': f'only {len(paired)} usable pairs in {entity_folder}'}

    on_images = [pair[0] for pair in paired]
    off_images = [pair[1] for pair in paired]

    distributions: Dict[str, Dict[str, List[float]]] = {
        name: {'mean_abs': [], 'rmse_4095': [], 'rmse_255': []}
        for name in ('self', 'unrelated', 'on_against_off')
    }

    for index in range(len(paired)):
        other = (index + 1) % len(paired)
        for name, left, right in (
            ('self', off_images[index], off_images[index]),
            ('unrelated', off_images[index], off_images[other]),
            ('on_against_off', on_images[index], off_images[index]),
        ):
            distributions[name]['mean_abs'].append(_mean_abs(left, right))
            distributions[name]['rmse_4095'].append(_rmse(left, right, 4095.0))
            distributions[name]['rmse_255'].append(_rmse(left, right, 255.0))

    summary: Dict[str, Any] = {'pairs': len(paired), 'distributions': {}}
    for name, values in distributions.items():
        summary['distributions'][name] = {
            statistic: {
                'mean': statistics.fmean(series),
                'min': min(series),
                'max': max(series),
            }
            for statistic, series in values.items()
        }
    return summary


def untouched_receivers(assets: str, task: str, emitter_index: int, method: str, epochs: int,
                        entity_folder: str, shared_folder: str, seed: int,
                        tolerance: float) -> Dict[str, Any]:
    """The control whose answer is already known.

    A receiver whose clip score is unchanged by the unlearning was, semantically, not touched.
    If its ``on`` image nevertheless differs from its ``off`` image as much as two pictures of
    different concepts do, the difference cannot be the unlearning -- there is none to see --
    and must be the initial noise. This separates "the images are unrelated" from "the method
    was destructive", which the aggregate numbers alone cannot do.
    """
    datasets = os.path.join(assets, 'datasets')
    per_pair_path = os.path.join(
        datasets, f'interferences_caused_by_{task}_{emitter_index}_{method}_{epochs}.json')
    metadata_path = os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json')
    if not os.path.exists(per_pair_path):
        return {'error': f'missing {per_pair_path}'}
    with open(per_pair_path, 'r', encoding='utf-8') as handle:
        per_pair: Dict[str, Dict[str, float]] = json.load(handle)
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        names = [entry['name'] for entry in json.load(handle)]

    entity_dir = os.path.join(datasets, entity_folder)
    shared_dir = os.path.join(datasets, shared_folder)
    rows: List[Dict[str, Any]] = []
    for name in names:
        row = per_pair.get(name)
        if row is None or 'clip_diff' not in row:
            continue
        if abs(row['clip_diff']) > tolerance:
            continue
        prompt = _prompt_of(task, name)
        on_path = os.path.join(entity_dir, f'on_{seed}_{prompt}.png')
        off_path = os.path.join(shared_dir, f'off_{seed}_{prompt}.png')
        if not (os.path.exists(on_path) and os.path.exists(off_path)):
            continue
        difference = _mean_abs(_load(on_path), _load(off_path))
        rows.append({
            'receiver': name,
            'clip_diff': row['clip_diff'],
            'ssim_stored': row.get('ssim'),
            'mean_abs_difference': difference,
        })

    return {
        'emitter_index': emitter_index,
        'tolerance_on_clip_diff': tolerance,
        'receivers_within_tolerance': len(rows),
        'mean_abs_difference_mean': statistics.fmean(r['mean_abs_difference'] for r in rows) if rows else None,
        'rows': sorted(rows, key=lambda r: abs(r['clip_diff']))[:12],
    }


def _prompt_of(task: str, name: str) -> str:
    """The generation prompt, reproducing ``get_target_overwrite`` for the three tasks."""
    if task == 'people':
        body = name.replace('_', ' ')
    elif task == 'breeds':
        article = 'an' if name[0].lower() in 'aeiou' else 'a'
        body = f'{article} {name}'.replace('_', ' ')
    else:
        article = 'an' if name[0].lower() in 'aeiou' else 'a'
        body = f'{article} {name} scene'.replace('_', ' ')
    return f'An image of {" ".join(body.split())}'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--limit', type=int, default=60, help='on_* images per case.')
    parser.add_argument('--tolerance', type=float, default=0.25,
                        help='|clip_diff| below which a receiver counts as semantically untouched.')
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_pairing_control.json'))
    args = parser.parse_args(argv)

    datasets = os.path.join(args.assets, 'datasets')
    report: Dict[str, Any] = {'datasets': datasets, 'cases': {}}
    for entity_folder, shared_folder, description in CASES:
        result = run_case(datasets, entity_folder, shared_folder, args.limit)
        report['cases'][entity_folder] = {'description': description, **result}
        print(f'=== {entity_folder}')
        print(f'    {description}')
        if 'error' in result:
            print(f'    {result["error"]}')
            continue
        print(f'    {result["pairs"]} pairs')
        header = f'    {"comparison":<16}{"mean|diff| 0-255":>18}{"rmse max_p=4095":>18}{"rmse max_p=255":>17}'
        print(header)
        for name in ('self', 'unrelated', 'on_against_off'):
            entry = result['distributions'][name]
            print(f'    {name:<16}{entry["mean_abs"]["mean"]:>18.4f}'
                  f'{entry["rmse_4095"]["mean"]:>18.4f}{entry["rmse_255"]["mean"]:>17.4f}')

    print('\n=== control: receivers whose clip score did not change ===')
    control = untouched_receivers(
        args.assets, 'breeds', 0, 'distil', 100,
        'generated_breeds_a dogo argentino_distil_100', 'generated_breeds_baseline',
        seed=42, tolerance=args.tolerance,
    )
    report['untouched_receiver_control'] = control
    if 'error' in control:
        print(f'    {control["error"]}')
    else:
        print(f'    {control["receivers_within_tolerance"]} receivers with '
              f'|clip_diff| <= {control["tolerance_on_clip_diff"]}')
        print(f'    their mean absolute image difference: {control["mean_abs_difference_mean"]}')
        print(f'    {"receiver":<34}{"clip_diff":>11}{"ssim":>9}{"mean|diff|":>12}')
        for row in control['rows']:
            ssim_text = f'{row["ssim_stored"]:.3f}' if row['ssim_stored'] is not None else '   -  '
            print(f'    {row["receiver"][:33]:<34}{row["clip_diff"]:>+11.4f}'
                  f'{ssim_text:>9}{row["mean_abs_difference"]:>12.2f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nPAIRING_CONTROL_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
