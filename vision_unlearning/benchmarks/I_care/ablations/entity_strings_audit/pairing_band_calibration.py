"""What does the pairing ratio look like when the two images are known to share their noise?

Ticket `2026-07-16-TargetPreprocessedBug`, stage S8. Criterion 2 declares a receiver's `on` and `off`
images a pair when the median mean-absolute pixel difference between them is at or below **0.25** of
the unrelated-pair floor, and unpaired at or above 0.75. Those two numbers were fixed in the plan
before any run, and -- this is the point of this script -- they were never calibrated against a case
whose answer is known. A first session came out at 0.56 and stayed there when restricted to the
receivers it had left visually intact, which is either a real defect or a badly chosen band.

There is a sample on disk that settles which. The every-epoch ablation's campaign grids were
generated at **batch size 1 with one prompt ordering shared by the baseline row and every epoch row**
(`manifest_s{seed}.json` records both), so an `off` image and an `on` image of the same entity are
paired *by construction*: same seed, same position, same generator state. Their determinism was
checked at the time by regenerating in a fresh process and comparing pixel for pixel. They therefore
answer the question this control needs answering -- what the ratio reads for a genuinely paired
comparison across a real unlearning edit -- and they answer it at a sweep of edit strengths, because
the grid holds an adapter per epoch.

What comes out is a calibration curve rather than a single number: the ratio at epoch 1, where the
edit is nearly nothing and the ratio must be near zero if the measurement works at all, rising with
the epoch count as the edit bites. If the ratio for a strongly trained, known-paired adapter also
sits near 0.5, then 0.25 does not separate paired from unpaired for a real edit and the criterion has
to be re-derived rather than the pipeline suspected.

The quantities are exactly `pairing_control.py`'s, so the numbers are comparable with it:

* **on-versus-off** -- median over the *receivers* (every entity except the trained target) of the
  mean absolute pixel difference between that entity's `on` and `off` image, 0-255 scale.
* **unrelated floor** -- entity *i*'s `off` image against entity *j*'s, every unordered pair.
* **ratio** -- the first over the second, against the plan's 0.25 and 0.75 bands.

The target's own value is reported beside them and is never mixed into the median: the target is the
entity the adapter was trained to destroy, so it is not a receiver.

Usage, from the repository root::

    PYTHONPATH=. python vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/pairing_band_calibration.py \\
        --grid-folder vision_unlearning/benchmarks/I_care/ablations/every_epoch/assets/epoch_grid_campaign_people \\
        --out vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/assets/pairing_band_people.json

    # every grid at once, one JSON each
    python .../pairing_band_calibration.py --all-campaign-grids

    # no data needed: proves the ratio can tell a paired pair from an unpaired one
    python .../pairing_band_calibration.py --self-check

Exit 0 when every requested grid was measured, 1 when a grid was unusable (missing images, no
manifest, fewer than two entities), 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABLATIONS = os.path.abspath(os.path.join(_HERE, '..'))

#: The bands criterion 2 fixes in the plan, repeated here so the output carries its own yardstick.
PAIRED_AT_OR_BELOW = 0.25
UNPAIRED_AT_OR_ABOVE = 0.75

#: The campaign grids the every-epoch ablation left on disk, one per task.
CAMPAIGN_GRIDS = (
    'epoch_grid_campaign_people',
    'epoch_grid_campaign_breeds',
    'epoch_grid_campaign_scenes',
)


@dataclass
class EpochRow:
    """The three quantities at one epoch, for one seed."""

    epoch: int
    n_receivers: int
    receiver_median: float
    target_value: float
    ratio: float

    def to_json(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'n_receivers': self.n_receivers,
            'receiver_on_off_median': round(self.receiver_median, 4),
            'target_on_off': round(self.target_value, 4),
            'ratio_median_over_floor': round(self.ratio, 4),
            'verdict_at_plan_bands': _verdict(self.ratio),
        }


def _verdict(ratio: float) -> str:
    if ratio <= PAIRED_AT_OR_BELOW:
        return 'paired'
    if ratio >= UNPAIRED_AT_OR_ABOVE:
        return 'unpaired'
    return 'unresolved'


def load_rgb(path: str) -> np.ndarray:
    """Read one image as a signed array, so that a difference cannot wrap around."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert('RGB'), dtype=np.int16)


def mean_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise SystemExit(f'ERROR: shapes {left.shape} and {right.shape} are not comparable')
    return float(np.abs(left - right).mean())


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation between two images' pixel values, flattened over colour channels.

    A second, structural view of the same question, and a much sharper one: two images that share
    their initial noise keep the same composition, so their pixels co-vary even when the subject has
    been replaced, while two images from different draws are uncorrelated whatever their average
    distance happens to be.
    """
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def unrelated_floor(images: List[np.ndarray]) -> Tuple[float, int]:
    """Median distance between two *different* entities' images, over every unordered pair."""
    values: List[float] = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            values.append(mean_abs(images[i], images[j]))
    if not values:
        raise SystemExit('ERROR: fewer than two entities, so no unrelated-pair floor exists')
    return statistics.median(values), len(values)


def measure_grid(folder: str, seed: int) -> Dict[str, Any]:
    """Measure one campaign grid at one seed, at every epoch it holds.

    The manifest is what makes this sample usable: it records the prompt order and the batch size
    that produced every row, which is the evidence that the `off` row and each `on` row share their
    initial noise. Without it the folder is just images and the comparison would be an assumption.
    """
    manifest_path = os.path.join(folder, f'manifest_s{seed}.json')
    if not os.path.isfile(manifest_path):
        raise SystemExit(f'ERROR: no manifest at {manifest_path}; cannot establish the noise pairing')
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    entities: List[str] = manifest['generation_order']
    epochs: List[int] = manifest['epochs']
    batch_size = manifest.get('batch_size')
    off_images = [load_rgb(os.path.join(folder, f'off_s{seed}_b{i}.png')) for i in range(len(entities))]
    floor, n_pairs = unrelated_floor(off_images)

    rows: List[EpochRow] = []
    for epoch in epochs:
        paths = [os.path.join(folder, f'on_ep{epoch}_s{seed}_b{i}.png') for i in range(len(entities))]
        if not all(os.path.isfile(path) for path in paths):
            continue
        values = [mean_abs(load_rgb(path), off_images[i]) for i, path in enumerate(paths)]
        receivers = values[1:]  # entity 0 is the trained target, by the grid's own convention
        rows.append(EpochRow(
            epoch=epoch,
            n_receivers=len(receivers),
            receiver_median=statistics.median(receivers),
            target_value=values[0],
            ratio=statistics.median(receivers) / floor if floor else float('nan'),
        ))

    return {
        'grid_folder': folder,
        'task': manifest.get('task'),
        'method': manifest.get('method'),
        'seed': seed,
        'batch_size_of_every_row': batch_size,
        'target_entity': entities[0],
        'n_entities': len(entities),
        'unrelated_pair_floor': round(floor, 4),
        'n_unrelated_pairs': n_pairs,
        'paired_at_or_below': PAIRED_AT_OR_BELOW,
        'unpaired_at_or_above': UNPAIRED_AT_OR_ABOVE,
        'epochs_measured': [row.epoch for row in rows],
        'rows': [row.to_json() for row in rows],
    }


def measure_cross_seed(folder: str, seed_same: int, seed_other: int) -> Dict[str, Any]:
    """Measure the same-seed distance against the cross-seed distance, for the same entity.

    This is the statistic the unrelated-pair floor cannot be: the size of an unlearning edit cancels
    out of it. For one receiver, ``d_same`` is its `on` image against its *own* seed's `off` image
    and ``d_cross`` is the same `on` image against the *other* seed's `off` image of the same entity
    and the same prompt. Both carry the edit; only the first can carry shared initial noise. So

    * noise shared  -> ``d_same / d_cross`` is well below 1 and falls further the more the two
      images have in common;
    * noise not shared -> the two distances measure the same thing and the ratio sits at about 1.

    A ratio meaningfully above 1 would be stranger than either and is reported rather than clipped.
    """
    manifest_path = os.path.join(folder, f'manifest_s{seed_same}.json')
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    entities: List[str] = manifest['generation_order']
    epochs: List[int] = manifest['epochs']

    off_same = [load_rgb(os.path.join(folder, f'off_s{seed_same}_b{i}.png')) for i in range(len(entities))]
    off_other = [load_rgb(os.path.join(folder, f'off_s{seed_other}_b{i}.png')) for i in range(len(entities))]

    rows: List[Dict[str, Any]] = []
    for epoch in epochs:
        paths = [os.path.join(folder, f'on_ep{epoch}_s{seed_same}_b{i}.png') for i in range(len(entities))]
        if not all(os.path.isfile(path) for path in paths):
            continue
        same: List[float] = []
        cross: List[float] = []
        same_corr: List[float] = []
        cross_corr: List[float] = []
        for i, path in enumerate(paths):
            on_image = load_rgb(path)
            same.append(mean_abs(on_image, off_same[i]))
            cross.append(mean_abs(on_image, off_other[i]))
            same_corr.append(correlation(on_image, off_same[i]))
            cross_corr.append(correlation(on_image, off_other[i]))
        receiver_same = same[1:]
        receiver_cross = cross[1:]
        ratios = [s / c for s, c in zip(receiver_same, receiver_cross) if c]
        rows.append({
            'epoch': epoch,
            'n_receivers': len(ratios),
            'receiver_same_seed_median': round(statistics.median(receiver_same), 4),
            'receiver_cross_seed_median': round(statistics.median(receiver_cross), 4),
            'receiver_ratio_median': round(statistics.median(ratios), 4),
            'receiver_same_seed_correlation_median': round(statistics.median(same_corr[1:]), 4),
            'receiver_cross_seed_correlation_median': round(statistics.median(cross_corr[1:]), 4),
            'target_same_seed': round(same[0], 4),
            'target_cross_seed': round(cross[0], 4),
            'target_same_seed_correlation': round(same_corr[0], 4),
        })

    return {
        'grid_folder': folder,
        'task': manifest.get('task'),
        'method': manifest.get('method'),
        'seed_same': seed_same,
        'seed_other': seed_other,
        'batch_size_of_every_row': manifest.get('batch_size'),
        'target_entity': entities[0],
        'n_entities': len(entities),
        'rows': rows,
    }


def print_cross_seed(result: Dict[str, Any]) -> None:
    print(f"\n=== cross-seed control: {os.path.basename(result['grid_folder'])} "
          f"task={result['task']} method={result['method']} "
          f"on seed {result['seed_same']} against off seeds {result['seed_same']} and "
          f"{result['seed_other']}")
    print(f"{'epoch':>6} {'receivers':>10} {'same-seed':>10} {'cross-seed':>11} {'ratio':>7} "
          f"{'corr same':>10} {'corr cross':>11}")
    for row in result['rows']:
        print(f"{row['epoch']:>6} {row['n_receivers']:>10} {row['receiver_same_seed_median']:>10.4f} "
              f"{row['receiver_cross_seed_median']:>11.4f} {row['receiver_ratio_median']:>7.4f} "
              f"{row['receiver_same_seed_correlation_median']:>10.4f} "
              f"{row['receiver_cross_seed_correlation_median']:>11.4f}")


def self_check() -> int:
    """Prove the ratio separates a paired comparison from an unpaired one, with no data on disk.

    Two synthetic entities: each `off` image is its own random field, and each `on` image is that
    same field with a small perturbation -- the analogue of a weight edit on shared noise. The
    paired ratio must be small and the cross-entity ratio must be around one, or the statistic
    itself is not measuring what this script claims.
    """
    rng = np.random.default_rng(0)
    off_a = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8).astype(np.int16)
    off_b = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8).astype(np.int16)
    edit = rng.integers(-5, 6, size=off_a.shape).astype(np.int16)
    on_a = np.clip(off_a + edit, 0, 255)

    floor, _ = unrelated_floor([off_a, off_b])
    paired_ratio = mean_abs(on_a, off_a) / floor
    unpaired_ratio = mean_abs(on_a, off_b) / floor
    print(f'unrelated floor      : {floor:.4f}')
    print(f'paired   (edit only) : {mean_abs(on_a, off_a):.4f}  ratio {paired_ratio:.4f}')
    print(f'unpaired (other draw): {mean_abs(on_a, off_b):.4f}  ratio {unpaired_ratio:.4f}')
    if paired_ratio >= PAIRED_AT_OR_BELOW:
        print('SELF_CHECK_FAILED: a small edit on shared noise did not read as paired')
        return 1
    if unpaired_ratio <= UNPAIRED_AT_OR_ABOVE:
        print('SELF_CHECK_FAILED: a different noise draw did not read as unpaired')
        return 1

    # The cross-seed statistic on the same synthetic material. `off_b` stands in for the same entity
    # drawn from the other seed, which is what the real control compares against. In the shared case
    # the `on` image is built on `off_a`; in the unshared case it is a third draw entirely, which is
    # what a pass that does not share noise with either baseline looks like.
    on_unshared = np.clip(rng.integers(0, 256, size=off_a.shape).astype(np.int16) + edit, 0, 255)
    cross_shared = mean_abs(on_a, off_a) / mean_abs(on_a, off_b)
    cross_unshared = mean_abs(on_unshared, off_a) / mean_abs(on_unshared, off_b)
    print(f'cross-seed, shared   : {cross_shared:.4f}')
    print(f'cross-seed, unshared : {cross_unshared:.4f}')
    if cross_shared >= 0.5:
        print('SELF_CHECK_FAILED: the cross-seed ratio did not fall when the noise was shared')
        return 1
    if not 0.9 <= cross_unshared <= 1.1:
        print('SELF_CHECK_FAILED: the cross-seed ratio left 1.0 when neither baseline shared the noise')
        return 1

    corr_shared = correlation(on_a, off_a)
    corr_unshared = correlation(on_a, off_b)
    print(f'correlation, shared  : {corr_shared:.4f}')
    print(f'correlation, unshared: {corr_unshared:.4f}')
    if corr_shared <= 0.5:
        print('SELF_CHECK_FAILED: the correlation did not rise when the noise was shared')
        return 1
    if abs(corr_unshared) >= 0.2:
        print('SELF_CHECK_FAILED: the correlation did not fall to zero for a different draw')
        return 1
    print('SELF_CHECK_OK: the ratio separates shared noise from a different draw')
    return 0


def print_table(result: Dict[str, Any]) -> None:
    print(f"\n=== {os.path.basename(result['grid_folder'])}  task={result['task']} "
          f"method={result['method']} seed={result['seed']} batch={result['batch_size_of_every_row']}")
    print(f"target={result['target_entity']}  entities={result['n_entities']}  "
          f"unrelated floor={result['unrelated_pair_floor']} over {result['n_unrelated_pairs']} pairs")
    print(f"{'epoch':>6} {'receivers':>10} {'median on-off':>14} {'ratio':>7} {'verdict':>11} "
          f"{'target on-off':>14}")
    for row in result['rows']:
        print(f"{row['epoch']:>6} {row['n_receivers']:>10} {row['receiver_on_off_median']:>14.4f} "
              f"{row['ratio_median_over_floor']:>7.4f} {row['verdict_at_plan_bands']:>11} "
              f"{row['target_on_off']:>14.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='pairing_band_calibration.py',
        description=(
            'Measure the on-versus-off pairing ratio on images that are paired by construction, '
            'at every epoch of an every-epoch campaign grid, so that criterion 2\'s bands can be '
            'read against a case whose answer is known.'
        ),
        epilog=(
            'Exit 0 every requested grid was measured; 1 a grid was unusable; 2 a usage error. '
            'The module docstring explains why these grids are a valid paired sample.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grid-folder', help='one every-epoch campaign grid folder')
    parser.add_argument('--all-campaign-grids', action='store_true',
                        help='measure the people, breeds and scenes campaign grids that the '
                             'every-epoch ablation left under its own assets folder')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43],
                        help='which seeds to measure; a grid holds one manifest per seed '
                             '(default: 42 43)')
    parser.add_argument('--out', help='where to write the JSON. With --all-campaign-grids this is '
                                      'a directory; with --grid-folder it is a file')
    parser.add_argument('--cross-seed', action='store_true',
                        help='also run the cross-seed control: each receiver\'s on image against '
                             'its own seed\'s off image and against the other seed\'s off image of '
                             'the same entity. The edit cancels out of that ratio, so it tests the '
                             'noise sharing itself rather than the size of the edit. Needs exactly '
                             'two seeds')
    parser.add_argument('--self-check', action='store_true',
                        help='run the synthetic paired-versus-unpaired check and exit; needs no data')
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()
    if bool(args.grid_folder) == bool(args.all_campaign_grids):
        parser.error('give exactly one of --grid-folder or --all-campaign-grids (or --self-check)')

    folders: List[str]
    if args.all_campaign_grids:
        folders = [os.path.join(_ABLATIONS, 'every_epoch', 'assets', name) for name in CAMPAIGN_GRIDS]
    else:
        folders = [args.grid_folder]

    status = 0
    for folder in folders:
        if not os.path.isdir(folder):
            print(f'SKIPPED (no such folder): {folder}')
            status = 1
            continue
        for seed in args.seeds:
            if not os.path.isfile(os.path.join(folder, f'manifest_s{seed}.json')):
                print(f'SKIPPED (no manifest for seed {seed}): {folder}')
                status = 1
                continue
            result = measure_grid(folder, seed)
            print_table(result)
            if args.cross_seed:
                if len(args.seeds) != 2:
                    parser.error('--cross-seed needs exactly two seeds')
                other = args.seeds[1] if seed == args.seeds[0] else args.seeds[0]
                if os.path.isfile(os.path.join(folder, f'manifest_s{other}.json')):
                    cross = measure_cross_seed(folder, seed, other)
                    result['cross_seed_control'] = cross
                    print_cross_seed(cross)
                else:
                    print(f'SKIPPED cross-seed control (no seed {other} in this grid)')
                    status = 1
            if args.out:
                # One file per grid AND per seed, always. A single --out path with several seeds
                # would have each seed overwrite the last, leaving a file whose contents depend on
                # the argument order.
                if args.all_campaign_grids:
                    os.makedirs(args.out, exist_ok=True)
                    path = os.path.join(args.out, f'{os.path.basename(folder)}_s{seed}.json')
                else:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    stem, extension = os.path.splitext(args.out)
                    path = args.out if len(args.seeds) == 1 else f'{stem}_s{seed}{extension}'
                with open(path, 'w', encoding='utf-8') as handle:
                    json.dump(result, handle, indent=2, ensure_ascii=False)
                print(f'written {path}')
    return status


if __name__ == '__main__':
    sys.exit(main())
