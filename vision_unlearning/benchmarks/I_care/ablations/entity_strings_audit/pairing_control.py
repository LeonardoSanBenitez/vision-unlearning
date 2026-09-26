"""Are an unlearned image and its baseline actually a pair? Measured, not assumed.

Two controls from the S8 pass criteria of ticket `2026-07-16-TargetPreprocessedBug`, run before any
further card time is spent:

**Criterion 1, the determinism self-control.** An image regenerated from the same prompt, seed,
batch size and library commit must be pixel-identical. If it is not, nothing downstream is
interpretable and there is no point measuring anything else. That half needs a generator and is run
by the session script; here we do its cheap cousin -- an image compared against itself must give
exactly 0, which proves the comparison function itself is not lying.

**Criterion 2, the pairing control.** For a receiver entity -- one the session did not target -- the
`on` image and the `off` baseline image at the same seed should differ only by whatever the
unlearning leaked onto it. If instead they differ by roughly as much as two *unrelated* entities'
images do, they are not a pair at all: they were drawn from different initial noise, and every
paired metric computed from them is measuring the noise draw.

The three quantities, all mean absolute pixel difference on the 0-255 scale:

* **self floor** -- an image against itself. Exactly 0 by construction; a sanity check on the code.
* **on-versus-off** -- the quantity under test, over the receiver set.
* **unrelated floor** -- entity *i*'s image against entity *j*'s, same pass, measured on these very
  images rather than taken from a previous study.

The verdict thresholds come from the plan and are fixed before running: at or below 0.25x the
unrelated floor is **paired**; at or above 0.75x is **unpaired**; between the two is **unresolved**
and stops the stage rather than being rounded to the nearer verdict.

Usage, from the repository root::

    PYTHONPATH=. python vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/pairing_control.py \\
        --on-folder  "assets/datasets/generated_people_George W Bush_uce_000" \\
        --off-folder "assets/datasets/generated_people_baseline" \\
        --task people --target George_W_Bush

Exit 0 when the verdict is `paired`, 1 for `unpaired` or `unresolved`, 2 on a usage or data error.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from vision_unlearning.datasets.entity_names import (  # noqa: E402
    canonical_entity,
    generation_prompt,
    type_entity_task,
)

#: ``{on|off}_{seed:02d}_{prompt}.png`` -- the shape `get_generated_dataset_file` writes.
_IMAGE_NAME = re.compile(r'^(?P<state>on|off)_(?P<seed>\d{2})_(?P<prompt>.+)\.png$')

#: Fixed here rather than chosen after seeing the numbers.
PAIRED_AT_OR_BELOW = 0.25
UNPAIRED_AT_OR_ABOVE = 0.75


@dataclass
class Result:
    """Every number this control produces, with the denominators behind each."""

    n_on: int = 0
    n_off: int = 0
    n_compared: int = 0
    self_floor: float = 0.0
    on_off_median: float = 0.0
    on_off_values: List[float] = field(default_factory=list)
    unrelated_floor: float = 0.0
    n_unrelated_pairs: int = 0
    target_on_off: Optional[float] = None
    ratio: float = 0.0
    verdict: str = 'not computed'

    def to_json(self) -> Dict[str, object]:
        return {
            'n_on_images': self.n_on,
            'n_off_images': self.n_off,
            'n_receivers_compared': self.n_compared,
            'self_floor': round(self.self_floor, 6),
            'on_versus_off_median': round(self.on_off_median, 4),
            'unrelated_pair_floor': round(self.unrelated_floor, 4),
            'n_unrelated_pairs_sampled': self.n_unrelated_pairs,
            'target_on_versus_off': (
                None if self.target_on_off is None else round(self.target_on_off, 4)
            ),
            'ratio_median_over_floor': round(self.ratio, 4),
            'paired_at_or_below': PAIRED_AT_OR_BELOW,
            'unpaired_at_or_above': UNPAIRED_AT_OR_ABOVE,
            'verdict': self.verdict,
        }


def _index(folder: str) -> Dict[Tuple[int, str], str]:
    """Return ``(seed, prompt) -> path`` for one generated-dataset folder."""
    found: Dict[Tuple[int, str], str] = {}
    if not os.path.isdir(folder):
        raise SystemExit(f'ERROR: no such folder {folder}')
    for name in sorted(os.listdir(folder)):
        match = _IMAGE_NAME.match(name)
        if match is None:
            continue
        found[(int(match.group('seed')), match.group('prompt'))] = os.path.join(folder, name)
    return found


def _load(path: str) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert('RGB'), dtype=np.float64)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise SystemExit(f'ERROR: shape mismatch {a.shape} vs {b.shape}')
    return float(np.mean(np.abs(a - b)))


def run(on_folder: str, off_folder: str, task: type_entity_task, target: str,
        seed_for_sampling: int = 42, n_unrelated: int = 100) -> Result:
    result = Result()
    on_images = _index(on_folder)
    off_images = _index(off_folder)
    result.n_on, result.n_off = len(on_images), len(off_images)

    shared = sorted(set(on_images) & set(off_images))
    if not shared:
        raise SystemExit('ERROR: the two folders share no (seed, prompt) key at all')

    target_prompt = generation_prompt(task, target)

    # Criterion 1's cheap half: the comparison itself must report 0 for identical input.
    first = _load(on_images[shared[0]])
    result.self_floor = _mean_abs_diff(first, first)

    # On versus off, over receivers only -- the target is expected to change and is reported apart.
    for key in shared:
        value = _mean_abs_diff(_load(on_images[key]), _load(off_images[key]))
        if key[1] == target_prompt:
            result.target_on_off = value
        else:
            result.on_off_values.append(value)
    result.n_compared = len(result.on_off_values)
    result.on_off_median = float(np.median(result.on_off_values)) if result.on_off_values else 0.0

    # The unrelated floor, measured on THESE images: different entities, same pass, same seed.
    rng = random.Random(seed_for_sampling)
    by_seed: Dict[int, List[Tuple[int, str]]] = {}
    for key in shared:
        by_seed.setdefault(key[0], []).append(key)
    unrelated: List[float] = []
    for _ in range(n_unrelated):
        seed = rng.choice(sorted(by_seed))
        keys = by_seed[seed]
        if len(keys) < 2:
            continue
        first_key, second_key = rng.sample(keys, 2)
        unrelated.append(_mean_abs_diff(_load(off_images[first_key]), _load(off_images[second_key])))
    result.n_unrelated_pairs = len(unrelated)
    result.unrelated_floor = float(np.median(unrelated)) if unrelated else 0.0

    if result.unrelated_floor == 0.0:
        result.verdict = 'undefined: the unrelated floor is zero'
    else:
        result.ratio = result.on_off_median / result.unrelated_floor
        if result.ratio <= PAIRED_AT_OR_BELOW:
            result.verdict = 'paired'
        elif result.ratio >= UNPAIRED_AT_OR_ABOVE:
            result.verdict = 'unpaired'
        else:
            result.verdict = 'unresolved'
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--on-folder', required=True, help="the session's generated images")
    parser.add_argument('--off-folder', required=True, help='the shared baseline for that task')
    parser.add_argument('--task', required=True, choices=['people', 'breeds', 'scenes'])
    parser.add_argument('--target', required=True, help='the entity this session unlearned')
    parser.add_argument('--output', default=None, help='write the numbers here as JSON')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run(args.on_folder, args.off_folder, args.task, args.target)

    print(f'on images                : {result.n_on}')
    print(f'off images               : {result.n_off}')
    print(f'receivers compared       : {result.n_compared}')
    print(f'self floor (must be 0)   : {result.self_floor:.6f}')
    print(f'on vs off, median        : {result.on_off_median:.4f}')
    print(f'unrelated pair floor     : {result.unrelated_floor:.4f} '
          f'(over {result.n_unrelated_pairs} sampled pairs)')
    if result.target_on_off is not None:
        print(f'target on vs off         : {result.target_on_off:.4f}  '
              f'(the entity that WAS unlearned; expected to be large)')
    print(f'ratio median/floor       : {result.ratio:.4f}')
    print(f'  paired at or below     : {PAIRED_AT_OR_BELOW}')
    print(f'  unpaired at or above   : {UNPAIRED_AT_OR_ABOVE}')
    print(f'VERDICT: {result.verdict}')

    if args.output:
        with open(args.output, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(result.to_json(), handle, indent=2)
        print(f'wrote {args.output}')

    if result.self_floor != 0.0:
        print('PAIRING_CONTROL_FAILED the self floor is not zero, so the comparison itself is wrong')
        return 1
    print(f'PAIRING_CONTROL {"OK" if result.verdict == "paired" else "FAILED"} {result.verdict}')
    return 0 if result.verdict == 'paired' else 1


if __name__ == '__main__':
    raise SystemExit(main())
