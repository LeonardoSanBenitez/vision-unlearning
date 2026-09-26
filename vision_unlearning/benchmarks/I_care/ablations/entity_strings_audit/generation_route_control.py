"""Do the baseline pass and the session pass draw the same initial noise? Measured, not argued.

Ticket `2026-07-16-TargetPreprocessedBug`, stage S8. The pairing control (`pairing_control.py`)
returned `unresolved`: a receiver's `on` image sits at 0.58 of the unrelated-pair floor away from its
`off` baseline, where the plan calls anything at or below 0.25 a pair. Restricting the measurement to
the 44 receivers the session left semantically intact did not move it (0.5801 against 0.5874), which
kills the obvious explanation that the damage itself inflates the distance.

Two candidates remain, and this script separates them:

**(a) The routes disagree.** The two passes reach the generator differently -- the baseline through
``GeneratedDataset._compute_from_scratch``, which builds the pipeline with
``AutoPipelineForText2Image``, and the session through ``UCE.get_pipeline_from_modified_weights``,
which builds it with ``DiffusionPipeline``. If those two loaders produce even slightly different
pipelines, no pair of images from them shares a starting point and every paired metric in the
benchmark is measuring the difference between two loaders.

**(b) The routes agree and a weight edit simply moves every pixel this much.** Then 0.25 is the wrong
band for a weight-editing method and the criterion has to be re-derived rather than the pipeline
fixed.

The experiment is the unedited model down both routes. Route A loads the pipeline the way the
baseline pass does; route B loads it the way the session does, *without* applying any unlearned
weights. Same prompts, same order, same seed, same batch size, same process. Three comparisons come
out of it:

* ``route_baseline`` against the stored baseline folder -- **criterion 1**, the determinism
  self-control: a fresh process, same everything, must reproduce the stored image exactly.
* ``route_session`` against ``route_baseline`` -- the question above. Zero means the loaders agree
  and hypothesis (b) stands.
* ``route_session`` against the stored baseline folder -- the end-to-end version of the same thing,
  reported because it is what a session actually compares against.

**Why the prompt list may be truncated but not reordered.** ``generate_dataset`` reseeds once per
seed and then advances one generator across the batches of that seed, so the noise an image gets
depends on its position in the list and on the batch size -- not on how many prompts follow it.
Truncating the list at a batch boundary therefore leaves the kept images bit-for-bit what they would
have been in the full run; truncating mid-batch, or reordering, does not. ``--limit-prompts`` is
refused unless it is a multiple of ``--batch-size`` for exactly that reason.

Usage, from the repository root::

    PYTHONPATH=. python vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/generation_route_control.py \\
        --task people \\
        --baseline-folder "assets/datasets/generated_people_baseline" \\
        --out-folder "vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/assets/route_control" \\
        --seed 42 --batch-size 8 --limit-prompts 16

    # no graphics card needed: proves the comparison can see a difference
    python .../generation_route_control.py --self-check

What it writes: ``<out-folder>/route_baseline/`` and ``<out-folder>/route_session/`` hold the
regenerated images (``off_{seed}_{prompt}.png``, the benchmark's own naming), and
``<out-folder>/route_control.json`` holds every number below plus the per-image table.

How to read the result. ``verdict`` is one of:

* ``routes_agree`` -- both comparisons are pixel-identical. The noise IS shared, so the pairing
  ratio is a real effect of the weight edit and criterion 2's band is what needs re-deriving.
* ``loaders_differ`` -- route B differs from route A. The session's loader is the bug; fix the route
  before interpreting any paired metric.
* ``regeneration_differs`` -- route A does not reproduce the stored baseline. Criterion 1 fails and
  nothing downstream is interpretable; suspect the commit, the batch size or the card before
  anything else.
* ``both_differ`` -- both of the above at once.

Exit codes: 0 ``routes_agree``; 1 any other verdict (a real difference was found); 2 a usage or data
error. A nonzero exit is a finding, not a crash.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from vision_unlearning.datasets.entity_names import (  # noqa: E402
    generation_prompt,
    type_entity_task,
)

#: The checkpoint every I-CARE image has ever been generated from.
MODEL_BASE_NAME = 'CompVis/stable-diffusion-v1-4'

#: Folder names under ``--out-folder``, one per route.
ROUTE_BASELINE = 'route_baseline'
ROUTE_SESSION = 'route_session'


@dataclass
class Comparison:
    """One set of image-against-image differences, with its own denominator."""

    name: str
    n_compared: int = 0
    n_identical: int = 0
    max_abs: float = 0.0
    mean_abs: float = 0.0
    worst_prompt: Optional[str] = None
    per_image: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def all_identical(self) -> bool:
        """True only when something was compared and every comparison was exact."""
        return self.n_compared > 0 and self.n_identical == self.n_compared

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'n_compared': self.n_compared,
            'n_pixel_identical': self.n_identical,
            'max_abs_over_images': round(self.max_abs, 6),
            'mean_abs_over_images': round(self.mean_abs, 6),
            'worst_prompt': self.worst_prompt,
            'all_pixel_identical': self.all_identical,
            'per_image': self.per_image,
        }


def image_difference(left: str, right: str) -> Tuple[float, float]:
    """Return ``(max absolute, mean absolute)`` pixel difference on the 0-255 scale.

    Both images are read as RGB and compared in signed arithmetic, because the unsigned
    subtraction of two byte arrays wraps around and would report a difference of 1 as 255.
    """
    with Image.open(left) as handle:
        a = np.asarray(handle.convert('RGB'), dtype=np.int16)
    with Image.open(right) as handle:
        b = np.asarray(handle.convert('RGB'), dtype=np.int16)
    if a.shape != b.shape:
        raise SystemExit(f'ERROR: {left} is {a.shape} and {right} is {b.shape}; not comparable')
    difference = np.abs(a - b)
    return float(difference.max()), float(difference.mean())


def compare_folders(
    name: str,
    left_folder: str,
    right_folder: str,
    filenames: List[str],
    prompts: List[str],
) -> Comparison:
    """Compare one named file in two folders, for every file the caller lists.

    A file missing on either side is an error rather than a skipped row: a comparison whose
    denominator quietly shrinks is the failure mode this whole control exists to catch.
    """
    result = Comparison(name=name)
    worst = -1.0
    for filename, prompt in zip(filenames, prompts):
        left = os.path.join(left_folder, filename)
        right = os.path.join(right_folder, filename)
        for path in (left, right):
            if not os.path.isfile(path):
                raise SystemExit(f'ERROR: expected image not found: {path}')
        max_abs, mean_abs = image_difference(left, right)
        result.per_image.append({
            'prompt': prompt,
            'file_name': filename,
            'max_abs': round(max_abs, 6),
            'mean_abs': round(mean_abs, 6),
        })
        result.n_compared += 1
        if max_abs == 0.0:
            result.n_identical += 1
        if max_abs > worst:
            worst = max_abs
            result.worst_prompt = prompt
        result.max_abs = max(result.max_abs, max_abs)
        result.mean_abs += mean_abs
    if result.n_compared:
        result.mean_abs /= result.n_compared
    return result


def build_prompts(task: type_entity_task, base_folder: str, limit: Optional[int]) -> List[str]:
    """Return the task's generation prompts in the order both image passes use them.

    The order is ``metadata_filtered``'s own, which is what ``pipeline_03`` and ``pipeline_04``
    both iterate, and the string is built by the same two calls they make -- checked here against
    the literal expression at those call sites so that a drift in either one is caught rather than
    silently producing a different image.
    """
    from vision_unlearning.datasets.testbed import (  # noqa: PLC0415
        get_metadata_filtered,
        get_target_overwrite,
    )
    metadata = get_metadata_filtered(task, base_folder=base_folder)
    prompts: List[str] = []
    for entry in metadata:
        at_call_site = f"An image of {get_target_overwrite(task, 'distil', entry['name'])[0]}"
        canonical = generation_prompt(task, entry['name'])
        if at_call_site != canonical:
            raise SystemExit(
                f'ERROR: the pipelines build {at_call_site!r} where entity_names builds '
                f'{canonical!r}. This control would not be regenerating the same images.'
            )
        prompts.append(canonical)
    return prompts if limit is None else prompts[:limit]


def load_pipeline(route: str, device: str) -> Any:
    """Build the pipeline the way *route*'s own production code builds it.

    ``route_baseline`` mirrors ``GeneratedDataset._compute_from_scratch``'s baseline branch;
    ``route_session`` mirrors ``UCE.get_pipeline_from_modified_weights`` with the unlearned tensors
    NOT applied, which is the whole point -- the two pipelines differ only by their loader.
    """
    import torch  # noqa: PLC0415
    if route == ROUTE_BASELINE:
        from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
        return AutoPipelineForText2Image.from_pretrained(
            MODEL_BASE_NAME,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
    from diffusers import DiffusionPipeline  # noqa: PLC0415
    return DiffusionPipeline.from_pretrained(
        MODEL_BASE_NAME,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)


def generate_route(
    route: str,
    prompts: List[str],
    filenames: List[str],
    out_folder: str,
    seed: int,
    batch_size: int,
) -> None:
    """Generate one route's images into ``out_folder/route``, then release the card."""
    import torch  # noqa: PLC0415
    from vision_unlearning.utils.data_generation import generate_dataset  # noqa: PLC0415

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folder = os.path.join(out_folder, route)
    os.makedirs(folder, exist_ok=True)
    print(f'--- {route}: loading pipeline on {device}', flush=True)
    pipeline = load_pipeline(route, device)
    print(f'--- {route}: generating {len(prompts)} images at seed {seed}, batch {batch_size}',
          flush=True)
    generate_dataset(
        model_base_name=None,
        lora_name=None,
        model_pipeline=pipeline,
        prompts=prompts,
        output_path=folder,
        filenames=filenames,
        seeds=[seed],
        batch_size=batch_size,
    )
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def self_check() -> int:
    """Prove the comparison can see a difference. No graphics card, no model, no images on disk.

    The mutation is the point: a control that can only return zero is not a control. Two identical
    arrays must compare exactly 0, and the same arrays with a single channel of a single pixel moved
    by one must not.
    """
    import tempfile
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as folder:
        same_a = os.path.join(folder, 'a.png')
        same_b = os.path.join(folder, 'b.png')
        mutated = os.path.join(folder, 'c.png')
        Image.fromarray(array).save(same_a, 'PNG')
        Image.fromarray(array).save(same_b, 'PNG')
        changed = array.copy()
        changed[0, 0, 0] = np.uint8((int(changed[0, 0, 0]) + 1) % 256)
        Image.fromarray(changed).save(mutated, 'PNG')

        identical_max, identical_mean = image_difference(same_a, same_b)
        mutated_max, mutated_mean = image_difference(same_a, mutated)

    print(f'identical pair : max_abs={identical_max} mean_abs={identical_mean}')
    print(f'one pixel moved: max_abs={mutated_max} mean_abs={mutated_mean:.8f}')
    if identical_max != 0.0 or identical_mean != 0.0:
        print('SELF_CHECK_FAILED: two identical images did not compare as identical')
        return 1
    if mutated_max != 1.0:
        print('SELF_CHECK_FAILED: a one-level change in one pixel was not seen')
        return 1
    print('SELF_CHECK_OK: the comparison returns 0 only when the images really are identical')
    return 0


def decide(regeneration: Comparison, loaders: Comparison) -> str:
    """Name the outcome from the two comparisons that can fail."""
    if regeneration.all_identical and loaders.all_identical:
        return 'routes_agree'
    if regeneration.all_identical:
        return 'loaders_differ'
    if loaders.all_identical:
        return 'regeneration_differs'
    return 'both_differ'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='generation_route_control.py',
        description=(
            'Regenerate baseline images down both of the benchmark\'s generation routes and '
            'compare them, pixel for pixel, against each other and against the stored baseline.'
        ),
        epilog=(
            'Exit 0 when both comparisons are pixel-identical (verdict routes_agree); 1 when a '
            'difference was found; 2 on a usage or data error. Read the module docstring for what '
            'each verdict means.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--task', default='people', choices=['people', 'breeds', 'scenes'],
                        help='which task\'s prompt list to regenerate (default: people)')
    parser.add_argument('--baseline-folder',
                        help='the stored shared baseline folder holding off_*.png')
    parser.add_argument('--out-folder',
                        help='where the two regenerated route folders and the JSON are written')
    parser.add_argument('--base-folder', default='assets',
                        help='the benchmark assets root, for the metadata (default: assets)')
    parser.add_argument('--seed', type=int, default=42,
                        help='the single seed to regenerate (default: 42)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='prompts per pipeline call; MUST match the run being reproduced '
                             '(default: 8)')
    parser.add_argument('--limit-prompts', type=int, default=None,
                        help='regenerate only the first N prompts. Must be a multiple of '
                             '--batch-size, so that every kept image occupies the same position in '
                             'the same batch as it did in the full run')
    parser.add_argument('--self-check', action='store_true',
                        help='run the mutation check on the comparison function and exit; needs '
                             'no graphics card, no model and no data')
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()

    missing = [name for name in ('baseline_folder', 'out_folder') if getattr(args, name) is None]
    if missing:
        parser.error('--' + ', --'.join(name.replace('_', '-') for name in missing) + ' is required '
                     '(or use --self-check)')
    if args.limit_prompts is not None and args.limit_prompts % args.batch_size != 0:
        parser.error(
            f'--limit-prompts {args.limit_prompts} is not a multiple of --batch-size '
            f'{args.batch_size}. A truncation inside a batch changes which noise each image gets, '
            'so the comparison against the stored run would be meaningless.'
        )
    if not os.path.isdir(args.baseline_folder):
        print(f'ERROR: no such baseline folder: {args.baseline_folder}')
        return 2

    prompts = build_prompts(args.task, args.base_folder, args.limit_prompts)
    if not prompts:
        print('ERROR: the prompt list is empty')
        return 2
    filenames = [f'off_{args.seed}_{prompt}.png' for prompt in prompts]
    os.makedirs(args.out_folder, exist_ok=True)

    print(f'=== generation route control: task={args.task} seed={args.seed} '
          f'batch={args.batch_size} prompts={len(prompts)}', flush=True)
    for route in (ROUTE_BASELINE, ROUTE_SESSION):
        generate_route(route, prompts, filenames, args.out_folder, args.seed, args.batch_size)

    regeneration = compare_folders(
        'route_baseline_versus_stored_baseline',
        os.path.join(args.out_folder, ROUTE_BASELINE), args.baseline_folder, filenames, prompts,
    )
    loaders = compare_folders(
        'route_session_versus_route_baseline',
        os.path.join(args.out_folder, ROUTE_SESSION),
        os.path.join(args.out_folder, ROUTE_BASELINE), filenames, prompts,
    )
    end_to_end = compare_folders(
        'route_session_versus_stored_baseline',
        os.path.join(args.out_folder, ROUTE_SESSION), args.baseline_folder, filenames, prompts,
    )
    verdict = decide(regeneration, loaders)

    payload: Dict[str, Any] = {
        'task': args.task,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'n_prompts': len(prompts),
        'model_base_name': MODEL_BASE_NAME,
        'baseline_folder': args.baseline_folder,
        'out_folder': args.out_folder,
        'comparisons': [regeneration.to_json(), loaders.to_json(), end_to_end.to_json()],
        'verdict': verdict,
    }
    result_path = os.path.join(args.out_folder, 'route_control.json')
    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print('--- results ---')
    for comparison in (regeneration, loaders, end_to_end):
        print(f'{comparison.name}: {comparison.n_identical} of {comparison.n_compared} '
              f'pixel-identical, max_abs={comparison.max_abs}, '
              f'mean_abs={comparison.mean_abs:.6f}, worst={comparison.worst_prompt!r}')
    print(f'VERDICT {verdict}')
    print(f'written {result_path}')
    return 0 if verdict == 'routes_agree' else 1


if __name__ == '__main__':
    sys.exit(main())
