"""The figures of the regeneration rehearsal's validation report, drawn from its outputs on disk.

Reads the folders the rehearsal scripts wrote under ``--s2`` (``determinism/``, ``routes/``,
``captures/``) and the rehearsal assets folder (``--assets``: the people baseline and the sessions),
and writes into ``--output``:

* ``differing_images_<comparison>.png`` -- for each comparison whose images are not all identical,
  every differing image: the reference, the candidate, and the absolute difference amplified, with
  its mean absolute difference on 0-255; and ``differing_images.md``, the table of every comparison;
* ``people_baseline_seed42.png`` -- the 100 people baseline images at seed 42;
* ``session_target_<entity>.png`` -- the target's prompt at every seed: baseline, then each method;
* ``session_receivers_seed42.png`` -- the first receivers other than the target at seed 42: baseline,
  then each method, with the session's per-pair scores in the subplot titles;
* ``training_runs_<method>.png`` -- with ``--pairs``, for each method whose session exists in both runs
  of a pair: every receiver's ``clip_diff`` in one run against the other, one panel per pair, and
  ``training_runs.md``, the table of every pair (Pearson correlation and mean absolute difference of the
  receivers' ``clip_diff`` and ``dino_diff``, receivers with equal values, the target's ``clip_diff`` in
  each run). A run is a folder under ``--s2`` whose ``assets/`` holds sessions of the same entity.

Usage::

    python report_figures.py --s2 S2_FOLDER --assets REHEARSAL_ASSETS --output DIR \\
        --methods uce distil salun --epochs 0 400 400 --index 0 [--pairs RUN_A:RUN_B ...]

Prints ``FIGURES <n>`` and ``FIGURES_DONE``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vision_unlearning.benchmarks.I_care.metadata import PER_PAIR_METRICS, get_interference_per_pair_path  # noqa: E402
from vision_unlearning.datasets.entity_names import canonical_entity, generation_prompt  # noqa: E402
from vision_unlearning.datasets.testbed import (  # noqa: E402
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_shared_baseline_folder,
)

DISPLAY = {'distil': 'spare', 'uce': 'uce', 'salun': 'salun'}
SEEDS = [42, 43, 44, 45]
AMPLIFY = 20


def _layout(figure: Any) -> None:
    """Tight layout below a fixed band for the figure title, so the title never covers the first row."""
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / figure.get_figheight()))


def _image(path: str) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert('RGB'))


def _comparisons(s2: str, prompts: List[str]) -> List[Tuple[str, str, str, str, int]]:
    """(name, reference folder, its lora state, candidate folder, prompts compared) for every comparison."""
    subset = os.path.join(s2, 'determinism', 'a', 'datasets', 'generated_people_baseline')
    found = [
        ('determinism_second_process', subset, 'off', os.path.join(s2, 'determinism', 'b', 'datasets', 'generated_people_baseline'), 16),
        ('other_a40', subset, 'off', os.path.join(s2, 'cards', 'other', 'datasets', 'generated_people_baseline'), 16),
        ('route_partial_weights', subset, 'on', os.path.join(s2, 'routes', 'route_partial_weights'), 16),
        ('route_adapter', subset, 'on', os.path.join(s2, 'routes', 'route_adapter'), 16),
        ('route_short_final_batch', subset, 'on', os.path.join(s2, 'routes', 'route_partial_weights_short'), 10),
        ('batch_1', subset, 'off', os.path.join(s2, 'captures', 'batch1', 'datasets', 'generated_people_baseline'), 16),
    ]
    return [entry for entry in found if os.path.isdir(entry[3])]


def differing_images(s2: str, prompts: List[str], output: str) -> List[str]:
    rows = ['| comparison | images compared | identical | mean absolute difference | worst image |', '|---|---|---|---|---|']
    written: List[str] = []
    for name, reference, state, candidate, count in _comparisons(s2, prompts):
        differing: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        values: List[float] = []
        for prompt in prompts[:count]:
            a = _image(os.path.join(reference, get_generated_dataset_file('off', 42, prompt)))
            b = _image(os.path.join(candidate, get_generated_dataset_file(state, 42, prompt)))  # type: ignore[arg-type]
            value = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
            values.append(value)
            if value > 0:
                differing.append((prompt, a, b, value))
        rows.append(f'| {name} | {count} | {count - len(differing)} | {np.mean(values):.4f} | {max(values):.4f} |')
        if not differing:
            continue
        # More than four differing images go into two groups of three columns, filled top to bottom, so
        # the figure keeps a page's proportions.
        groups = 2 if len(differing) > 4 else 1
        lines = -(-len(differing) // groups)
        figure, axes = plt.subplots(lines, 3 * groups, figsize=(9 * groups, 3.1 * lines), squeeze=False)
        for unused in axes.flat:
            unused.axis('off')
        for position, (prompt, a, b, value) in enumerate(differing):
            row, offset = position % lines, 3 * (position // lines)
            difference = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * AMPLIFY, 0, 255).astype(np.uint8)
            for column, (picture, title) in enumerate(((a, f'reference\n{prompt}'), (b, f'{name}\nseed=42'),
                                                       (difference, f'absolute difference x{AMPLIFY}\nmean={value:.3f} on 0-255'))):
                axes[row][offset + column].imshow(picture)
                axes[row][offset + column].set_title(title, fontsize=8)
        figure.suptitle(f'{name} | reference=baseline subset seed=42 | differing images={len(differing)}/{count}', fontsize=9)
        _layout(figure)
        path = os.path.join(output, f'differing_images_{name}.png')
        figure.savefig(path, dpi=80)
        plt.close(figure)
        written.append(path)
    with open(os.path.join(output, 'differing_images.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(rows) + '\n')
    return written


def baseline_sheet(assets: str, names: List[str], output: str) -> str:
    folder = get_shared_baseline_folder('people', assets)
    figure, axes = plt.subplots(10, 10, figsize=(15, 16.5))
    for axis, name in zip(axes.flat, names):
        axis.imshow(_image(os.path.join(folder, get_generated_dataset_file('off', 42, generation_prompt('people', name)))))
        axis.set_title(canonical_entity('people', name), fontsize=6)
        axis.axis('off')
    figure.suptitle(f'people baseline | seed=42 | entities={len(names)} | batch=8 | noise=per_image_processor', fontsize=10)
    _layout(figure)
    path = os.path.join(output, 'people_baseline_seed42.png')
    figure.savefig(path, dpi=70)
    plt.close(figure)
    return path


def session_target(assets: str, target: str, methods: List[Tuple[str, int]], output: str) -> str:
    prompt = generation_prompt('people', target)
    rows: List[Tuple[str, str, str]] = [('baseline', get_shared_baseline_folder('people', assets), 'off')]
    rows += [(f'{DISPLAY[m]} epochs={e}', get_generated_dataset_folder('people', m, e, canonical_entity('people', target), assets), 'on')  # type: ignore[arg-type]
             for m, e in methods]
    figure, axes = plt.subplots(len(rows), len(SEEDS), figsize=(2.6 * len(SEEDS), 2.8 * len(rows)), squeeze=False)
    for r, (label, folder, state) in enumerate(rows):
        for c, seed in enumerate(SEEDS):
            axes[r][c].imshow(_image(os.path.join(folder, get_generated_dataset_file(state, seed, prompt))))  # type: ignore[arg-type]
            axes[r][c].set_title(f'{label}\nseed={seed}', fontsize=8)
            axes[r][c].axis('off')
    figure.suptitle(f'target={canonical_entity("people", target)} | prompt="{prompt}" | rows=baseline then each method', fontsize=9)
    _layout(figure)
    path = os.path.join(output, f'session_target_{target}.png')
    figure.savefig(path, dpi=70)
    plt.close(figure)
    return path


def session_receivers(assets: str, target: str, index: int, receivers: List[str], methods: List[Tuple[str, int]], output: str) -> str:
    rows: List[Tuple[str, str, str, Dict[str, Dict[str, float]]]] = [('baseline', get_shared_baseline_folder('people', assets), 'off', {})]
    for m, e in methods:
        with open(get_interference_per_pair_path('people', index, m, e, base_folder=assets), encoding='utf-8') as handle:  # type: ignore[arg-type]
            scores = json.load(handle)
        rows.append((f'{DISPLAY[m]} epochs={e}', get_generated_dataset_folder('people', m, e, canonical_entity('people', target), assets), 'on', scores))  # type: ignore[arg-type]
    figure, axes = plt.subplots(len(rows), len(receivers), figsize=(2.6 * len(receivers), 3.0 * len(rows)), squeeze=False)
    for r, (label, folder, state, scores) in enumerate(rows):
        for c, name in enumerate(receivers):
            axes[r][c].imshow(_image(os.path.join(folder, get_generated_dataset_file(state, 42, generation_prompt('people', name)))))  # type: ignore[arg-type]
            second = '' if not scores else f"\nclip_diff={scores[name]['clip_diff']:.2f} ssim={scores[name]['ssim']:.2f}"
            axes[r][c].set_title(f'{label} | {canonical_entity("people", name)}{second}', fontsize=7)
            axes[r][c].axis('off')
    figure.suptitle(f'target={canonical_entity("people", target)} | receivers={len(receivers)} | seed=42 | scores=mean over 4 seeds', fontsize=9)
    _layout(figure)
    path = os.path.join(output, 'session_receivers_seed42.png')
    figure.savefig(path, dpi=70)
    plt.close(figure)
    return path


def _pearson(a: List[float], b: List[float]) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def training_runs(s2: str, target: str, index: int, pairs: List[Tuple[str, str]], methods: List[Tuple[str, int]], output: str) -> List[str]:
    """Each trained session against its retraining in another run: the receivers' scores, pair by pair."""
    rows = ['| method | run A | run B | receivers | equal clip_diff | clip_diff Pearson | clip_diff mean absolute difference '
            '| dino_diff Pearson | dino_diff mean absolute difference | target clip_diff A | target clip_diff B |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    written: List[str] = []
    for method, epochs in methods:
        found = []
        for a, b in pairs:
            paths = [get_interference_per_pair_path('people', index, method, epochs, base_folder=os.path.join(s2, run, 'assets'))  # type: ignore[arg-type]
                     for run in (a, b)]
            if all(os.path.exists(p) for p in paths):
                scores = []
                for p in paths:
                    with open(p, encoding='utf-8') as handle:
                        scores.append(json.load(handle))
                found.append((a, b, scores[0], scores[1]))
        if not found:
            continue
        columns = min(4, len(found))
        lines = -(-len(found) // columns)
        figure, axes = plt.subplots(lines, columns, figsize=(3.4 * columns, 3.6 * lines), squeeze=False)
        for unused in axes.flat[len(found):]:
            unused.axis('off')
        for c, (a, b, first, second) in enumerate(found):
            receivers = sorted(n for n in first if n != target)
            x = [first[n]['clip_diff'] for n in receivers]
            y = [second[n]['clip_diff'] for n in receivers]
            dino_x = [first[n]['dino_diff'] for n in receivers]
            dino_y = [second[n]['dino_diff'] for n in receivers]
            equal = sum(1 for u, v in zip(x, y) if u == v)
            clip_mad = float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))
            dino_mad = float(np.mean(np.abs(np.asarray(dino_x) - np.asarray(dino_y))))
            axis = axes[c // columns][c % columns]
            low, high = min(x + y), max(x + y)
            axis.plot([low, high], [low, high], color='grey', linewidth=0.8)
            axis.scatter(x, y, s=8)
            axis.set_xlabel(f'{a}: receiver clip_diff', fontsize=8)
            axis.set_ylabel(f'{b}: receiver clip_diff', fontsize=8)
            axis.set_title(f'{a} against {b}\nPearson={_pearson(x, y):.3f} mean absolute difference={clip_mad:.3f}\n'
                           f'equal={equal} of {len(receivers)}', fontsize=7)
            axis.tick_params(labelsize=7)
            rows.append(f'| {DISPLAY[method]} | {a} | {b} | {len(receivers)} | {equal} | {_pearson(x, y):.3f} | {clip_mad:.3f} '
                        f'| {_pearson(dino_x, dino_y):.3f} | {dino_mad:.3f} | {first[target]["clip_diff"]:.2f} | {second[target]["clip_diff"]:.2f} |')
        figure.suptitle(f'method={DISPLAY[method]} epochs={epochs} | target={canonical_entity("people", target)} | '
                        f'points=receivers | line=equal values', fontsize=9)
        _layout(figure)
        path = os.path.join(output, f'training_runs_{DISPLAY[method]}.png')
        figure.savefig(path, dpi=90)
        plt.close(figure)
        written.append(path)
    with open(os.path.join(output, 'training_runs.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(rows) + '\n')
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--s2', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--methods', nargs='+', default=['uce', 'distil', 'salun'])
    parser.add_argument('--epochs', nargs='+', type=int, default=[0, 400, 400])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--receivers', type=int, default=6)
    parser.add_argument('--pairs', nargs='*', default=[], help='RUN_A:RUN_B, folders under --s2 holding assets/')
    args = parser.parse_args()
    if len(args.methods) != len(args.epochs):
        parser.error('one --epochs value per --methods value')
    if any(pair.count(':') != 1 for pair in args.pairs):
        parser.error('each --pairs value is RUN_A:RUN_B')

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.assets, 'metadata_people_2_enriched_filtered.json'), encoding='utf-8') as handle:
        names = [entity['name'] for entity in json.load(handle)]
    prompts = [generation_prompt('people', name) for name in names]
    target = names[args.index]
    methods = list(zip(args.methods, args.epochs))
    written = differing_images(args.s2, prompts, args.output)
    written.append(baseline_sheet(args.assets, names, args.output))
    written.append(session_target(args.assets, target, methods, args.output))
    written.append(session_receivers(args.assets, target, args.index, [n for n in names if n != target][:args.receivers], methods, args.output))
    pairs = [(a, b) for a, b in (pair.split(':') for pair in args.pairs)]
    written += training_runs(args.s2, target, args.index, pairs, methods, args.output)
    for path in written:
        print('FIGURE', path)
    print(f'FIGURES {len(written)}')
    print('FIGURES_DONE')


if __name__ == '__main__':
    main()
