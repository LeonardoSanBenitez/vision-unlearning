"""The figures of the regeneration's second validation report (the baselines), drawn from the campaign's files.

Reads the campaign assets folder (``--assets``) and writes into ``--output``:

* ``baseline_{task}_seed{seed}.png`` -- the 100 baseline images of one task at one seed, titled with each
  entity's canonical name and its own prompt's rank among the task's prompts at that seed;
* ``competence_{task}.png`` -- from the task's competence file: the own prompt's rank over the 400 images
  (1 is the highest score among the task's 100 prompts), and the specificity over the 400 images with
  half its mean, the scale reference ``f_min``, marked;
* ``ladder_{task}_{method}_{entity}.png`` (with ``--ladder``) -- one calibration entity's own prompt at every measured
  dial of a method, the baseline first, each row titled with its forget effect ``F``;
* ``competence.md`` -- per task, the screen's count, the scale reference, the mean specificity, and the
  entities the screen does not recognise.

Usage::

    python report_02_figures.py --assets CAMPAIGN_ASSETS --output DIR [--tasks people scenes breeds]

Prints ``FIGURES <n>`` and ``FIGURES_DONE``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vision_unlearning.benchmarks.I_care.baseline_competence import competence_path  # noqa: E402
from vision_unlearning.datasets.entity_names import canonical_entity, generation_prompt, type_entity_task  # noqa: E402
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_shared_baseline_folder  # noqa: E402

SEEDS = [42, 43, 44, 45]


def _layout(figure: Any) -> None:
    """Tight layout below a fixed band for the figure title, so the title never covers the first row."""
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / figure.get_figheight()))


def baseline_sheet(assets: str, task: type_entity_task, seed: int, competence: Dict[str, Any], output: str) -> str:
    """The task's 100 baseline images at one seed, each titled with its entity and its own prompt's rank."""
    folder = get_shared_baseline_folder(task, assets)
    names = competence['entities']
    figure, axes = plt.subplots(10, 10, figsize=(15, 17))
    for axis, name in zip(axes.flat, names):
        with Image.open(os.path.join(folder, get_generated_dataset_file('off', seed, generation_prompt(task, name)))) as handle:
            axis.imshow(np.asarray(handle.convert('RGB')))
        rank = competence['per_image'][name][str(seed)]['rank']
        axis.set_title(f'{canonical_entity(task, name)}\nrank {rank}', fontsize=6)
        axis.axis('off')
    figure.suptitle(f'{task} baseline | seed={seed} | entities={len(names)} | batch=8 | noise=per_image_processor | '
                    f'rank = own prompt among the {len(names)} prompts, 1 highest', fontsize=10)
    _layout(figure)
    path = os.path.join(output, f'baseline_{task}_seed{seed}.png')
    figure.savefig(path, dpi=70)
    plt.close(figure)
    return path


def competence_figure(task: str, competence: Dict[str, Any], output: str) -> str:
    """Own-prompt ranks and specificities over every image of the task, with f_min marked."""
    images = [competence['per_image'][name][str(seed)] for name in competence['entities'] for seed in competence['seeds']]
    ranks = [image['rank'] for image in images]
    specificities = [image['specificity'] for image in images]
    screen, f_min = competence['screen'], competence['f_min']
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    left.hist(ranks, bins=np.arange(0.5, max(ranks) + 1.5, 1), color='tab:blue')
    left.set_xlabel('rank of the own prompt among the task prompts\n(1 = highest score)')
    left.set_ylabel('images')
    left.set_title(f'n={len(ranks)} images | rank 1: {sum(1 for r in ranks if r == 1)} | rank at most 5: '
                   f'{sum(1 for r in ranks if r <= 5)}\nrecognised entities {screen["recognised"]}/{screen["entities"]} '
                   f'(threshold {screen["threshold"]})', fontsize=9)
    right.hist(specificities, bins=40, color='tab:gray')
    right.axvline(f_min, color='tab:red')
    right.set_xlabel('specificity: own-prompt CLIP score\nminus the mean score against the other prompts')
    right.set_ylabel('images')
    right.set_title(f'n={len(specificities)} images | mean {competence["mean_specificity"]:.3f} | '
                    f'f_min = half the mean = {f_min:.3f} (red)', fontsize=9)
    figure.suptitle(f'baseline competence | task={task} | scorer=CLIP ViT-B/16 | seeds={competence["seeds"]}', fontsize=10)
    _layout(figure)
    path = os.path.join(output, f'competence_{task}.png')
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return path


def ladder_targets(assets: str, root: str, task: type_entity_task, method: str, entity: str, output: str) -> str:
    """One calibration entity's own prompt at every measured dial of a method: the baseline, then a row per dial."""
    from vision_unlearning.benchmarks.I_care import calibration
    from vision_unlearning.benchmarks.I_care.equalization import forget_effect
    from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_pair_path, per_seed_path_of
    from vision_unlearning.datasets.testbed import get_generated_dataset_folder
    with open(os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        entities = [e['name'] for e in json.load(handle)]
    index = entities.index(entity)
    prompt = generation_prompt(task, entity)
    dials = calibration.measured_dials(root, task, method)
    figure, axes = plt.subplots(len(dials) + 1, len(SEEDS), figsize=(2.2 * len(SEEDS), 2.4 * (len(dials) + 1)), squeeze=False)
    rows: List[Tuple[str, str, str, Optional[float]]] = [('baseline', get_shared_baseline_folder(task, assets), 'off', None)]
    for dial in dials:
        base = calibration.readout_assets(root, task, method, dial)
        epochs = calibration.epochs_of(method, dial)
        folder = get_generated_dataset_folder(task, method, epochs, canonical_entity(task, entity), base)  # type: ignore[arg-type]
        per_seed_path = per_seed_path_of(get_interference_per_pair_path(task, index, method, epochs, base_folder=base))  # type: ignore[arg-type]
        effect = None
        if os.path.isfile(per_seed_path):
            with open(per_seed_path, encoding='utf-8') as handle:
                row = json.load(handle)['receivers'][entity]
            effect = forget_effect(row['clip_off'], row['clip_on'])
        rows.append((f'{method} dial={calibration.dial_name(dial)}', folder, 'on', effect))
    for r, (label, folder, state, effect) in enumerate(rows):
        for c, seed in enumerate(SEEDS):
            axis = axes[r][c]
            path = os.path.join(folder, get_generated_dataset_file(state, seed, prompt))  # type: ignore[arg-type]
            if os.path.isfile(path):
                with Image.open(path) as handle:
                    axis.imshow(np.asarray(handle.convert('RGB')))
            title = f'{label} seed={seed}' if effect is None else f'{label} F={effect:.2f} seed={seed}'
            axis.set_title(title, fontsize=6)
            axis.axis('off')
    figure.suptitle(f'calibration | task={task} | method={method} | target={canonical_entity(task, entity)} | '
                    f'F = mean over seeds of own-prompt CLIP lost (row)', fontsize=9)
    _layout(figure)
    path = os.path.join(output, f'ladder_{task}_{method}_{entity}.png')
    figure.savefig(path, dpi=80)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', nargs='+', default=['people', 'scenes', 'breeds'])
    parser.add_argument('--ladder', nargs=3, metavar=('CALIBRATION_ROOT', 'METHOD', 'ENTITY'), default=None,
                        help='draw only ladder_<task>_<method>_<entity>.png, for the first of --tasks')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.ladder is not None:
        root, method, entity = args.ladder
        print(ladder_targets(args.assets, root, args.tasks[0], method, entity, args.output))
        print('FIGURES_DONE')
        return
    paths: List[str] = []
    table = ['| task | recognised entities | threshold | f_min | mean specificity | not recognised |', '|---|---|---|---|---|---|']
    for task in args.tasks:
        with open(competence_path(task, args.assets), encoding='utf-8') as handle:
            competence = json.load(handle)
        for seed in SEEDS:
            paths.append(baseline_sheet(args.assets, task, seed, competence, args.output))
        paths.append(competence_figure(task, competence, args.output))
        screen = competence['screen']
        table.append(f"| {task} | {screen['recognised']}/{screen['entities']} | {screen['threshold']} | {competence['f_min']:.4f} | "
                     f"{competence['mean_specificity']:.4f} | {', '.join(screen['not_recognised']) or 'none'} |")
    with open(os.path.join(args.output, 'competence.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(table) + '\n')
    print(f'FIGURES {len(paths)}')
    print('FIGURES_DONE')


if __name__ == '__main__':
    main()
