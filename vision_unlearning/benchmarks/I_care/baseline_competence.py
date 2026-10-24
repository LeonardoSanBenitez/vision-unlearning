"""Baseline competence: whether the scorer recognises each entity in its own baseline images, and ``f_min``.

For every baseline image of a task -- entity ``i`` at seed ``s`` -- this scores the image against
**every** prompt of the task, with the scorer ``pipeline_06`` uses for ``clip_diff`` and called the way
``pipeline_06`` calls it: ``MetricImageTextSimilarity(metrics=['clip']).score_batch_same_text`` over one
entity's seed images, in seed order, for one prompt. An entity's score against its own prompt is
therefore the same number ``pipeline_06`` records as that receiver's ``clip_off``.

From that table, per entity and seed (:func:`summarise`):

* **the own score** and **the own prompt's rank** among the task's prompts, 1 being the highest score;
  ties count against the entity (a prompt scoring the same as the own prompt ranks above it);
* **the specificity**, the own score minus the mean score against the other prompts
  (``equalization.specificity``);

and per task the two quantities equalization reads before any calibration session runs:

* **the screen** (``equalization.competence_screen``): how many entities have their own prompt among
  the top 5 in at least 3 of the 4 seeds. It never excludes an entity;
* **the scale reference** ``f_min`` (``equalization.scale_reference``): half the mean specificity over
  every entity and seed. It raises when it is not finite and strictly positive, and equalization does
  not start for that task.

The file is ``assets/datasets/baseline_competence_{task}{model_segment}.json``. It holds the full
table (every score of every image against every prompt), the per-image summary, the screen, ``f_min``,
and what ties it to its images: the seeds, the ordered prompt list's hash, and the sha256 of the
baseline's ``provenance.json``. ``pipeline_09 --stages baseline-competence`` uploads it.

Run ``python -m vision_unlearning.benchmarks.I_care.baseline_competence --help``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from vision_unlearning.benchmarks.I_care.configuration import GENERATE_DATASET_SEEDS, model_segment
from vision_unlearning.benchmarks.I_care.equalization import (
    IntegrityFailure,
    competence_screen,
    scale_reference,
    specificity,
)
from vision_unlearning.benchmarks.I_care.provenance import PROVENANCE_FILENAME, file_sha256, prompt_list_sha256, read_provenance
from vision_unlearning.datasets.entity_names import generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_shared_baseline_folder

SCHEMA_VERSION = 1
SCORER = "openai/clip-vit-base-patch16, MetricImageTextSimilarity.score_batch_same_text over one entity's seed images"


def competence_path(task: str, base_folder: str = 'assets', model: str = 'sd1.4') -> str:
    """Where a task's baseline competence file lives."""
    return os.path.join(base_folder, 'datasets', f'baseline_competence_{task}{model_segment(model)}.json')  # type: ignore[arg-type]


def summarise(
    entities: Sequence[str],
    seeds: Sequence[int],
    scores: Mapping[str, Mapping[int, Sequence[float]]],
) -> Dict[str, Any]:
    """Own score, rank and specificity per image; the screen and ``f_min`` per task.

    ``scores[entity][seed]`` is that image's score against every prompt, in the order of
    ``entities`` (prompt ``k`` is entity ``k``'s). A missing image or a list of the wrong length
    raises :class:`IntegrityFailure`.
    """
    per_entity: Dict[str, Any] = {}
    ranks: Dict[str, List[int]] = {}
    specificities: List[float] = []
    for index, entity in enumerate(entities):
        per_seed: Dict[str, Any] = {}
        ranks[entity] = []
        for seed in seeds:
            row = list(scores.get(entity, {}).get(seed, []))
            if len(row) != len(entities):
                raise IntegrityFailure(f'{entity} at seed {seed}: {len(row)} scores for {len(entities)} prompts')
            own = row[index]
            others = row[:index] + row[index + 1:]
            rank = 1 + sum(1 for value in others if value >= own)
            value = specificity(own, others)
            per_seed[str(seed)] = {'own': own, 'rank': rank, 'specificity': value}
            ranks[entity].append(rank)
            specificities.append(value)
        per_entity[entity] = per_seed
    return {
        'per_image': per_entity,
        'screen': competence_screen(ranks),
        'mean_specificity': sum(specificities) / len(specificities),
        'f_min': scale_reference(specificities),
    }


def score_baseline(
    task: type_entity_task,
    entities: Sequence[str],
    seeds: Sequence[int],
    base_folder: str,
    scorer: Any = None,
) -> Dict[str, Dict[int, List[float]]]:
    """Score every baseline image against every prompt of the task. Returns ``scores[entity][seed]``.

    ``scorer`` defaults to the CLIP scorer ``pipeline_06`` uses; it needs a card to be fast.
    """
    from PIL import Image
    if scorer is None:
        from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity
        scorer = MetricImageTextSimilarity(metrics=['clip'])
    prompts = [generation_prompt(task, name) for name in entities]
    folder = get_shared_baseline_folder(task, base_folder)
    scores: Dict[str, Dict[int, List[float]]] = {}
    for entity, own_prompt in zip(entities, prompts):
        images = [Image.open(os.path.join(folder, get_generated_dataset_file('off', seed, own_prompt))) for seed in seeds]
        by_seed: Dict[int, List[float]] = {seed: [] for seed in seeds}
        for prompt in prompts:
            results = scorer.score_batch_same_text(images, prompt)
            for seed, result in zip(seeds, results):
                by_seed[seed].append(float(result['clip']))
        scores[entity] = by_seed
    return scores


def write_competence(
    task: type_entity_task,
    base_folder: str,
    seeds: Optional[Sequence[int]] = None,
    scorer: Any = None,
) -> Dict[str, Any]:
    """Score the task's baseline, summarise it, and write the competence file. Returns what was written.

    Refuses a baseline whose ``provenance.json`` records another prompt list or other seeds.
    """
    seeds = list(GENERATE_DATASET_SEEDS) if seeds is None else list(seeds)
    with open(os.path.join(base_folder, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        entities = [entity['name'] for entity in json.load(handle)]
    prompts = [generation_prompt(task, name) for name in entities]
    folder = get_shared_baseline_folder(task, base_folder)
    read_provenance(folder, prompts, seeds)
    scores = score_baseline(task, entities, seeds, base_folder, scorer)
    record = {
        'schema_version': SCHEMA_VERSION,
        'task': task,
        'seeds': seeds,
        'entities': list(entities),
        'prompt_list_sha256': prompt_list_sha256(prompts),
        'baseline_provenance_sha256': file_sha256(os.path.join(folder, PROVENANCE_FILENAME)),
        'scorer': SCORER,
        **summarise(entities, seeds, scores),
        'scores': {entity: {str(seed): row for seed, row in by_seed.items()} for entity, by_seed in scores.items()},
    }
    path = competence_path(task, base_folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m vision_unlearning.benchmarks.I_care.baseline_competence', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Prints one line, COMPETENCE task=<t> recognised=<n>/<N> passes=<yes|no> f_min=<v> '
               'mean_specificity=<v> path=<file>. Exit status: 0 written (a failed screen is still 0: it is '
               'investigated, never an error); 1 integrity failure (a missing score, or f_min not finite and '
               'positive), nothing written; 2 bad arguments.')
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--base-folder', default='assets')
    args = parser.parse_args(argv)
    try:
        record = write_competence(args.task, args.base_folder)
    except IntegrityFailure as exc:
        print(f'COMPETENCE_FAILED task={args.task} {exc}')
        return 1
    screen = record['screen']
    print(f"COMPETENCE task={args.task} recognised={screen['recognised']}/{screen['entities']} "
          f"passes={'yes' if screen['passes'] else 'no'} f_min={record['f_min']:.4f} "
          f"mean_specificity={record['mean_specificity']:.4f} path={competence_path(args.task, args.base_folder)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
