"""Rehearsal: generate a baseline subset through the benchmark's baseline route.

The first ``--n-prompts`` prompts of a task, at the given seeds, through
``GeneratedDataset._compute_from_scratch`` -- the code the baseline is produced by -- under the
benchmark's noise convention and batch size, with ``provenance.json``. Run it twice in two processes
and compare the folders with ``compare_folders.py`` to check generation is deterministic; run it on
two cards to check they agree.

Usage::

    python generate_subset.py --task people --n-prompts 16 --seeds 42 --output DIR [--assets ASSETS]

Prints ``SUBSET_DONE <n images>``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, cast

from vision_unlearning.benchmarks.I_care.configuration import GENERATION_BATCH_SIZE, GENERATION_NOISE
from vision_unlearning.benchmarks.I_care.provenance import producer_facts
from vision_unlearning.datasets.entity_names import generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import GeneratedDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--n-prompts', type=int, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', required=True, help='an empty folder; the images land in its datasets/generated_<task>_baseline/')
    parser.add_argument('--assets', default='assets', help='where the task metadata file is read from')
    args = parser.parse_args()

    task = cast(type_entity_task, args.task)
    with open(os.path.join(args.assets, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        names = [entity['name'] for entity in json.load(handle)]
    prompts: List[str] = [generation_prompt(task, name) for name in names[:args.n_prompts]]
    dataset = GeneratedDataset(
        task=task, base_folder=args.output, noise=GENERATION_NOISE,
        provenance=producer_facts(precision='float16', producer='rehearsal generate_subset', task=task),
    )
    folder = dataset._compute_from_scratch(args.seeds, prompts, batch_size=GENERATION_BATCH_SIZE)
    count = sum(1 for name in os.listdir(folder) if name.endswith('.png'))
    print(f'SUBSET_DONE {count}')


if __name__ == '__main__':
    main()
