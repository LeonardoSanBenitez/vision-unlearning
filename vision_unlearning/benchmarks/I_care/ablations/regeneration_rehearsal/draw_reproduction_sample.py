"""Draw the sessions an independent re-execution of the I-CARE corpus reproduces, per reproduction tier.

The corpus declares, before its sessions run, a tolerance per reproduction tier and the sample each tier is
tested on. The four tiers:

* **T0** published per-pair, metadata and similarity files -> everything derived from them. Sample: everything,
  so nothing is drawn.
* **T1** published images -> per-pair files. One session per (task, method): 9.
* **T2** published model weights -> images, on a card model other than the one that produced them. One session
  per method, its task drawn: 3.
* **T3** the recipe and the pinned base model -> a trained session, then its per-pair file. One session per
  method, its task drawn: 3.

Every draw uses ``random.Random`` seeded with a string naming the seed, the tier, the method and (for T1) the
task, the same convention as ``equalization.draw_panels``, so any one draw can be repeated alone and adding a
tier never moves another. Entities are drawn from all 100 of a task, in the metadata's order, by name; no entity
is drawn by a score.

Usage::

    python draw_reproduction_sample.py --assets ASSETS --output sample.json [--seed 20261027]

Reads ``ASSETS/metadata_{task}_2_enriched_filtered.json`` for the three tasks. Writes ``--output``: the seed and,
per tier, a list of sessions ``{"task", "method", "index", "entity"}``. Prints one line per session and
``SAMPLE_DONE <n>`` last. Exit codes: 0 written; 2 a metadata file is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

TASKS = ['people', 'scenes', 'breeds']
METHODS = ['uce', 'distil', 'salun']
DEFAULT_SEED = 20261027


def draw(entities: Dict[str, List[str]], seed: int) -> Dict[str, List[Dict[str, object]]]:
    """The sessions of tiers T1, T2 and T3, from each task's entity names in metadata order."""
    def session(rng: random.Random, task: str, method: str) -> Dict[str, object]:
        index = rng.randrange(len(entities[task]))
        return {'task': task, 'method': method, 'index': index, 'entity': entities[task][index]}

    sample: Dict[str, List[Dict[str, object]]] = {'T1': [], 'T2': [], 'T3': []}
    for task in TASKS:
        for method in METHODS:
            sample['T1'].append(session(random.Random(f'{seed}:T1:{task}:{method}'), task, method))
    for tier in ('T2', 'T3'):
        for method in METHODS:
            rng = random.Random(f'{seed}:{tier}:{method}')
            sample[tier].append(session(rng, rng.choice(TASKS), method))
    return sample


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', required=True, help='The I-CARE assets folder holding the three metadata files.')
    parser.add_argument('--output', required=True, help='Where to write the sample (JSON).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    entities: Dict[str, List[str]] = {}
    for task in TASKS:
        path = os.path.join(args.assets, f'metadata_{task}_2_enriched_filtered.json')
        if not os.path.isfile(path):
            print(f'missing {path}', file=sys.stderr)
            return 2
        with open(path, encoding='utf-8') as handle:
            entities[task] = [entry['name'] for entry in json.load(handle)]
    sample = draw(entities, args.seed)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump({'seed': args.seed, **sample}, handle, indent=2)
    for tier, sessions in sample.items():
        for s in sessions:
            print(tier, s['task'], s['method'], s['index'], s['entity'])
    print(f"SAMPLE_DONE {sum(len(s) for s in sample.values())}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
