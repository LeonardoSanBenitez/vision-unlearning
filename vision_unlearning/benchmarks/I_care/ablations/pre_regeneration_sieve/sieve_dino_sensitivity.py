"""How much of ``dino_diff`` survives when the two images do not share initial noise.

``dino_diff`` is the DINOv2 cosine between a receiver's baseline image and its image from an
unlearned model. If the two are drawn from different initial latents, the metric is no longer
"how much did this picture change"; the question is what it becomes, because a semantic
embedding is largely blind to the noise draw while a pixel metric is not.

Four populations are scored, all with the same model and transform:

* **different prompt**   -- two baseline images of different concepts. The floor.
* **same prompt, other seed** -- two baseline images of the same concept from different seeds.
  This is what "same concept, unrelated noise" scores, and it is the right reference for a
  corpus whose pairs do not share noise.
* **untouched receiver** -- the on/off pair of a receiver whose clip score did not change.
* **the erased entity**  -- the on/off pair of the entity that was actually unlearned.

If the untouched receivers sit at the same-prompt-other-seed level and the erased entity sits
below it, the metric still separates "concept preserved" from "concept destroyed" and its loss
is variance rather than meaning. If the untouched receivers sit at the different-prompt floor,
it separates nothing.

Needs torch and DINOv2 from the hub. Run::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_dino_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))
TASK = 'breeds'
EMITTER_INDEX = 0
METHOD = 'distil'
EPOCHS = 100
ENTITY_FOLDER = 'generated_breeds_a dogo argentino_distil_100'
BASELINE_FOLDER = 'generated_breeds_baseline'


def _prompt_of(name: str) -> str:
    article = 'an' if name[0].lower() in 'aeiou' else 'a'
    body = ' '.join(f'{article} {name}'.replace('_', ' ').split())
    return f'An image of {body}'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--tolerance', type=float, default=0.25,
                        help='|clip_diff| below which a receiver counts as semantically untouched.')
    parser.add_argument('--limit', type=int, default=12, help='Receivers per population.')
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_dino_sensitivity.json'))
    args = parser.parse_args(argv)

    from PIL import Image  # noqa: PLC0415
    from vision_unlearning.benchmarks.I_care.embeddings import (  # noqa: PLC0415
        compute_dino_image_similarity, load_dino_model,
    )

    datasets = os.path.join(args.assets, 'datasets')
    entity_dir = os.path.join(datasets, ENTITY_FOLDER)
    baseline_dir = os.path.join(datasets, BASELINE_FOLDER)
    with open(os.path.join(args.assets, f'metadata_{TASK}_2_enriched_filtered.json'),
              'r', encoding='utf-8') as handle:
        names = [entry['name'] for entry in json.load(handle)]
    with open(os.path.join(datasets,
              f'interferences_caused_by_{TASK}_{EMITTER_INDEX}_{METHOD}_{EPOCHS}.json'),
              'r', encoding='utf-8') as handle:
        per_pair: Dict[str, Dict[str, float]] = json.load(handle)

    model, transform, device = load_dino_model()

    def cosine(left: str, right: str) -> float:
        return compute_dino_image_similarity(
            Image.open(left), Image.open(right), model, transform, device,
        )

    untouched = [n for n in names
                 if n in per_pair and abs(per_pair[n].get('clip_diff', 9e9)) <= args.tolerance]
    untouched = untouched[:args.limit]
    target = names[EMITTER_INDEX]

    populations: Dict[str, List[float]] = {
        'different_prompt': [],
        'same_prompt_other_seed': [],
        'untouched_receiver_on_vs_off': [],
        'erased_entity_on_vs_off': [],
    }
    per_receiver: Dict[str, Dict[str, float]] = {}

    for index, name in enumerate(untouched):
        prompt = _prompt_of(name)
        other = _prompt_of(untouched[(index + 1) % len(untouched)])
        off_42 = os.path.join(baseline_dir, f'off_42_{prompt}.png')
        off_43 = os.path.join(baseline_dir, f'off_43_{prompt}.png')
        other_42 = os.path.join(baseline_dir, f'off_42_{other}.png')
        on_42 = os.path.join(entity_dir, f'on_42_{prompt}.png')
        if not all(os.path.exists(p) for p in (off_42, off_43, other_42, on_42)):
            continue
        values = {
            'different_prompt': cosine(off_42, other_42),
            'same_prompt_other_seed': cosine(off_42, off_43),
            'untouched_receiver_on_vs_off': cosine(off_42, on_42),
        }
        for key, value in values.items():
            populations[key].append(value)
        per_receiver[name] = {**values, 'clip_diff': per_pair[name]['clip_diff']}

    target_prompt = _prompt_of(target)
    target_off = os.path.join(baseline_dir, f'off_42_{target_prompt}.png')
    target_on = os.path.join(entity_dir, f'on_42_{target_prompt}.png')
    if os.path.exists(target_off) and os.path.exists(target_on):
        populations['erased_entity_on_vs_off'].append(cosine(target_off, target_on))

    report: Dict[str, Any] = {
        'task': TASK, 'emitter': target, 'method': METHOD, 'epochs': EPOCHS,
        'tolerance_on_clip_diff': args.tolerance,
        'receivers': list(per_receiver),
        'per_receiver': per_receiver,
        'populations': {
            name: {
                'n': len(series),
                'mean': statistics.fmean(series) if series else None,
                'min': min(series) if series else None,
                'max': max(series) if series else None,
            }
            for name, series in populations.items()
        },
    }

    print(f'emitter: {target}   method: {METHOD}   receivers scored: {len(per_receiver)}')
    print(f'{"population":<32}{"n":>4}{"mean":>10}{"min":>10}{"max":>10}   (DINOv2 cosine)')
    for name in ('different_prompt', 'same_prompt_other_seed',
                 'untouched_receiver_on_vs_off', 'erased_entity_on_vs_off'):
        entry = report['populations'][name]
        if entry['mean'] is None:
            print(f'{name:<32}{entry["n"]:>4}{"-":>10}')
            continue
        print(f'{name:<32}{entry["n"]:>4}{entry["mean"]:>10.4f}'
              f'{entry["min"]:>10.4f}{entry["max"]:>10.4f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nDINO_SENSITIVITY_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
