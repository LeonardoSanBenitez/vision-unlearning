"""How many receivers did this session damage, and in which of two ways? Counted, not eyeballed.

A session targets one entity. The other 99 are **receivers** and are supposed to survive. Looking at
one of them and reporting what it looked like is an anecdote; this counts all of them.

Two failure modes have been seen by eye and they are different things, so they are counted
separately rather than lumped into "interference":

* **bled** -- the receiver became the *substitute concept*. Tony Blair rendered as a child. The
  image is a perfectly good photograph; it is simply of the wrong subject. A metric that only asks
  "how far did this image move" cannot tell this from ordinary drift, which is exactly how it went
  unnoticed: in an earlier session the identity that had been replaced scored the *smallest* change
  of the four inspected.
* **destroyed** -- the receiver became nothing. A flat texture with no subject. Seen in the
  published corpus, where a receiver of an old UCE session is an olive smear.

**How they are told apart.** Every image is scored with CLIP against two texts: the receiver's own
generation prompt, and the same template filled with the task's substitute concept. Each receiver
then has two deltas, `on` minus `off`:

* ``delta_own``       -- did it stop looking like itself?
* ``delta_substitute``-- did it start looking like the substitute?

Bleeding moves the second one up while the first goes down. Destruction moves both down: the image
resembles nothing in particular. Preservation moves neither.

**The threshold is measured, not chosen.** The scale of "no change" is taken from the `off` images
themselves -- the spread of a receiver's own-prompt score across the four seeds, which is variation
with no unlearning in it at all. The classification threshold is a stated multiple of that spread,
printed with the result so a reader can re-cut it. Every per-receiver number is written out, so the
classification can be redone without regenerating anything.

Usage, from the repository root::

    PYTHONPATH=. python .../receiver_damage.py \\
        --on-folder  "assets/datasets/generated_people_George W Bush_uce_000" \\
        --off-folder "assets/datasets/generated_people_baseline" \\
        --task people --target George_W_Bush --output damage.json

Exit 0 when the scan completes, 2 on a usage or data error. It reports; it does not pass or fail,
because what counts as an acceptable bleed rate is a decision for the plan, not for this script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from vision_unlearning.datasets.entity_names import (  # noqa: E402
    generation_prompt,
    substitute_concept,
    type_entity_task,
)

_IMAGE_NAME = re.compile(r'^(?P<state>on|off)_(?P<seed>\d{2})_(?P<prompt>.+)\.png$')

#: How many multiples of the no-unlearning spread count as a real move. Stated, not tuned.
THRESHOLD_IN_SPREADS = 3.0


@dataclass
class Receiver:
    prompt: str
    delta_own: float
    delta_substitute: float
    verdict: str = 'preserved'


@dataclass
class Damage:
    n_receivers: int = 0
    seed_spread: float = 0.0
    threshold: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)
    receivers: List[Receiver] = field(default_factory=list)
    target_delta_own: Optional[float] = None
    target_delta_substitute: Optional[float] = None


def _index(folder: str) -> Dict[Tuple[int, str], str]:
    if not os.path.isdir(folder):
        raise SystemExit(f'ERROR: no such folder {folder}')
    found: Dict[Tuple[int, str], str] = {}
    for name in sorted(os.listdir(folder)):
        match = _IMAGE_NAME.match(name)
        if match is not None:
            found[(int(match.group('seed')), match.group('prompt'))] = os.path.join(folder, name)
    return found


def _score_all(paths: Sequence[str], texts: Sequence[str], device: str) -> np.ndarray:
    """Return an (images x texts) matrix of CLIP similarities, in the benchmark's own model."""
    from transformers import CLIPModel, CLIPProcessor
    name = 'openai/clip-vit-base-patch16'
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)

    with torch.no_grad():
        text_inputs = processor(text=list(texts), return_tensors='pt', padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        rows: List[np.ndarray] = []
        batch = 32
        for start in range(0, len(paths), batch):
            images = [Image.open(p).convert('RGB') for p in paths[start:start + batch]]
            image_inputs = processor(images=images, return_tensors='pt').to(device)
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            rows.append((image_features @ text_features.T).cpu().numpy())
            for image in images:
                image.close()
    return np.concatenate(rows, axis=0) * 100.0


def run(on_folder: str, off_folder: str, task: type_entity_task, target: str,
        device: str = 'cuda') -> Damage:
    on_images, off_images = _index(on_folder), _index(off_folder)
    shared = sorted(set(on_images) & set(off_images))
    if not shared:
        raise SystemExit('ERROR: the two folders share no (seed, prompt) key')

    prompts = sorted({key[1] for key in shared})
    substitute_text = f'An image of {substitute_concept(task)}'
    texts = prompts + [substitute_text]
    text_index = {text: i for i, text in enumerate(texts)}
    substitute_column = text_index[substitute_text]

    on_paths = [on_images[key] for key in shared]
    off_paths = [off_images[key] for key in shared]
    on_scores = _score_all(on_paths, texts, device)
    off_scores = _score_all(off_paths, texts, device)

    # Per prompt, average the deltas over seeds; and take the no-unlearning spread from `off`.
    own_deltas: Dict[str, List[float]] = defaultdict(list)
    sub_deltas: Dict[str, List[float]] = defaultdict(list)
    off_own_by_prompt: Dict[str, List[float]] = defaultdict(list)
    for row, (_seed, prompt) in enumerate(shared):
        own = text_index[prompt]
        own_deltas[prompt].append(float(on_scores[row, own] - off_scores[row, own]))
        sub_deltas[prompt].append(
            float(on_scores[row, substitute_column] - off_scores[row, substitute_column]))
        off_own_by_prompt[prompt].append(float(off_scores[row, own]))

    # The scale of "nothing happened": how much a receiver's own score varies across seeds with no
    # unlearning at all. Median over prompts of the per-prompt standard deviation.
    spreads = [float(np.std(values)) for values in off_own_by_prompt.values() if len(values) > 1]
    damage = Damage()
    damage.seed_spread = float(np.median(spreads)) if spreads else 0.0
    damage.threshold = THRESHOLD_IN_SPREADS * damage.seed_spread

    target_prompt = generation_prompt(task, target)
    for prompt in prompts:
        delta_own = float(np.mean(own_deltas[prompt]))
        delta_sub = float(np.mean(sub_deltas[prompt]))
        if prompt == target_prompt:
            damage.target_delta_own = delta_own
            damage.target_delta_substitute = delta_sub
            continue
        receiver = Receiver(prompt=prompt, delta_own=delta_own, delta_substitute=delta_sub)
        lost_itself = delta_own < -damage.threshold
        gained_substitute = delta_sub > damage.threshold
        if lost_itself and gained_substitute:
            receiver.verdict = 'bled'
        elif lost_itself:
            receiver.verdict = 'destroyed'
        elif gained_substitute:
            receiver.verdict = 'drifted toward substitute'
        damage.receivers.append(receiver)

    damage.n_receivers = len(damage.receivers)
    counts: Dict[str, int] = defaultdict(int)
    for receiver in damage.receivers:
        counts[receiver.verdict] += 1
    damage.counts = dict(counts)
    return damage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--on-folder', required=True)
    parser.add_argument('--off-folder', required=True)
    parser.add_argument('--task', required=True, choices=['people', 'breeds', 'scenes'])
    parser.add_argument('--target', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    damage = run(args.on_folder, args.off_folder, args.task, args.target, args.device)

    print(f'receivers                 : {damage.n_receivers}')
    print(f'no-unlearning seed spread : {damage.seed_spread:.4f} CLIP points')
    print(f'threshold ({THRESHOLD_IN_SPREADS}x spread)   : {damage.threshold:.4f}')
    print('')
    for verdict in ('preserved', 'bled', 'destroyed', 'drifted toward substitute'):
        count = damage.counts.get(verdict, 0)
        share = 100.0 * count / damage.n_receivers if damage.n_receivers else 0.0
        print(f'  {verdict:26s} {count:4d}  ({share:.1f}%)')
    print('')
    if damage.target_delta_own is not None:
        print(f'target delta own          : {damage.target_delta_own:+.4f}')
        print(f'target delta substitute   : {damage.target_delta_substitute:+.4f}')

    worst = sorted(damage.receivers, key=lambda r: r.delta_own)[:5]
    print('\nfive receivers that lost the most of themselves:')
    for receiver in worst:
        print(f'  {receiver.delta_own:+8.3f} own  {receiver.delta_substitute:+8.3f} sub  '
              f'{receiver.verdict:26s} {receiver.prompt}')

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump({
                'n_receivers': damage.n_receivers,
                'seed_spread': round(damage.seed_spread, 6),
                'threshold_in_spreads': THRESHOLD_IN_SPREADS,
                'threshold': round(damage.threshold, 6),
                'counts': damage.counts,
                'target_delta_own': damage.target_delta_own,
                'target_delta_substitute': damage.target_delta_substitute,
                'receivers': [
                    {'prompt': r.prompt, 'delta_own': round(r.delta_own, 4),
                     'delta_substitute': round(r.delta_substitute, 4), 'verdict': r.verdict}
                    for r in damage.receivers
                ],
            }, handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.output}')

    print('RECEIVER_DAMAGE_DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
