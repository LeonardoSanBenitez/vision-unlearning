"""Check one unlearning session against the pipeline-level pass criteria, and draw its contact sheet.

Reads what the benchmark's own stages wrote for one session -- the per-pair interference file
(pipeline_06), the task's per-entity file (pipeline_07) and the baseline embeddings (pipeline_05) --
plus the generated images and the model card, and writes one JSON verdict and one PNG.

Criteria, numbered as in the session plan:

5. The target is the most damaged: its ``clip_diff`` is the most negative of the 100 entries.
6. A distant receiver is barely touched: the receiver whose baseline DINOv2 mean embedding is least
   cosine-similar to the target's has ``|clip_diff| <= 2.0``.
7. Metric recomputation: on the target, the worst, the median and the least-affected receiver (by
   ``clip_diff``; least affected = closest to zero), ``clip_diff``, ``dino_diff`` and
   ``brisque_diff`` are recomputed image by image from the raw PNGs and must agree with the
   per-pair file within ``rtol=1e-4``. Before that, each metric must pass its known-value oracle:
   an image against itself gives exactly the no-change value, and an image against pure noise gives
   a value far from it. If an oracle fails, the recomputation is not attempted.
9. Aggregation identity: in the per-entity file, ``forget_clip_diff`` equals the target's per-pair
   ``clip_diff`` and ``retain_average_clip_diff`` equals the mean over the other 99.
10. The model card holds no absolute path.

The contact sheet is seed 42: the target, the two most and the two least damaged receivers by
``clip_diff``, each before (``off``) and after (``on``), labelled with task, method, entity and seed.

Exit code 0 when every criterion passes, 1 when any fails, 2 on a usage or missing-input error.
The JSON and PNG are written in both of the first two cases.

Example:
    python ablations/entity_strings_audit/s8_session_evidence.py --base-folder assets \\
        --task breeds --method distil --epochs 100 --index 14 --out-dir ablations/entity_strings_audit/assets/s8_evidence
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from vision_unlearning.benchmarks.I_care import get_interference_per_pair_path
from vision_unlearning.benchmarks.I_care.configuration import ALGORITHM_REGISTRY
from vision_unlearning.benchmarks.I_care.embeddings import compute_dino_image_similarity, load_dino_model
from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_entity_path
from vision_unlearning.datasets.testbed import (
    GeneratedDataset,
    get_metadata_filtered,
    get_target_overwrite,
    get_unlearned_model_folder,
)
from vision_unlearning.metrics import MetricImageTextSimilarity, MetricQuality

SEEDS = [42, 43, 44, 45]
RTOL = 1e-4
ATOL = 1e-6
DISTANT_RECEIVER_BOUND = 2.0
# A drive letter only when no letter precedes it, so the "s:/" inside "https://" is not a hit.
ABSOLUTE_PATH = re.compile(r'(/home/|/Users/|/root/|(?<![A-Za-z])[A-Za-z]:[\\/])')


def _prompt(task: str, method: str, name: str) -> str:
    return f"An image of {get_target_overwrite(task, method, name)[0]}"  # type: ignore[arg-type]


def _images(task: str, method: str, epochs: int, target: str, prompt: str, base: str, seed: int) -> Tuple[Image.Image, Image.Image]:
    """The (off, on) pair for one receiver prompt at one seed, located the way pipeline_06 locates it."""
    target_hf = get_target_overwrite(task, method, target)[0]  # type: ignore[arg-type]
    ds = GeneratedDataset(
        task=task, target=target_hf, method=method, num_train_epochs=epochs,  # type: ignore[arg-type]
        artifact_kind=ALGORITHM_REGISTRY[method].artifact_kind,  # type: ignore[index]
        artifact_filename=ALGORITHM_REGISTRY[method].artifact_filename,  # type: ignore[index]
        base_folder=base,
    )
    off = Image.open(GeneratedDataset.get_off_image_path(task, target_hf, method, epochs, seed, prompt, base_folder=base))  # type: ignore[arg-type]
    on = Image.open(ds.file_path('on', seed, prompt))
    return off.convert('RGB'), on.convert('RGB')


def _close(a: float, b: float) -> bool:
    return bool(abs(a - b) <= ATOL + RTOL * abs(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-folder', required=True)
    parser.add_argument('--task', required=True, choices=['people', 'breeds', 'scenes'])
    parser.add_argument('--method', required=True, choices=list(ALGORITHM_REGISTRY))
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--index', required=True, type=int, help='metadata index of the unlearned entity')
    parser.add_argument('--tag', default='', help='suffix for the output names, e.g. _oldguide')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    task, method, epochs, base = args.task, args.method, args.epochs, args.base_folder
    metadata = get_metadata_filtered(task, base_folder=base)  # type: ignore[arg-type]
    names = [m['name'] for m in metadata]
    target = names[args.index]
    pair_path = get_interference_per_pair_path(task, args.index, method, epochs, base_folder=base)  # type: ignore[arg-type]
    if not os.path.exists(pair_path):
        print(f'ERROR: no per-pair file at {pair_path}')
        return 2
    with open(pair_path, encoding='utf-8') as f:
        per_pair: Dict[str, Dict[str, float]] = json.load(f)
    if len(per_pair) != 100 or target not in per_pair:
        print(f'ERROR: per-pair file has {len(per_pair)} entries; target {target!r} present: {target in per_pair}')
        return 2

    clip = {name: float(v['clip_diff']) for name, v in per_pair.items()}
    receivers = [n for n in names if n != target]
    by_clip = sorted(receivers, key=lambda n: clip[n])
    result: Dict[str, Any] = {
        'session': {'base_folder': base, 'task': task, 'method': method, 'epochs': epochs,
                    'index': args.index, 'target': target, 'per_pair_file': pair_path},
        'criteria': {},
    }
    crit = result['criteria']

    # 5 -- the target is the most damaged
    most_damaged = min(clip, key=lambda n: clip[n])
    crit['5_target_most_damaged'] = {
        'pass': most_damaged == target, 'target_clip_diff': clip[target],
        'most_damaged': most_damaged, 'most_damaged_clip_diff': clip[most_damaged],
    }

    # 6 -- the most distant receiver, by baseline DINOv2 embeddings, barely moves
    emb_path = os.path.join(base, 'datasets', f'embeddings_{task}_original.json')
    with open(emb_path, encoding='utf-8') as f:
        records = json.load(f)['embeddings']
    buckets: Dict[str, List[List[float]]] = {}
    for r in records:
        buckets.setdefault(r['prompt'], []).append(r['embedding'])
    means = {p: np.mean(np.asarray(v), axis=0) for p, v in buckets.items()}
    means = {p: m / np.linalg.norm(m) for p, m in means.items()}
    t_vec = means[_prompt(task, method, target)]
    similarity = {n: float(np.dot(t_vec, means[_prompt(task, method, n)])) for n in receivers}
    distant = min(similarity, key=lambda n: similarity[n])
    crit['6_distant_receiver_untouched'] = {
        'pass': abs(clip[distant]) <= DISTANT_RECEIVER_BOUND, 'receiver': distant,
        'cosine_similarity_to_target': similarity[distant], 'clip_diff': clip[distant],
        'bound': DISTANT_RECEIVER_BOUND, 'similarity_source': emb_path,
    }

    # 7 -- oracles first, then recomputation on the stratified set
    metric_clip = MetricImageTextSimilarity(metrics=['clip'])
    metric_quality = MetricQuality()
    dino_model, dino_transform, dino_device = load_dino_model()
    probe_off, _ = _images(task, method, epochs, target, _prompt(task, method, target), base, SEEDS[0])
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, size=(probe_off.height, probe_off.width, 3), dtype=np.uint8))
    probe_text = _prompt(task, method, target)
    oracle = {
        'clip_identical': metric_clip.score(probe_off, probe_text)['clip'] - metric_clip.score(probe_off, probe_text)['clip'],
        'clip_noise': metric_clip.score(noise, probe_text)['clip'] - metric_clip.score(probe_off, probe_text)['clip'],
        'dino_identical': compute_dino_image_similarity(probe_off, probe_off, dino_model, dino_transform, dino_device),
        'dino_noise': compute_dino_image_similarity(probe_off, noise, dino_model, dino_transform, dino_device),
        'brisque_identical': metric_quality.score(probe_off)['brisque'] - metric_quality.score(probe_off)['brisque'],
        'brisque_noise': metric_quality.score(noise)['brisque'] - metric_quality.score(probe_off)['brisque'],
    }
    oracle_pass = {
        'clip': oracle['clip_identical'] == 0.0 and oracle['clip_noise'] < -5.0,
        'dino': abs(oracle['dino_identical'] - 1.0) < 1e-4 and oracle['dino_noise'] < 0.5,
        'brisque': oracle['brisque_identical'] == 0.0 and abs(oracle['brisque_noise']) > 10.0,
    }
    stratified = {
        'target': target,
        'worst_receiver': by_clip[0],
        'median_receiver': by_clip[len(by_clip) // 2],
        'least_affected_receiver': min(receivers, key=lambda n: abs(clip[n])),
    }
    rows: Dict[str, Any] = {}
    all_agree = all(oracle_pass.values())
    if all_agree:
        for role, name in stratified.items():
            prompt = _prompt(task, method, name)
            c, d, b = [], [], []
            for seed in SEEDS:
                off, on = _images(task, method, epochs, target, prompt, base, seed)
                c.append(metric_clip.score(on, prompt)['clip'] - metric_clip.score(off, prompt)['clip'])
                d.append(compute_dino_image_similarity(off, on, dino_model, dino_transform, dino_device))
                b.append(metric_quality.score(on)['brisque'] - metric_quality.score(off)['brisque'])
            recomputed = {'clip_diff': float(np.mean(c)), 'dino_diff': float(np.mean(d)), 'brisque_diff': float(np.mean(b))}
            agree = {k: _close(recomputed[k], float(per_pair[name][k])) for k in recomputed}
            all_agree = all_agree and all(agree.values())
            rows[role] = {'entity': name, 'recomputed': recomputed,
                          'pipeline': {k: float(per_pair[name][k]) for k in recomputed}, 'agree': agree}
    crit['7_metric_recomputation'] = {'pass': all_agree, 'rtol': RTOL, 'atol': ATOL,
                                      'oracle': oracle, 'oracle_pass': oracle_pass, 'stratified': rows}

    # 9 -- aggregation identity against the per-entity file
    entity_path = get_interference_per_entity_path(task, base_folder=base)  # type: ignore[arg-type]
    agg: Dict[str, Any] = {'per_entity_file': entity_path}
    if os.path.exists(entity_path):
        with open(entity_path, encoding='utf-8') as f:
            per_entity = json.load(f)
        row = _entity_row(per_entity, target)
        forget = _lookup(row, f'metric_{method}_{epochs}_forget_clip_diff')
        retain = _lookup(row, f'metric_{method}_{epochs}_retain_average_clip_diff')
        expected_retain = statistics.fmean(clip[n] for n in receivers)
        agg.update({'forget_clip_diff': forget, 'expected_forget': clip[target],
                    'retain_average_clip_diff': retain, 'expected_retain_average': expected_retain})
        agg['pass'] = (forget is not None and retain is not None
                       and _close(forget, clip[target]) and _close(retain, expected_retain))
    else:
        agg['pass'] = False
        agg['error'] = 'no per-entity file'
    crit['9_aggregation_identity'] = agg

    # 10 -- the model card holds no absolute path
    model_dir = get_unlearned_model_folder(task, method, epochs, target, base_folder=base)  # type: ignore[arg-type]
    card = os.path.join(model_dir, 'README.md')
    if os.path.exists(card):
        with open(card, encoding='utf-8') as f:
            hits = sorted({m.group(0) for m in ABSOLUTE_PATH.finditer(f.read())})
        crit['10_model_card_clean'] = {'pass': not hits, 'card': card, 'absolute_path_markers': hits}
    else:
        crit['10_model_card_clean'] = {'pass': None, 'card': card, 'note': 'this method writes no model card'}

    # Contact sheet -- seed 42
    shown = [('target', target), ('most damaged', by_clip[0]), ('2nd most damaged', by_clip[1]),
             ('least damaged', by_clip[-1]), ('2nd least damaged', by_clip[-2])]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f'{task}_{method}_{epochs:03d}{args.tag}'
    sheet_path = os.path.join(args.out_dir, f'contact_{stem}.png')
    _contact_sheet(shown, clip, task, method, epochs, target, base, sheet_path)
    result['contact_sheet'] = {'path': sheet_path, 'seed': SEEDS[0], 'rows': [{'role': r, 'entity': n, 'clip_diff': clip[n]} for r, n in shown]}

    verdicts = [v['pass'] for v in crit.values() if v['pass'] is not None]
    result['all_pass'] = all(verdicts)
    out_json = os.path.join(args.out_dir, f'criteria_{stem}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    for key, verdict in crit.items():
        print(f'{key}: {"PASS" if verdict["pass"] else ("n/a" if verdict["pass"] is None else "FAIL")}')
    print(f'written {out_json} and {sheet_path}')
    return 0 if result['all_pass'] else 1


def _entity_row(per_entity: Any, target: str) -> Dict[str, Any]:
    """The per-entity record of ``target``: pipeline_07 writes a list of records keyed by ``name``."""
    for r in per_entity:
        if r.get('name') == target:
            return dict(r)
    return {}


def _lookup(row: Dict[str, Any], stem: str) -> Optional[float]:
    """The value whose key is ``stem`` followed by pipeline_07's direction marker, e.g.
    ``metric_uce_0_forget_clip_diff (↓)``. None when there is not exactly one such key."""
    keys = [k for k in row if k == stem or k.startswith(stem + ' (')]
    return float(row[keys[0]]) if len(keys) == 1 and row[keys[0]] is not None else None


def _contact_sheet(shown: List[Tuple[str, str]], clip: Dict[str, float], task: str, method: str, epochs: int,
                   target: str, base: str, path: str) -> None:
    cell, label_h = 256, 34
    sheet = Image.new('RGB', (2 * cell, len(shown) * (cell + label_h)), 'white')
    draw = ImageDraw.Draw(sheet)
    for r, (role, name) in enumerate(shown):
        off, on = _images(task, method, epochs, target, _prompt(task, method, name), base, SEEDS[0])
        y = r * (cell + label_h)
        draw.text((4, y + 2), f'{task} / {method} {epochs} / unlearned: {target} / seed {SEEDS[0]}', fill='black')
        draw.text((4, y + 16), f'{role}: {name}   clip_diff {clip[name]:+.2f}   [off | on]', fill='black')
        sheet.paste(off.resize((cell, cell)), (0, y + label_h))
        sheet.paste(on.resize((cell, cell)), (cell, y + label_h))
    sheet.save(path)


if __name__ == '__main__':
    sys.exit(main())
