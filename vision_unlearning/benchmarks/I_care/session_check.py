"""The check that closes one campaign session, and the ledger line that records it.

A session is one unit: ``pipeline_03`` trains (or edits) and generates, ``pipeline_05`` embeds,
``pipeline_06`` scores. This runs after them and decides whether the session delivered, by its files:

* **counts** -- ``seeds x prompts`` images in the session's image folder, as many embedding records,
  one per-pair row per receiver with a finite value of every metric, and per-seed lists of the seed
  count in the per-seed sibling;
* **pairing** -- the session's ``provenance.json`` pairing record against the task baseline's, with
  ``pairing.check`` (class ``image_pairs``): every image started from the baseline's latent, through
  the same scheduler and arguments;
* **digests** -- the sha256 of every file the session wrote, by path below the base folder.

It appends one line to the campaign ledger (``run_ledger``) with the session key, the attempt, the
producer facts (commit, card, the hashes of ``pins.json`` and the freeze), the digests, the counts, the
pairing verdict and the calibration flag, with status ``ok`` -- or ``failed`` and the reason, in which
case it exits 1 so the queue stops. Run ``python -m vision_unlearning.benchmarks.I_care.session_check
--help``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from vision_unlearning.benchmarks.I_care.configuration import GENERATE_DATASET_SEEDS
from vision_unlearning.benchmarks.I_care.metadata import (
    PER_PAIR_METRICS,
    get_embedding_output_path,
    get_interference_per_pair_path,
    per_seed_path_of,
)
from vision_unlearning.benchmarks.I_care.pairing import PairingFailure, check
from vision_unlearning.benchmarks.I_care.provenance import read_provenance
from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger
from vision_unlearning.datasets.entity_names import canonical_entity, generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import (
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_shared_baseline_folder,
    get_unlearned_model_folder,
)


class SessionCheckFailure(RuntimeError):
    """A session did not deliver what a session must; the reasons are listed."""


def session_key(task: str, method: str, epochs: int, index: int) -> str:
    """The key a session is known by in the ledger and the expected-session list."""
    return f'{task}/{method}/{epochs}/{index}'


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digests(paths: Sequence[str], base_folder: str) -> Dict[str, str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files += [os.path.join(root, name) for root, _, names in os.walk(path) for name in names]
        elif os.path.isfile(path):
            files.append(path)
    return {os.path.relpath(f, base_folder).replace(os.sep, '/'): _sha256(f) for f in sorted(files)}


def check_session(
    task: type_entity_task,
    method: str,
    epochs: int,
    index: int,
    entity: str,
    receivers: Sequence[str],
    base_folder: str,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Count, certify and digest one session. Returns the ledger fields; raises on any failure.

    `seeds` defaults to the benchmark's generation seeds.
    """
    seeds = list(GENERATE_DATASET_SEEDS) if seeds is None else list(seeds)
    prompts = [generation_prompt(task, name) for name in receivers]
    target = canonical_entity(task, entity)
    images = get_generated_dataset_folder(task, method, epochs, target, base_folder)  # type: ignore[arg-type]
    model = get_unlearned_model_folder(task, method, epochs, entity, base_folder)  # type: ignore[arg-type]
    baseline = get_shared_baseline_folder(task, base_folder)
    embeddings = get_embedding_output_path(task, target, method, epochs, base_folder=base_folder)
    per_pair = get_interference_per_pair_path(task, index, method, epochs, base_folder=base_folder)  # type: ignore[arg-type]
    per_seed = per_seed_path_of(per_pair)
    problems: List[str] = []
    expected = len(seeds) * len(prompts)

    counts: Dict[str, int] = {}
    expected_names = {get_generated_dataset_file('on', s, p) for s in seeds for p in prompts}
    present = set(os.listdir(images)) if os.path.isdir(images) else set()
    counts['images'] = len(expected_names & present)
    if counts['images'] != expected:
        problems.append(f'{counts["images"]} of {expected} images in {images}')
    try:
        with open(embeddings, encoding='utf-8') as handle:
            counts['embeddings'] = len(json.load(handle)['embeddings'])
    except (OSError, KeyError, ValueError) as exc:
        counts['embeddings'] = 0
        problems.append(f'embeddings unreadable: {exc}')
    if counts['embeddings'] != expected:
        problems.append(f'{counts["embeddings"]} of {expected} embedding records')
    try:
        with open(per_pair, encoding='utf-8') as handle:
            rows = json.load(handle)
        counts['per_pair_rows'] = sum(
            1 for r in receivers if r in rows and all(isinstance(rows[r].get(m), (int, float)) and math.isfinite(rows[r][m])
                                                      for m in PER_PAIR_METRICS))
        with open(per_seed, encoding='utf-8') as handle:
            seed_rows = json.load(handle)
        counts['per_seed_rows'] = sum(
            1 for r in receivers if r in seed_rows['receivers']
            and all(len(seed_rows['receivers'][r][m]) == len(seeds) for m in PER_PAIR_METRICS))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        counts.setdefault('per_pair_rows', 0)
        counts.setdefault('per_seed_rows', 0)
        problems.append(f'per-pair files unreadable: {exc}')
    for name in ('per_pair_rows', 'per_seed_rows'):
        if counts[name] != len(receivers):
            problems.append(f'{counts[name]} of {len(receivers)} {name.replace("_", " ")} complete')

    pairing = 'not checked'
    try:
        check(read_provenance(images, prompts, seeds)['pairing'], read_provenance(baseline, prompts, seeds)['pairing'],
              prompts, seeds, 'image_pairs', file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]
        pairing = 'paired'
    except (PairingFailure, FileNotFoundError, ValueError) as exc:
        pairing = f'failed: {exc}'
        problems.append(f'pairing {pairing}')

    fields = {
        'session': session_key(task, method, epochs, index),
        'counts': counts,
        'pairing': pairing,
        'outputs': _digests([model, images, embeddings, per_pair, per_seed], base_folder),
        'problems': problems,
    }
    if problems:
        raise SessionCheckFailure('; '.join(problems), fields)
    return fields


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m vision_unlearning.benchmarks.I_care.session_check',
                                     description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--method', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--index', type=int, required=True, help="the target's index in the task metadata")
    parser.add_argument('--base-folder', default='assets')
    parser.add_argument('--ledger', required=True, help='the campaign ledger to append to')
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--provenance-of', default=None,
                        help="a provenance.json whose producer facts (commit, card, pins and freeze hashes) the line carries; "
                             "default: the session's image folder's own")
    parser.add_argument('--calibration', action='store_true', help='mark the session as a calibration run')
    args = parser.parse_args(argv)

    task = args.task
    with open(os.path.join(args.base_folder, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        receivers = [entity['name'] for entity in json.load(handle)]
    entity = receivers[args.index]
    key = session_key(task, args.method, args.epochs, args.index)
    ledger = RunLedger(args.ledger)
    try:
        fields = check_session(task, args.method, args.epochs, args.index, entity, receivers, args.base_folder)
        status, message = 'ok', None
    except SessionCheckFailure as exc:
        fields, status, message = exc.args[1], 'failed', exc.args[0]
    images = get_generated_dataset_folder(task, args.method, args.epochs, canonical_entity(task, entity), args.base_folder)  # type: ignore[arg-type]
    source = args.provenance_of or os.path.join(images, 'provenance.json')
    facts: Dict[str, Any] = {}
    if os.path.isfile(source):
        with open(source, encoding='utf-8') as handle:
            record = json.load(handle)
        facts = {k: record.get(k) for k in ('commit', 'tree_clean', 'card_model', 'pins_sha256', 'freeze_sha256')}
    fields.update({'attempt': args.attempt, 'calibration': args.calibration, **facts})
    ledger.record(key, status, exception_type=None if status == 'ok' else 'SessionCheckFailure', message=message, fields=fields)
    ledger.close()
    print(f'SESSION_CHECK {status} {key} {json.dumps(fields["counts"])} pairing={fields["pairing"][:60]}')
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
