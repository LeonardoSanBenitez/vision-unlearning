"""The check that closes one campaign unit -- a session, or a task's baseline -- and its ledger line.

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

``--baseline`` closes a task's baseline instead (:func:`check_baseline`), the unit every session is later
paired against: its ``seeds x prompts`` images and their pairing record (every key once, one latent per
seed, every file named for its prompt and seed, an allowed scheduler), as many baseline embedding records,
the ``act`` fingerprints and the ``unet_latent`` capture each certified against the baseline's inputs
(classes ``act`` and ``unet_latent``) and the capture's own decode check covering every latent, the five
similarity matrices complete and finite, and the competence file with a finite, positive ``f_min``. Its
ledger key is ``{task}/baseline``. The ``unet_latent`` cache is a local debug cache, never published, so it
is checked but not digested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from vision_unlearning.benchmarks.I_care.baseline_competence import competence_path
from vision_unlearning.benchmarks.I_care.configuration import GENERATE_DATASET_SEEDS
from vision_unlearning.benchmarks.I_care.metadata import (
    PER_PAIR_METRICS,
    BaselineEmbeddings,
    receivers_in_order,
    get_embedding_output_path,
    get_interference_per_pair_path,
    per_seed_path_of,
)
from vision_unlearning.benchmarks.I_care.pairing import PairingFailure, check, restrict
from vision_unlearning.benchmarks.I_care.provenance import read_provenance
from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger
from vision_unlearning.benchmarks.I_care.similarity import Similarity, UnetLatentSimilarity
from vision_unlearning.utils.mechanistic_interpretability import get_act_fingerprints_path
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


def readout_key(task: str, method: str, dial: str, index: int) -> str:
    """The key of an equalization read-out: the method's dial value (epochs, or UCE's erase scale) and the target."""
    return f'calibration/{task}/{method}/{dial}/{index}'


def baseline_key(task: str) -> str:
    """The key a task's baseline is known by in the ledger and the expected-session list."""
    return f'{task}/baseline'


#: The similarity matrices a task's baseline unit holds.
BASELINE_SIMILARITIES = ('clip', 'jacc', 'dino', 'act', 'unet_latent')


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
    entities: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Count, certify and digest one session. Returns the ledger fields; raises on any failure.

    `seeds` defaults to the benchmark's generation seeds. `entities` is the task's full canonical list,
    when `receivers` is only part of it: then the session is a **read-out** (equalization's), which has no
    embeddings, and its images are certified against the baseline restricted to their prompts.
    """
    seeds = list(GENERATE_DATASET_SEEDS) if seeds is None else list(seeds)
    entities = list(receivers) if entities is None else list(entities)
    readout = list(receivers) != entities
    prompts = [generation_prompt(task, name) for name in receivers]
    all_prompts = [generation_prompt(task, name) for name in entities]
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
    if not readout:
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
        reference = read_provenance(baseline, all_prompts, seeds)['pairing']
        if readout:
            reference = restrict(reference, all_prompts, prompts)
        check(read_provenance(images, prompts, seeds)['pairing'], reference,
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


def _load(path: str) -> Any:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def check_baseline(
    task: type_entity_task,
    entities: Sequence[str],
    base_folder: str,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Count, certify and digest a task's baseline unit. Returns the ledger fields; raises on any failure.

    `entities` are the task's metadata names in canonical order; `seeds` defaults to the benchmark's.
    """
    seeds = list(GENERATE_DATASET_SEEDS) if seeds is None else list(seeds)
    prompts = [generation_prompt(task, name) for name in entities]
    expected = len(seeds) * len(prompts)
    folder = get_shared_baseline_folder(task, base_folder)
    embeddings = BaselineEmbeddings(task=task, base_folder=base_folder)._get_data_path_local()  # type: ignore[arg-type]
    act = get_act_fingerprints_path(task, 'sd1.4', base_folder)
    latents = UnetLatentSimilarity(task=task, base_folder=base_folder).cache_path()  # type: ignore[arg-type]
    matrices = {s: Similarity(task=task, similarity_metric=s, base_folder=base_folder)._get_data_path_local()  # type: ignore[arg-type]
                for s in BASELINE_SIMILARITIES}
    competence = competence_path(task, base_folder)
    problems: List[str] = []
    counts: Dict[str, int] = {}
    pairing: Dict[str, str] = {}

    present = set(os.listdir(folder)) if os.path.isdir(folder) else set()
    counts['images'] = len({get_generated_dataset_file('off', s, p) for s in seeds for p in prompts} & present)
    if counts['images'] != expected:
        problems.append(f'{counts["images"]} of {expected} images in {folder}')

    base: Optional[Dict[str, Any]] = None
    try:
        base = read_provenance(folder, prompts, seeds)['pairing']
        check(base, base, prompts, seeds, 'image_pairs', file_name_of=get_generated_dataset_file,  # type: ignore[arg-type]
              candidate_lora_state='off')
        pairing['images'] = 'paired'
    except (PairingFailure, FileNotFoundError, ValueError, KeyError) as exc:
        pairing['images'] = f'failed: {exc}'
        problems.append(f'baseline pairing record {pairing["images"]}')

    try:
        records = _load(embeddings)['embeddings']
        covered = {(r['prompt'], int(r['seed'])) for r in records} & {(p, s) for p in prompts for s in seeds}
        counts['embeddings'] = len(covered)
        if len(records) != expected or counts['embeddings'] != expected:
            problems.append(f'{len(records)} embedding records, {counts["embeddings"]} of {expected} images covered')
    except (OSError, KeyError, ValueError, TypeError) as exc:
        counts['embeddings'] = 0
        problems.append(f'baseline embeddings unreadable: {exc}')

    for name, path in (('act', act), ('unet_latent', latents)):
        try:
            record = _load(path)
            if name == 'act':
                counts['act_entities'] = len({r['entity'] for r in record['fingerprints']} & set(entities))
                if counts['act_entities'] != len(entities):
                    problems.append(f'{counts["act_entities"]} of {len(entities)} act fingerprints')
            else:
                counts['unet_latent_verified'] = int(record['metadata']['baseline_verification']['n_checked'])
                if counts['unet_latent_verified'] != expected:
                    problems.append(f'{counts["unet_latent_verified"]} of {expected} latents checked against their baseline image')
            if base is None:
                raise ValueError('no baseline pairing record to certify against')
            check(record['metadata']['pairing'], base, prompts, seeds, name)  # type: ignore[arg-type]
            pairing[name] = 'paired'
        except (PairingFailure, OSError, KeyError, ValueError, TypeError) as exc:
            pairing[name] = f'failed: {exc}'
            problems.append(f'{name} {pairing[name]}')

    counts['similarity_matrices'] = 0
    for metric, path in matrices.items():
        try:
            rows = _load(path)
            complete = [row['emitter'] for row in rows] == list(entities) and all(
                isinstance(row.get(r), (int, float)) and math.isfinite(row[r]) for row in rows for r in entities)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            problems.append(f'similarity {metric} unreadable: {exc}')
            continue
        if complete:
            counts['similarity_matrices'] += 1
        else:
            problems.append(f'similarity {metric}: not a complete, finite {len(entities)}x{len(entities)} matrix in entity order')

    summary: Dict[str, Any] = {}
    try:
        record = _load(competence)
        counts['competence_entities'] = len(set(record['per_image']) & set(entities))
        f_min = float(record['f_min'])
        summary = {'f_min': f_min, 'recognised': record['screen']['recognised'], 'passes': record['screen']['passes']}
        if counts['competence_entities'] != len(entities) or record['seeds'] != seeds:
            problems.append(f'competence covers {counts["competence_entities"]} of {len(entities)} entities at seeds {record["seeds"]}')
        if not (math.isfinite(f_min) and f_min > 0):
            problems.append(f'competence f_min is {f_min}')
    except (OSError, KeyError, ValueError, TypeError) as exc:
        counts['competence_entities'] = 0
        problems.append(f'competence unreadable: {exc}')

    fields = {
        'session': baseline_key(task),
        'counts': counts,
        'pairing': 'paired' if sorted(pairing) == ['act', 'images', 'unet_latent'] and set(pairing.values()) == {'paired'}
                   else json.dumps(pairing),
        'competence': summary,
        'outputs': _digests([folder, embeddings, act, *matrices.values(), competence], base_folder),
        'problems': problems,
    }
    if problems:
        raise SessionCheckFailure('; '.join(problems), fields)
    return fields


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m vision_unlearning.benchmarks.I_care.session_check',
                                     description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--baseline', action='store_true',
                        help="check the task's baseline unit instead of a session; --method, --epochs and --index are then not taken")
    parser.add_argument('--method', help='required unless --baseline')
    parser.add_argument('--epochs', type=int, help='required unless --baseline')
    parser.add_argument('--index', type=int, help="the target's index in the task metadata; required unless --baseline")
    parser.add_argument('--base-folder', default='assets')
    parser.add_argument('--ledger', required=True, help='the campaign ledger to append to')
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--provenance-of', default=None,
                        help="a provenance.json whose producer facts (commit, card, pins and freeze hashes) the line carries; "
                             "default: the session's image folder's own")
    parser.add_argument('--calibration', action='store_true', help='mark the session as a calibration run')
    parser.add_argument('--receivers', nargs='+', default=None,
                        help='an equalization read-out: the receivers its images and scores cover (metadata names); '
                             'needs --readout-dial and --calibration, and has no embeddings')
    parser.add_argument('--readout-dial', default=None,
                        help="the read-out's dial value as the ladder spells it (epochs, or UCE's erase scale); its ledger key is "
                             "calibration/<task>/<method>/<dial>/<index>")
    args = parser.parse_args(argv)
    if not args.baseline and None in (args.method, args.epochs, args.index):
        parser.error('--method, --epochs and --index are required unless --baseline')
    if (args.receivers is None) != (args.readout_dial is None) or (args.receivers is not None and not args.calibration):
        parser.error('--receivers and --readout-dial go together, and a read-out is a calibration run (--calibration)')

    task = args.task
    with open(os.path.join(args.base_folder, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        receivers = [entity['name'] for entity in json.load(handle)]
    scored = receivers_in_order(receivers, args.receivers)
    ledger = RunLedger(args.ledger)
    if args.baseline:
        key = baseline_key(task)
        images = get_shared_baseline_folder(task, args.base_folder)
    else:
        entity = receivers[args.index]
        key = (session_key(task, args.method, args.epochs, args.index) if args.receivers is None
               else readout_key(task, args.method, args.readout_dial, args.index))
        images = get_generated_dataset_folder(task, args.method, args.epochs, canonical_entity(task, entity), args.base_folder)  # type: ignore[arg-type]
    try:
        if args.baseline:
            fields = check_baseline(task, receivers, args.base_folder)
        else:
            fields = check_session(task, args.method, args.epochs, args.index, entity, scored, args.base_folder,
                                   entities=receivers)
        status, message = 'ok', None
    except SessionCheckFailure as exc:
        fields, status, message = exc.args[1], 'failed', exc.args[0]
    source = args.provenance_of or os.path.join(images, 'provenance.json')
    facts: Dict[str, Any] = {}
    if os.path.isfile(source):
        with open(source, encoding='utf-8') as handle:
            record = json.load(handle)
        facts = {k: record.get(k) for k in ('commit', 'tree_clean', 'card_model', 'pins_sha256', 'freeze_sha256')}
    fields.update({'session': key, 'attempt': args.attempt, 'calibration': args.calibration, **facts})
    ledger.record(key, status, exception_type=None if status == 'ok' else 'SessionCheckFailure', message=message, fields=fields)
    ledger.close()
    print(f'SESSION_CHECK {status} {key} {json.dumps(fields["counts"])} pairing={fields["pairing"][:60]}')
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
