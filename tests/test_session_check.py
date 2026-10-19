"""The check that closes a campaign session: counts, input pairing, digests, and its ledger line."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import session_check
from vision_unlearning.benchmarks.I_care.metadata import (
    PER_PAIR_METRICS,
    get_embedding_output_path,
    get_interference_per_pair_path,
    per_seed_path_of,
)
from vision_unlearning.benchmarks.I_care.provenance import prompt_list_sha256
from vision_unlearning.benchmarks.I_care.run_ledger import read_records
from vision_unlearning.datasets.entity_names import canonical_entity, generation_prompt
from vision_unlearning.datasets.testbed import (
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_shared_baseline_folder,
    get_unlearned_model_folder,
)
from vision_unlearning.utils.noise import pairing_record

TASK = 'people'
METHOD = 'uce'
EPOCHS = 1
RECEIVERS = ['George_W_Bush', 'Serena_Williams', 'Gray_Davis']
SEEDS = [42, 43]
INDEX = 0
PNDM = {'skip_prk_steps': True, 'num_train_timesteps': 1000}
KWARGS = {'num_inference_steps': 50, 'guidance_scale': 7.5, 'batch_size': 8, 'dtype': 'float16'}


def _write_provenance(folder: str, lora_state: str, latent_of_seed: Dict[int, str]) -> None:
    prompts = [generation_prompt(TASK, r) for r in RECEIVERS]
    keys = [{'prompt_index': i, 'seed': s, 'file_name': get_generated_dataset_file(lora_state, s, p),  # type: ignore[arg-type]
             'initial_latent_sha256': latent_of_seed[s]} for s in SEEDS for i, p in enumerate(prompts)]
    record = {'commit': 'abc', 'tree_clean': True, 'card_model': 'A40', 'pins_sha256': 'p', 'freeze_sha256': 'f',
              'seeds': SEEDS, 'prompts_ordered': prompts, 'prompt_list_sha256': prompt_list_sha256(prompts),
              'pairing': pairing_record(keys, 'PNDMScheduler', PNDM, KWARGS, latent_hash_schema_version=1)}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'provenance.json'), 'w', encoding='utf-8') as handle:
        json.dump(record, handle)


def _session(base: str) -> Dict[str, str]:
    """A complete session on disk: model, images, embeddings, per-pair files, and the baseline."""
    target = canonical_entity(TASK, RECEIVERS[INDEX])
    with open(os.path.join(base, f'metadata_{TASK}_2_enriched_filtered.json'), 'w', encoding='utf-8') as handle:
        json.dump([{'name': r} for r in RECEIVERS], handle)
    latents = {s: f'latent-{s}' for s in SEEDS}
    _write_provenance(get_shared_baseline_folder(TASK, base), 'off', latents)
    images = get_generated_dataset_folder(TASK, METHOD, EPOCHS, target, base)  # type: ignore[arg-type]
    _write_provenance(images, 'on', latents)
    for s in SEEDS:
        for r in RECEIVERS:
            with open(os.path.join(images, get_generated_dataset_file('on', s, generation_prompt(TASK, r))), 'wb') as handle:
                handle.write(f'{s}{r}'.encode())
    model = get_unlearned_model_folder(TASK, METHOD, EPOCHS, RECEIVERS[INDEX], base)  # type: ignore[arg-type]
    os.makedirs(model)
    with open(os.path.join(model, 'uce_sd_weights.safetensors'), 'wb') as handle:
        handle.write(b'weights')
    embeddings = get_embedding_output_path(TASK, target, METHOD, EPOCHS, base_folder=base)
    os.makedirs(os.path.dirname(embeddings), exist_ok=True)
    with open(embeddings, 'w', encoding='utf-8') as handle:
        json.dump({'embeddings': [{'n': n} for n in range(len(SEEDS) * len(RECEIVERS))]}, handle)
    per_pair = get_interference_per_pair_path(TASK, INDEX, METHOD, EPOCHS, base_folder=base)  # type: ignore[arg-type]
    os.makedirs(os.path.dirname(per_pair), exist_ok=True)
    with open(per_pair, 'w', encoding='utf-8') as handle:
        json.dump({r: {m: 0.5 for m in PER_PAIR_METRICS} for r in RECEIVERS}, handle)
    with open(per_seed_path_of(per_pair), 'w', encoding='utf-8') as handle:
        json.dump({'seeds': SEEDS, 'receivers': {r: {m: [0.5] * len(SEEDS) for m in PER_PAIR_METRICS} for r in RECEIVERS}}, handle)
    return {'images': images, 'embeddings': embeddings, 'per_pair': per_pair, 'baseline': get_shared_baseline_folder(TASK, base)}


def _run(base: str, ledger: str, monkeypatch: Any) -> int:
    monkeypatch.setattr(session_check, 'GENERATE_DATASET_SEEDS', SEEDS)
    return session_check.main(['--task', TASK, '--method', METHOD, '--epochs', str(EPOCHS), '--index', str(INDEX),
                               '--base-folder', base, '--ledger', ledger, '--attempt', '1'])


def _only_record(ledger: str) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = read_records(ledger)
    assert len(records) == 1
    return records[0]


def test_a_complete_paired_session_is_ok_and_writes_one_ledger_line(tmp_path: Any, monkeypatch: Any) -> None:
    base, ledger = str(tmp_path / 'assets'), str(tmp_path / 'ledger.jsonl')
    os.makedirs(base)
    _session(base)
    assert _run(base, ledger, monkeypatch) == 0
    record = _only_record(ledger)
    assert record['status'] == 'ok'
    fields = record['fields']
    assert fields['session'] == f'{TASK}/{METHOD}/{EPOCHS}/{INDEX}'
    assert fields['counts'] == {'images': 6, 'embeddings': 6, 'per_pair_rows': 3, 'per_seed_rows': 3}
    assert fields['pairing'] == 'paired'
    assert fields['commit'] == 'abc' and fields['pins_sha256'] == 'p' and fields['attempt'] == 1
    # every file the session wrote is digested: 6 images + provenance, the model, embeddings, two per-pair files
    assert len(fields['outputs']) == 6 + 1 + 1 + 1 + 2
    assert all(len(v) == 64 for v in fields['outputs'].values())


def test_a_missing_image_fails_the_session(tmp_path: Any, monkeypatch: Any) -> None:
    base, ledger = str(tmp_path / 'assets'), str(tmp_path / 'ledger.jsonl')
    os.makedirs(base)
    paths = _session(base)
    os.remove(os.path.join(paths['images'], get_generated_dataset_file('on', 43, generation_prompt(TASK, RECEIVERS[2]))))
    assert _run(base, ledger, monkeypatch) == 1
    record = _only_record(ledger)
    assert record['status'] == 'failed' and '5 of 6 images' in record['message']


def test_a_non_finite_score_fails_the_session(tmp_path: Any, monkeypatch: Any) -> None:
    base, ledger = str(tmp_path / 'assets'), str(tmp_path / 'ledger.jsonl')
    os.makedirs(base)
    paths = _session(base)
    with open(paths['per_pair'], encoding='utf-8') as handle:
        rows = json.load(handle)
    rows[RECEIVERS[1]]['dino_diff'] = float('nan')
    with open(paths['per_pair'], 'w', encoding='utf-8') as handle:
        json.dump(rows, handle)
    assert _run(base, ledger, monkeypatch) == 1
    assert '2 of 3 per pair rows complete' in _only_record(ledger)['message']


def test_a_session_on_other_noise_than_the_baseline_fails_pairing(tmp_path: Any, monkeypatch: Any) -> None:
    base, ledger = str(tmp_path / 'assets'), str(tmp_path / 'ledger.jsonl')
    os.makedirs(base)
    paths = _session(base)
    _write_provenance(paths['images'], 'on', {42: 'latent-42', 43: 'another-latent'})
    assert _run(base, ledger, monkeypatch) == 1
    record = _only_record(ledger)
    assert record['fields']['pairing'].startswith('failed') and 'another latent' in record['message']


def test_check_session_raises_with_the_fields_attached(tmp_path: Any) -> None:
    base = str(tmp_path / 'assets')
    os.makedirs(base)
    paths = _session(base)
    os.remove(paths['embeddings'])
    with pytest.raises(session_check.SessionCheckFailure) as caught:
        session_check.check_session(TASK, METHOD, EPOCHS, INDEX, RECEIVERS[INDEX], RECEIVERS, base, seeds=SEEDS)  # type: ignore[arg-type]
    assert caught.value.args[1]['counts']['embeddings'] == 0
