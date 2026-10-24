"""An equalization read-out -- a few receivers' images and scores -- is checked and recorded as its own unit.

Fails on the code before this change: ``session_check`` could only check a session of every receiver.
A read-out holds a few prompts, has no embeddings, and its pairing record is indexed by position in its
own short prompt list, so the check counted it short, found no embeddings, and compared its keys with
the wrong baseline keys.
"""
import json
import os
from typing import Any, Dict, List, Literal

import pytest

from vision_unlearning.benchmarks.I_care import session_check
from vision_unlearning.benchmarks.I_care.metadata import (
    PER_PAIR_METRICS,
    get_interference_per_pair_path,
    per_seed_path_of,
    receivers_in_order,
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

TASK: Literal['people'] = 'people'
METHOD = 'distil'
EPOCHS = 50
ENTITIES = ['George_W_Bush', 'Serena_Williams', 'Gray_Davis', 'Ray_Romano']
READOUT = ['Gray_Davis', 'George_W_Bush']          # given out of order on purpose
SEEDS = [42, 43]
INDEX = 0
PNDM = {'skip_prk_steps': True, 'num_train_timesteps': 1000}
KWARGS = {'num_inference_steps': 50, 'guidance_scale': 7.5, 'batch_size': 8, 'dtype': 'float16'}


def _provenance(folder: str, names: List[str], lora_state: str) -> None:
    prompts = [generation_prompt(TASK, n) for n in names]
    keys = [{'prompt_index': i, 'seed': s, 'file_name': get_generated_dataset_file(lora_state, s, p),  # type: ignore[arg-type]
             'initial_latent_sha256': f'latent-{s}'} for s in SEEDS for i, p in enumerate(prompts)]
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'provenance.json'), 'w', encoding='utf-8') as handle:
        json.dump({'commit': 'c', 'tree_clean': True, 'card_model': 'A40', 'pins_sha256': 'p', 'freeze_sha256': 'f',
                   'seeds': SEEDS, 'prompts_ordered': prompts, 'prompt_list_sha256': prompt_list_sha256(prompts),
                   'pairing': pairing_record(keys, 'PNDMScheduler', PNDM, KWARGS, latent_hash_schema_version=1)}, handle)


def _readout(base: str) -> List[str]:
    scored = receivers_in_order(ENTITIES, READOUT)
    with open(os.path.join(base, f'metadata_{TASK}_2_enriched_filtered.json'), 'w', encoding='utf-8') as handle:
        json.dump([{'name': n} for n in ENTITIES], handle)
    _provenance(get_shared_baseline_folder(TASK, base), ENTITIES, 'off')
    images = get_generated_dataset_folder(TASK, METHOD, EPOCHS, canonical_entity(TASK, ENTITIES[INDEX]), base)  # type: ignore[arg-type]
    _provenance(images, scored, 'on')
    for s in SEEDS:
        for n in scored:
            open(os.path.join(images, get_generated_dataset_file('on', s, generation_prompt(TASK, n))), 'wb').close()
    model = get_unlearned_model_folder(TASK, METHOD, EPOCHS, ENTITIES[INDEX], base)  # type: ignore[arg-type]
    os.makedirs(model)
    open(os.path.join(model, 'pytorch_lora_weights.safetensors'), 'wb').close()
    per_pair = get_interference_per_pair_path(TASK, INDEX, METHOD, EPOCHS, base_folder=base)  # type: ignore[arg-type]
    os.makedirs(os.path.dirname(per_pair), exist_ok=True)
    with open(per_pair, 'w', encoding='utf-8') as handle:
        json.dump({n: {m: 0.5 for m in PER_PAIR_METRICS} for n in scored}, handle)
    with open(per_seed_path_of(per_pair), 'w', encoding='utf-8') as handle:
        json.dump({'receivers': {n: {m: [0.5] * len(SEEDS) for m in PER_PAIR_METRICS} for n in scored}}, handle)
    return scored


def test_the_receivers_keep_the_task_order_and_unknown_names_are_refused() -> None:
    assert receivers_in_order(ENTITIES, READOUT) == ['George_W_Bush', 'Gray_Davis']
    assert receivers_in_order(ENTITIES, None) == ENTITIES
    with pytest.raises(ValueError):
        receivers_in_order(ENTITIES, ['Nobody'])
    with pytest.raises(ValueError):
        receivers_in_order(ENTITIES, ['Gray_Davis', 'Gray_Davis'])


def test_a_complete_readout_passes_without_embeddings(tmp_path: Any) -> None:
    base = str(tmp_path)
    scored = _readout(base)
    fields: Dict[str, Any] = session_check.check_session(TASK, METHOD, EPOCHS, INDEX, ENTITIES[INDEX], scored, base,
                                                         SEEDS, entities=ENTITIES)
    assert fields['counts'] == {'images': 4, 'per_pair_rows': 2, 'per_seed_rows': 2}
    assert fields['pairing'] == 'paired'


def test_a_readout_on_other_noise_fails(tmp_path: Any) -> None:
    base = str(tmp_path)
    scored = _readout(base)
    path = os.path.join(get_shared_baseline_folder(TASK, base), 'provenance.json')
    with open(path, encoding='utf-8') as handle:
        record = json.load(handle)
    for key in record['pairing']['keys']:
        if key['seed'] == 43:
            key['initial_latent_sha256'] = 'another'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle)
    with pytest.raises(session_check.SessionCheckFailure, match='another latent'):
        session_check.check_session(TASK, METHOD, EPOCHS, INDEX, ENTITIES[INDEX], scored, base, SEEDS, entities=ENTITIES)


def test_the_command_records_a_readout_under_its_dial(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    base = str(tmp_path)
    _readout(base)
    monkeypatch.setattr(session_check, 'GENERATE_DATASET_SEEDS', SEEDS)
    ledger = os.path.join(base, 'ledger.jsonl')
    code = session_check.main(['--task', TASK, '--method', METHOD, '--epochs', str(EPOCHS), '--index', str(INDEX),
                               '--base-folder', base, '--ledger', ledger, '--attempt', '1', '--calibration',
                               '--receivers', *READOUT, '--readout-dial', '50'])
    assert code == 0
    [line] = read_records(ledger)
    assert (line['context'], line['status'], line['fields']['session'], line['fields']['calibration']) == \
        ('calibration/people/distil/50/0', 'ok', 'calibration/people/distil/50/0', True)


def test_a_readout_must_be_a_calibration_run_with_its_dial(tmp_path: Any) -> None:
    with pytest.raises(SystemExit):
        session_check.main(['--task', TASK, '--method', METHOD, '--epochs', '50', '--index', '0', '--ledger',
                            str(tmp_path / 'l.jsonl'), '--attempt', '1', '--receivers', 'Gray_Davis'])
