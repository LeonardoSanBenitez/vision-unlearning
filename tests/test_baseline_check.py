"""The check that closes a task's baseline unit, and its ledger line.

Fails on the code before this change: ``session_check`` had no ``check_baseline`` and no ``--baseline``,
so a baseline, which every session is paired against, entered the campaign with no count, no
certification of its captures and no ledger line.
"""
import json
import math
import os
from typing import Any, Dict, List, Literal

import pytest

from vision_unlearning.benchmarks.I_care import session_check
from vision_unlearning.benchmarks.I_care.baseline_competence import competence_path
from vision_unlearning.benchmarks.I_care.metadata import BaselineEmbeddings
from vision_unlearning.benchmarks.I_care.provenance import prompt_list_sha256
from vision_unlearning.benchmarks.I_care.run_ledger import read_records
from vision_unlearning.benchmarks.I_care.similarity import Similarity, UnetLatentSimilarity
from vision_unlearning.datasets.entity_names import generation_prompt
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_shared_baseline_folder
from vision_unlearning.utils.mechanistic_interpretability import get_act_fingerprints_path
from vision_unlearning.utils.noise import pairing_record

TASK: Literal['people'] = 'people'
ENTITIES = ['George_W_Bush', 'Serena_Williams', 'Gray_Davis']
SEEDS = [42, 43]
PNDM = {'skip_prk_steps': True, 'num_train_timesteps': 1000}
IMAGE_KWARGS = {'num_inference_steps': 50, 'guidance_scale': 7.5, 'batch_size': 8, 'dtype': 'float16'}
ACT_KWARGS = {'num_inference_steps': 10, 'guidance_scale': 7.5, 'batch_size': 8, 'dtype': 'float16'}


def _prompts() -> List[str]:
    return [generation_prompt(TASK, e) for e in ENTITIES]  # type: ignore[arg-type]


def _record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    keys = [{'prompt_index': i, 'seed': s, 'file_name': get_generated_dataset_file('off', s, p),
             'initial_latent_sha256': f'latent-{s}'} for s in SEEDS for i, p in enumerate(_prompts())]
    return pairing_record(keys, 'PNDMScheduler', PNDM, kwargs, latent_hash_schema_version=1)


def _dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle)


def _baseline_unit(base: str) -> None:
    """A complete baseline unit on disk."""
    prompts = _prompts()
    _dump(os.path.join(base, f'metadata_{TASK}_2_enriched_filtered.json'), [{'name': e} for e in ENTITIES])
    folder = get_shared_baseline_folder(TASK, base)
    _dump(os.path.join(folder, 'provenance.json'), {
        'commit': 'abc', 'tree_clean': True, 'card_model': 'A40', 'pins_sha256': 'p', 'freeze_sha256': 'f',
        'seeds': SEEDS, 'prompts_ordered': prompts, 'prompt_list_sha256': prompt_list_sha256(prompts),
        'pairing': _record(IMAGE_KWARGS)})
    for s in SEEDS:
        for p in prompts:
            with open(os.path.join(folder, get_generated_dataset_file('off', s, p)), 'wb') as handle:
                handle.write(f'{s}{p}'.encode())
    _dump(BaselineEmbeddings(task=TASK, base_folder=base)._get_data_path_local(),  # type: ignore[arg-type]
          {'embeddings': [{'prompt': p, 'seed': s, 'embedding': [1.0]} for p in prompts for s in SEEDS]})
    _dump(get_act_fingerprints_path(TASK, 'sd1.4', base),
          {'metadata': {'pairing': _record(ACT_KWARGS)}, 'fingerprints': [{'entity': e, 'fingerprint': [1.0]} for e in ENTITIES]})
    _dump(UnetLatentSimilarity(task=TASK, base_folder=base).cache_path(),  # type: ignore[arg-type]
          {'metadata': {'pairing': _record(IMAGE_KWARGS), 'baseline_verification': {'n_checked': 6, 'max_mean_abs_difference': 0.001}},
           'entities': []})
    for metric in session_check.BASELINE_SIMILARITIES:
        _dump(Similarity(task=TASK, similarity_metric=metric, base_folder=base)._get_data_path_local(),  # type: ignore[arg-type]
              [{'emitter': e, **{r: 0.5 for r in ENTITIES}} for e in ENTITIES])
    _dump(competence_path(TASK, base), {'seeds': SEEDS, 'per_image': {e: {} for e in ENTITIES}, 'f_min': 2.5,
                                        'screen': {'recognised': 3, 'passes': False}})


def test_a_complete_baseline_unit_passes_and_is_digested(tmp_path: Any) -> None:
    base = str(tmp_path)
    _baseline_unit(base)
    fields = session_check.check_baseline(TASK, ENTITIES, base, SEEDS)  # type: ignore[arg-type]
    assert fields['session'] == 'people/baseline'
    assert fields['counts'] == {'images': 6, 'embeddings': 6, 'act_entities': 3, 'unet_latent_verified': 6,
                                'similarity_matrices': 5, 'competence_entities': 3}
    assert fields['pairing'] == 'paired'
    assert fields['competence'] == {'f_min': 2.5, 'recognised': 3, 'passes': False}
    outputs = fields['outputs']
    assert len([p for p in outputs if p.startswith('datasets/generated_people_baseline/')]) == 7  # 6 images + provenance
    assert 'datasets/embeddings_people_original.json' in outputs
    assert 'datasets/act_fingerprints_people_sd1.4.json' in outputs
    assert 'datasets/baseline_competence_people.json' in outputs
    assert sum(p.startswith('similarity_') for p in outputs) == 5
    assert not any('unet_latents' in p for p in outputs)


def _break(base: str, what: str) -> None:
    if what == 'image':
        os.remove(os.path.join(get_shared_baseline_folder(TASK, base), get_generated_dataset_file('off', 43, _prompts()[1])))
    elif what == 'act steps':
        _dump(get_act_fingerprints_path(TASK, 'sd1.4', base),
              {'metadata': {'pairing': _record(IMAGE_KWARGS)}, 'fingerprints': [{'entity': e, 'fingerprint': [1.0]} for e in ENTITIES]})
    elif what == 'latents unverified':
        path = UnetLatentSimilarity(task=TASK, base_folder=base).cache_path()  # type: ignore[arg-type]
        _dump(path, {'metadata': {'pairing': _record(IMAGE_KWARGS), 'baseline_verification': {'n_checked': 5}}, 'entities': []})
    elif what == 'matrix nan':
        path = Similarity(task=TASK, similarity_metric='dino', base_folder=base)._get_data_path_local()  # type: ignore[arg-type]
        _dump(path, [{'emitter': e, **{r: (math.nan if r == e else 0.5) for r in ENTITIES}} for e in ENTITIES])
    elif what == 'f_min':
        _dump(competence_path(TASK, base), {'seeds': SEEDS, 'per_image': {e: {} for e in ENTITIES}, 'f_min': 0.0,
                                            'screen': {'recognised': 3, 'passes': False}})


@pytest.mark.parametrize('what, reported', [
    ('image', '5 of 6 images'),
    ('act steps', 'act failed'),
    ('latents unverified', '5 of 6 latents checked'),
    ('matrix nan', 'similarity dino'),
    ('f_min', 'competence f_min is 0.0'),
])
def test_each_defect_fails_the_unit(tmp_path: Any, what: str, reported: str) -> None:
    base = str(tmp_path)
    _baseline_unit(base)
    _break(base, what)
    with pytest.raises(session_check.SessionCheckFailure, match=reported):
        session_check.check_baseline(TASK, ENTITIES, base, SEEDS)  # type: ignore[arg-type]


def test_the_command_writes_one_ledger_line_keyed_by_the_task_baseline(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    base = str(tmp_path)
    _baseline_unit(base)
    monkeypatch.setattr(session_check, 'GENERATE_DATASET_SEEDS', SEEDS)
    ledger = os.path.join(base, 'ledger.jsonl')
    code = session_check.main(['--task', TASK, '--baseline', '--base-folder', base, '--ledger', ledger, '--attempt', '1'])
    assert code == 0
    [line] = read_records(ledger)
    assert (line['context'], line['status']) == ('people/baseline', 'ok')
    assert (line['fields']['commit'], line['fields']['card_model'], line['fields']['attempt']) == ('abc', 'A40', 1)


def test_a_session_still_needs_its_method_epochs_and_index(tmp_path: Any) -> None:
    with pytest.raises(SystemExit):
        session_check.main(['--task', TASK, '--ledger', str(tmp_path / 'l.jsonl'), '--attempt', '1'])
