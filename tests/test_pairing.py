"""The input-pairing check, on synthetic records where the right verdict is known.

Fails on the code before: ``vision_unlearning.benchmarks.I_care.pairing`` did not exist, and the
first corpus paired every unlearned image with a baseline image drawn from other noise without any
check noticing.

Every negative case below keeps the scalar metadata equal -- the same number of keys, the same seeds,
the same prompt list -- so none of them can pass by comparing counts.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from vision_unlearning.benchmarks.I_care.pairing import (
    ACT_STEPS,
    PairingFailure,
    check,
    output_distance,
    pairing_record,
    restrict,
)
from vision_unlearning.datasets.testbed import get_generated_dataset_file

PROMPTS = [f'An image of entity {k}' for k in range(10)]   # 10 prompts: batches of 8 and a short 2
SEEDS = [42, 43, 44, 45]
PNDM = {'skip_prk_steps': True, 'num_train_timesteps': 1000, 'beta_schedule': 'scaled_linear'}
KWARGS = {'num_inference_steps': 50, 'guidance_scale': 7.5, 'batch_size': 8, 'dtype': 'float16'}


def _record(lora_state: str, kwargs: Dict[str, Any] = KWARGS, scheduler: str = 'PNDMScheduler',
            config: Dict[str, Any] = PNDM) -> Dict[str, Any]:
    keys = [
        {'prompt_index': i, 'seed': s, 'file_name': get_generated_dataset_file(lora_state, s, p),  # type: ignore[arg-type]
         'initial_latent_sha256': f'latent-of-seed-{s}'}
        for s in SEEDS for i, p in enumerate(PROMPTS)
    ]
    return pairing_record(keys, scheduler, config, kwargs, latent_hash_schema_version=1)


def _check(candidate: Dict[str, Any], baseline: Dict[str, Any], check_class: str = 'image_pairs') -> Dict[str, Any]:
    return check(candidate, baseline, PROMPTS, SEEDS, check_class, file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]


def _key(record: Dict[str, Any], index: int, seed: int) -> Dict[str, Any]:
    return next(k for k in record['keys'] if k['prompt_index'] == index and k['seed'] == seed)


# -- what passes ----------------------------------------------------------------------------------


def test_identical_inputs_pass() -> None:
    assert _check(_record('on'), _record('off'))['verdict'] == 'paired'


def test_a_strongly_edited_model_with_matching_inputs_passes(tmp_path: Any) -> None:
    """The check reads inputs only: a black output against a white baseline is still a pair."""
    Image.new('RGB', (8, 8), (255, 255, 255)).save(tmp_path / 'off.png')
    Image.new('RGB', (8, 8), (0, 0, 0)).save(tmp_path / 'on.png')
    assert output_distance(str(tmp_path / 'off.png'), str(tmp_path / 'on.png')) == 255.0
    assert _check(_record('on'), _record('off'))['verdict'] == 'paired'


def test_zero_preserved_receivers_is_not_a_pairing_failure() -> None:
    """Every receiver destroyed is an interference outcome; the inputs are what pairing is about."""
    assert _check(_record('on'), _record('off'))['keys'] == len(PROMPTS) * len(SEEDS)


# -- what fails -----------------------------------------------------------------------------------


@pytest.mark.parametrize('index', [3, 9], ids=['a non-first occurrence of the seed', 'the short final batch'])
def test_an_altered_latent_fails(index: int) -> None:
    candidate = _record('on')
    _key(candidate, index, 44)['initial_latent_sha256'] = 'some-other-latent'
    with pytest.raises(PairingFailure):
        _check(candidate, _record('off'))


def test_a_whole_seed_on_other_noise_fails_even_when_internally_consistent() -> None:
    candidate = _record('on')
    for key in candidate['keys']:
        if key['seed'] == 45:
            key['initial_latent_sha256'] = 'drawn-elsewhere'
    with pytest.raises(PairingFailure, match='another latent'):
        _check(candidate, _record('off'))


def test_a_missing_key_fails() -> None:
    candidate = _record('on')
    candidate['keys'] = candidate['keys'][:-1]
    with pytest.raises(PairingFailure, match='missing'):
        _check(candidate, _record('off'))


def test_a_duplicated_key_fails_even_with_the_count_unchanged() -> None:
    candidate = _record('on')
    candidate['keys'][-1] = copy.deepcopy(candidate['keys'][0])
    assert len(candidate['keys']) == len(PROMPTS) * len(SEEDS)
    with pytest.raises(PairingFailure, match='duplicated'):
        _check(candidate, _record('off'))


def test_a_prompt_and_seed_association_swapped_between_two_files_fails() -> None:
    candidate = _record('on')
    a, b = _key(candidate, 1, 43), _key(candidate, 2, 43)
    a['file_name'], b['file_name'] = b['file_name'], a['file_name']
    with pytest.raises(PairingFailure, match='not named for their prompt and seed'):
        _check(candidate, _record('off'))


def test_a_scheduler_class_change_fails() -> None:
    with pytest.raises(PairingFailure, match='not allowed'):
        _check(_record('on', scheduler='EulerAncestralDiscreteScheduler'), _record('off'))


def test_a_scheduler_configuration_change_fails() -> None:
    with pytest.raises(PairingFailure, match='scheduler configuration differs'):
        _check(_record('on', config={**PNDM, 'beta_schedule': 'linear'}), _record('off'))
    with pytest.raises(PairingFailure, match='requires'):
        _check(_record('on', config={**PNDM, 'skip_prk_steps': False}), _record('off'))


def test_scheduler_bookkeeping_entries_do_not_count_as_a_change() -> None:
    # diffusers builds _use_default_values from a set, so two processes loading the same scheduler
    # record it in different orders; that and the other underscore entries are not parameters.
    base = _record('off', config={**PNDM, '_class_name': 'PNDMScheduler', '_use_default_values': ['prediction_type', 'timestep_spacing']})
    cand = _record('on', config={**PNDM, '_class_name': 'PNDMScheduler', '_use_default_values': ['timestep_spacing', 'prediction_type']})
    assert _check(cand, base)['verdict'] == 'paired'


@pytest.mark.parametrize('changed', [{'num_inference_steps': 49}, {'guidance_scale': 7.0}])
def test_a_changed_effective_argument_fails(changed: Dict[str, Any]) -> None:
    with pytest.raises(PairingFailure, match='effective generation arguments'):
        _check(_record('on', kwargs={**KWARGS, **changed}), _record('off'))


# -- the activation capture ------------------------------------------------------------------------


def test_the_activation_capture_at_ten_steps_passes() -> None:
    act = _record('on', kwargs={**KWARGS, 'num_inference_steps': ACT_STEPS})
    assert _check(act, _record('off'), 'act')['verdict'] == 'paired'


def test_the_activation_capture_at_another_step_count_fails() -> None:
    with pytest.raises(PairingFailure, match='10 steps'):
        _check(_record('on', kwargs={**KWARGS, 'num_inference_steps': 50}), _record('off'), 'act')


def test_the_activation_capture_with_a_step_count_differing_between_entities_fails() -> None:
    act = _record('on', kwargs={**KWARGS, 'num_inference_steps': ACT_STEPS})
    _key(act, 4, 42)['num_inference_steps'] = 12
    with pytest.raises(PairingFailure, match='10 steps'):
        _check(act, _record('off'), 'act')


def test_the_activation_capture_still_compares_guidance() -> None:
    act = _record('on', kwargs={**KWARGS, 'num_inference_steps': ACT_STEPS, 'guidance_scale': 5.0})
    with pytest.raises(PairingFailure, match='guidance'):
        _check(act, _record('off'), 'act')


def test_the_latent_capture_is_held_to_the_image_assertions() -> None:
    with pytest.raises(PairingFailure, match='effective generation arguments'):
        _check(_record('on', kwargs={**KWARGS, 'num_inference_steps': ACT_STEPS}), _record('off'), 'unet_latent')


def test_records_hashed_under_different_encodings_are_refused() -> None:
    candidate = _record('on')
    candidate['latent_hash_schema_version'] = 2
    with pytest.raises(PairingFailure, match='encodings'):
        _check(candidate, _record('off'))


# -- a read-out of a few prompts, certified against the baseline restricted to them --------------

SUBSET = [PROMPTS[7], PROMPTS[2], PROMPTS[9]]


def _readout(subset: List[str] = SUBSET) -> Dict[str, Any]:
    keys = [
        {'prompt_index': i, 'seed': s, 'file_name': get_generated_dataset_file('on', s, p),  # type: ignore[arg-type]
         'initial_latent_sha256': f'latent-of-seed-{s}'}
        for s in SEEDS for i, p in enumerate(subset)
    ]
    return pairing_record(keys, 'PNDMScheduler', PNDM, KWARGS, latent_hash_schema_version=1)


def test_a_readout_passes_against_the_baseline_restricted_to_its_prompts() -> None:
    """Fails on the code before: there was no restriction, and a read-out's keys, indexed by position in
    its own short prompt list, could only be compared with the whole baseline's and failed."""
    restricted = restrict(_record('off'), PROMPTS, SUBSET)
    assert len(restricted['keys']) == len(SUBSET) * len(SEEDS)
    summary = check(_readout(), restricted, SUBSET, SEEDS, 'image_pairs', file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]
    assert summary['keys'] == 12


def test_a_readout_whose_files_are_swapped_still_fails_after_restriction() -> None:
    readout = _readout()
    first, second = readout['keys'][0], readout['keys'][1]
    first['file_name'], second['file_name'] = second['file_name'], first['file_name']
    with pytest.raises(PairingFailure, match='not named for their prompt and seed'):
        check(readout, restrict(_record('off'), PROMPTS, SUBSET), SUBSET, SEEDS, 'image_pairs', file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]


def test_restricting_to_a_prompt_the_record_does_not_hold_fails() -> None:
    with pytest.raises(PairingFailure, match='not in the record'):
        restrict(_record('off'), PROMPTS, ['An image of nobody'])
