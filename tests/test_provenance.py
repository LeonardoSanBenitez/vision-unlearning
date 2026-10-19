"""provenance.json: written beside what a run produced, and refused when read under other inputs.

Fails on the code before: no generated folder or model folder recorded anything about what produced
it, so an image folder made from a reordered prompt list or another seed list looked exactly like
the right one -- the defect that once voided a whole latent capture.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from vision_unlearning.benchmarks.I_care.provenance import (
    ProvenanceMismatch,
    file_sha256,
    producer_facts,
    prompt_list_sha256,
    read_provenance,
    write_provenance,
)

PROMPTS = ['An image of a', 'An image of b', 'An image of c']
SEEDS = [42, 43]


def _folder(tmp_path: Any, prompts: list = PROMPTS, seeds: list = SEEDS) -> str:
    folder = str(tmp_path / 'generated')
    write_provenance(folder, {'seeds': seeds, 'prompts_ordered': prompts, 'prompt_list_sha256': prompt_list_sha256(prompts)})
    return folder


def test_a_record_is_written_and_read_back(tmp_path: Any) -> None:
    assert read_provenance(_folder(tmp_path), PROMPTS, SEEDS)['seeds'] == SEEDS


def test_a_reordered_prompt_list_is_refused(tmp_path: Any) -> None:
    with pytest.raises(ProvenanceMismatch, match='a reordering of'):
        read_provenance(_folder(tmp_path), list(reversed(PROMPTS)), SEEDS)


def test_another_prompt_list_is_refused(tmp_path: Any) -> None:
    with pytest.raises(ProvenanceMismatch, match='another'):
        read_provenance(_folder(tmp_path), PROMPTS[:2], SEEDS)


def test_another_seed_list_is_refused(tmp_path: Any) -> None:
    with pytest.raises(ProvenanceMismatch, match='seeds'):
        read_provenance(_folder(tmp_path), PROMPTS, [42, 44])


def test_a_folder_without_a_record_is_refused(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match='records nothing'):
        read_provenance(str(tmp_path), PROMPTS, SEEDS)


def test_the_producer_facts_carry_the_pinned_input_hashes(tmp_path: Any) -> None:
    pins = tmp_path / 'pins.json'
    pins.write_text('{"base_model": "abc"}')
    facts = producer_facts(precision='float16', pins_path=str(pins), card='NVIDIA A40', task='people')
    assert facts['pins_sha256'] == file_sha256(str(pins)) and facts['freeze_sha256'] is None
    assert facts['card_model'] == 'NVIDIA A40' and facts['precision'] == 'float16' and facts['task'] == 'people'
    assert set(facts) >= {'commit', 'tree_clean'}
    json.dumps(facts)
