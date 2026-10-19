"""Every generated image is named, and prompted, by the one builder each.

Fails on the code before: thirteen sites spelled the generation prompt by hand as
``f"An image of {get_target_overwrite(...)[0]}"`` and four built image file names without the
two-digit seed padding the readers use, so a seed below 10 named a file no reader could find.
"""
from __future__ import annotations

import os
import re
from typing import List

import pytest

from vision_unlearning.datasets.entity_names import generation_prompt
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_target_overwrite

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vision_unlearning')
EXCLUDED = ('u_care', 'hub_care', 'ablations', 'reports')
HAND_BUILT = [
    re.compile(r'f["\']An image of \{get_target_overwrite'),
    re.compile(r'f["\'](?:on|off)_\{seed\}_'),
    re.compile(r'f["\']\{lora_state\}_\{seed\}_'),
]


def _sources() -> List[str]:
    paths = []
    for root, dirs, files in os.walk(PACKAGE):
        dirs[:] = [d for d in dirs if d not in EXCLUDED]
        paths += [os.path.join(root, name) for name in files if name.endswith('.py')]
    return paths


def test_no_generation_prompt_or_image_name_is_built_by_hand() -> None:
    offenders = []
    for path in _sources():
        with open(path, encoding='utf-8') as handle:
            for number, line in enumerate(handle, start=1):
                if any(pattern.search(line) for pattern in HAND_BUILT):
                    offenders.append(f'{os.path.relpath(path, PACKAGE)}:{number}')
    assert not offenders, offenders


@pytest.mark.parametrize('task, name', [
    ('people', 'George_W_Bush'), ('people', 'Jennifer_Lopez'),
    ('breeds', 'basenji dog'), ('breeds', 'english setter dog'),
    ('scenes', 'abbey'), ('scenes', 'office'), ('scenes', 'football_field'),
])
def test_the_one_builder_spells_the_prompt_the_old_composition_did(task: str, name: str) -> None:
    assert generation_prompt(task, name) == f"An image of {get_target_overwrite(task, 'distil', name)[0]}"  # type: ignore[arg-type]


def test_a_single_digit_seed_names_the_file_its_readers_look_for() -> None:
    assert get_generated_dataset_file('off', 7, 'An image of x') == 'off_07_An image of x.png'
