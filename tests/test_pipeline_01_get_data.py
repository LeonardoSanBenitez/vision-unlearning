"""The split builder: one sorted selection per entity, every split derived from it, built twice equal.

Fails on the code before: there was no such script -- the splits came from a notebook whose breeds
and scenes builders took images in filesystem listing order (scenes after an unseeded random
shuffle) and named them with random uuids, so two builds were not the same splits.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

import pytest
from PIL import Image

from vision_unlearning.benchmarks.I_care.pipeline_01_get_data import (
    MANIFEST,
    SplitBuildError,
    build,
    compare,
    main,
    select_per_entity,
)

ENTITIES = ['abbey', 'badlands', 'office']


def _writer(colour: int) -> Callable[[str], None]:
    def write(path: str) -> None:
        Image.new('RGB', (4, 4), (colour, colour, colour)).save(path, format='JPEG')
    return write


def _candidates(per_entity: int = 4) -> Dict[str, Dict[str, Callable[[str], None]]]:
    """Identifiers inserted in descending order, so a listing order and the sorted order disagree."""
    candidates: Dict[str, Dict[str, Callable[[str], None]]] = {}
    for e, entity in enumerate(ENTITIES):
        for k in reversed(range(per_entity)):
            candidates.setdefault(entity, {})[f'{entity}/img_{k}.jpg'] = _writer(10 * e + k)
    return candidates


def test_the_selection_is_the_sorted_first_n_not_the_listing_order() -> None:
    selection = select_per_entity(_candidates(), ENTITIES, 2)
    assert [identifier for identifier, _ in selection['abbey']] == ['abbey/img_0.jpg', 'abbey/img_1.jpg']


def test_an_entity_with_too_few_images_is_refused() -> None:
    candidates = _candidates()
    del candidates['office']['office/img_0.jpg']
    with pytest.raises(SplitBuildError, match='fewer than 4'):
        select_per_entity(candidates, ENTITIES, 4)


def _build(folder: str) -> Dict[str, Any]:
    return build('scenes', ENTITIES, 2, _candidates(), folder, {'kind': 'fixture'})


def test_every_split_follows_the_recipe(tmp_path: Any) -> None:
    manifest = _build(str(tmp_path / 'a'))
    for target in ENTITIES:
        forget, retain = manifest['splits'][target]['forget'], manifest['splits'][target]['retain']
        assert forget == [f'{target}_0000.jpg', f'{target}_0001.jpg']
        assert len(retain) == 2 * (len(ENTITIES) - 1)
        assert not any(name.rsplit('_', 1)[0] == target for name in retain)
    assert manifest['audit_failures'] == []
    with open(tmp_path / 'a' / 'abbey' / 'train_forget' / 'metadata.jsonl', encoding='utf-8') as handle:
        assert json.loads(handle.readline()) == {'file_name': 'abbey_0000.jpg', 'text': 'An image of an abbey scene'}


def test_an_entity_s_retained_images_are_the_same_files_in_every_split(tmp_path: Any) -> None:
    _build(str(tmp_path / 'a'))
    seen = set()
    for target in ('abbey', 'badlands'):
        path = tmp_path / 'a' / target / 'train_retain' / 'office_0001.jpg'
        seen.add(os.path.realpath(path) if os.path.islink(path) else os.path.getsize(path))
        assert path.read_bytes() == (tmp_path / 'a' / '_selected' / 'office' / 'office_0001.jpg').read_bytes()
    assert len(seen) == 1


def test_two_builds_are_identical_and_a_difference_is_reported(tmp_path: Any) -> None:
    _build(str(tmp_path / 'a'))
    _build(str(tmp_path / 'b'))
    assert compare(str(tmp_path / 'a' / MANIFEST), str(tmp_path / 'b' / MANIFEST)) == []
    assert main(['--compare', str(tmp_path / 'a' / MANIFEST), str(tmp_path / 'b' / MANIFEST)]) == 0
    with open(tmp_path / 'b' / MANIFEST, encoding='utf-8') as handle:
        manifest = json.load(handle)
    manifest['selected']['office'][0]['sha256'] = '0' * 64
    with open(tmp_path / 'b' / MANIFEST, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle)
    differences = compare(str(tmp_path / 'a' / MANIFEST), str(tmp_path / 'b' / MANIFEST))
    assert 'selected' in differences and any('office' in d for d in differences)


def test_a_folder_already_built_is_refused(tmp_path: Any) -> None:
    _build(str(tmp_path / 'a'))
    with pytest.raises(SplitBuildError, match='already holds a build'):
        _build(str(tmp_path / 'a'))
