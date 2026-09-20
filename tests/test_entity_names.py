"""Contract for the entity-name transforms.

Every string the I-CARE benchmark builds from an entity name starts here: the prompt an image is
generated with, the entity recorded beside its embeddings, the caption a model is trained on, the
folder an artifact is written to. This file is the contract those strings are held to, and it is
deliberately free of torch so that it runs in the lite tier -- the tier that actually gates a
change. (The previous characterization of these transforms lived in ``test_testbed.py``, which
``tests/conftest.py`` excludes from collection whenever torch is absent, so the check that was
meant to protect the corpus never ran in the environment that was used to verify it.)

Two different claims are made here, and they are kept apart on purpose:

* **Nothing the corpus depends on may change.** The generation prompt of all 300 entities is
  frozen against ``tests/fixtures/entity_names_expected.json``. That file is a snapshot of the
  behaviour that produced every published image, and a diff to it is an orphaned corpus.
* **What was wrong must now be right.** The canonical form is idempotent, people's underscores are
  gone, breeds carry their article, and the article rule knows the words English pronounces
  against their spelling. These assertions fail against the implementation that preceded them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.datasets.entity_names import (
    article_for,
    canonical_entity,
    display_entity,
    generation_prompt,
    substitute_concept,
)
from vision_unlearning.datasets.testbed import get_target_overwrite, get_target_preprocessed

TASKS = ['people', 'breeds', 'scenes']
METHODS = ['distil', 'munba', 'uce', 'salun']

_FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'entity_names_expected.json')
_METADATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'vision_unlearning', 'benchmarks', 'I_care', 'assets',
)


def _expected() -> Dict[str, List[Dict[str, str]]]:
    with open(_FIXTURE, 'r', encoding='utf-8') as handle:
        return json.load(handle)['tasks']


def _rows(task: str) -> List[Dict[str, str]]:
    return _expected()[task]


def _live_metadata_names(task: str) -> List[str]:
    path = os.path.join(_METADATA_DIR, f'metadata_{task}_2_enriched_filtered.json')
    with open(path, 'r', encoding='utf-8') as handle:
        metadata: List[Dict[str, Any]] = json.load(handle)
    return [entry['name'] for entry in metadata]


class TestPublishedCorpusIsNotOrphaned:
    """The 300 prompts the corpus was generated with, frozen."""

    @pytest.mark.parametrize('task', TASKS)
    def test_generation_prompt_is_byte_identical_for_every_entity(self, task: str) -> None:
        rows = _rows(task)
        assert len(rows) == 100, f'{task}: expected 100 entities in the fixture, found {len(rows)}'
        wrong = [
            (row['name'], row['generation_prompt'], generation_prompt(task, row['name']))
            for row in rows
            if generation_prompt(task, row['name']) != row['generation_prompt']
        ]
        assert not wrong, f'{task}: {len(wrong)} of {len(rows)} generation prompts changed: {wrong[:5]}'

    @pytest.mark.parametrize('task', TASKS)
    def test_generation_prompt_does_not_depend_on_the_unlearning_method(self, task: str) -> None:
        for row in _rows(task):
            forms = {method: get_target_overwrite(task, method, row['name'])[0] for method in METHODS}
            assert len(set(forms.values())) == 1, f'{task}/{row["name"]}: method changes the form: {forms}'

    @pytest.mark.parametrize('task', TASKS)
    def test_substitute_concept_matches_the_published_one(self, task: str) -> None:
        for row in _rows(task):
            assert substitute_concept(task) == row['substitute_concept']

    @pytest.mark.parametrize('task', TASKS)
    def test_fixture_still_describes_the_entities_on_disk(self, task: str) -> None:
        """Guards the fixture against drifting away from the metadata it was taken from.

        Skipped where that metadata is absent -- it is git-ignored, so continuous integration
        never has it. This is the one check that needs the real corpus; every other check in this
        file runs everywhere.
        """
        path = os.path.join(_METADATA_DIR, f'metadata_{task}_2_enriched_filtered.json')
        if not os.path.exists(path):
            pytest.skip(f'task metadata not present on this machine: {path}')
        assert [row['name'] for row in _rows(task)] == _live_metadata_names(task)


class TestCanonicalForm:
    """What the transform must now produce, which is not what it produced before."""

    @pytest.mark.parametrize('task', TASKS)
    def test_is_idempotent_over_every_entity(self, task: str) -> None:
        once = [canonical_entity(task, row['name']) for row in _rows(task)]
        twice = [canonical_entity(task, name) for name in once]
        wrong = [(a, b) for a, b in zip(once, twice) if a != b]
        assert not wrong, f'{task}: {len(wrong)} of {len(once)} names change on a second pass: {wrong[:5]}'

    def test_is_idempotent_over_the_forms_already_stored_on_disk(self) -> None:
        """The exact strings the scenes embedding files carry today must survive untouched.

        ``pipeline_05`` hands an already-transformed name to ``embeddings.py``, which transforms it
        again; that is how ``an an abbey scene scene`` came to be written into all 301 scenes
        embedding files. Idempotence is what makes that composition harmless.
        """
        for stored in ['an abbey scene', 'a velodrome outdoor scene', 'an organ loft exterior scene']:
            assert canonical_entity('scenes', stored) == stored

    def test_people_lose_their_underscores(self) -> None:
        assert canonical_entity('people', 'Mark_Philippoussis') == 'Mark Philippoussis'
        assert canonical_entity('people', 'Roh_Moo-hyun') == 'Roh Moo-hyun'

    def test_breeds_carry_their_article(self) -> None:
        assert canonical_entity('breeds', 'basenji dog') == 'a basenji dog'
        assert canonical_entity('breeds', 'affenpinscher dog') == 'an affenpinscher dog'

    def test_scenes_carry_their_article_and_suffix(self) -> None:
        assert canonical_entity('scenes', 'abbey') == 'an abbey scene'
        assert canonical_entity('scenes', 'phone_booth') == 'a phone booth scene'

    def test_generation_prompt_is_the_canonical_form_by_construction(self) -> None:
        for task in TASKS:
            for row in _rows(task):
                assert generation_prompt(task, row['name']) == f'An image of {canonical_entity(task, row["name"])}'

    def test_rejects_an_unknown_task(self) -> None:
        with pytest.raises(NotImplementedError):
            canonical_entity('objects', 'anything')  # type: ignore[arg-type]

    def test_rejects_an_empty_name(self) -> None:
        for empty in ['', '   ', '_']:
            with pytest.raises(ValueError):
                canonical_entity('breeds', empty)


class TestArticle:
    """The cases the corpus does not contain, which is why the corpus cannot test them."""

    @pytest.mark.parametrize('word, expected', [
        ('abbey', 'an'),
        ('igloo', 'an'),
        ('basenji', 'a'),
        ('hour', 'an'),
        ('honest broker', 'an'),
        ('university', 'a'),
        ('utility room', 'a'),
        ('european street', 'a'),
        ('one way street', 'a'),
    ])
    def test_follows_pronunciation_where_spelling_disagrees(self, word: str, expected: str) -> None:
        assert article_for(word) == expected

    def test_the_exceptions_reach_the_canonical_form(self) -> None:
        assert canonical_entity('scenes', 'utility_room') == 'a utility room scene'
        assert canonical_entity('scenes', 'hour_glass') == 'an hour glass scene'


class TestPublicSurface:
    """Moving the functions must not break an import that already exists."""

    def test_testbed_still_exposes_both_helpers(self) -> None:
        assert get_target_preprocessed('scenes', 'abbey') == 'an abbey scene'
        assert get_target_overwrite('scenes', 'uce', 'abbey') == ('an abbey scene', 'the moon')

    def test_the_two_helpers_now_agree_with_each_other(self) -> None:
        """The defect this ticket is named after: the two helpers disagreed for two tasks of three."""
        for task in TASKS:
            for row in _rows(task)[:20]:
                preprocessed, _ = get_target_overwrite(task, 'uce', row['name'])
                assert get_target_preprocessed(task, row['name']) == preprocessed

    def test_display_entity_is_public_and_behaves_as_the_private_helper_did(self) -> None:
        assert display_entity('a bouvier des flandres dog') == 'bouvier des flandres dog'
        assert display_entity('An ice skating rink') == 'ice skating rink'
        assert display_entity('George W. Bush') == 'George W. Bush'
        assert display_entity('a' + 'x' * 40, max_chars=10) == 'a' + 'x' * 8 + '…'
