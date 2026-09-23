"""The evaluation prompts of an unlearning session, pinned exactly.

These four strings per side decide the two numbers a session reports while it trains, so they are
asserted verbatim rather than by shape. Three properties matter beyond the exact text:

1. the first forget prompt is the string the benchmark generates and scores its corpus with, so the
   training-time number and the benchmark number are commensurable;
2. the retain side asks for the task's substitute concept -- for **all three** tasks. It used to
   ask, for breeds only, about the next entity in metadata order, which made one task's "retain"
   number answer a different question from the other two;
3. no prompt depends on anything but the task and the entity.

There is also a source-level check that neither pipeline builds a list of its own again.
"""
import re
from pathlib import Path
from typing import List

import pytest

from vision_unlearning.benchmarks.I_care import prompts
from vision_unlearning.datasets.entity_names import generation_prompt, substitute_concept


_PIPELINES = (
    'vision_unlearning/benchmarks/I_care/pipeline_03_unlearn_model.py',
    'vision_unlearning/benchmarks/I_care/pipeline_04_generate_dataset.py',
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class TestTheExactPrompts:
    def test_people(self) -> None:
        forget, retain = prompts.evaluation_prompts('people', 'Mark_Philippoussis')
        assert forget == [
            'An image of Mark Philippoussis',
            'Photograph of Mark Philippoussis; high definition',
            'A picture of Mark Philippoussis in the rain',
            'A picture of Mark Philippoussis running',
        ]
        assert retain == [
            'An image of a child',
            'Photograph of a child; high definition',
            'A picture of a child in the rain',
            'A picture of a child running',
        ]

    def test_breeds(self) -> None:
        forget, retain = prompts.evaluation_prompts('breeds', 'bouvier des flandres dog')
        assert forget == [
            'An image of a bouvier des flandres dog',
            'Photograph of a bouvier des flandres dog; high definition',
            'A picture of a bouvier des flandres dog in the rain',
            'A picture of a bouvier des flandres dog running',
        ]
        assert retain == [
            'An image of a cat',
            'Photograph of a cat; high definition',
            'A picture of a cat in the rain',
            'A picture of a cat running',
        ]

    def test_scenes(self) -> None:
        forget, retain = prompts.evaluation_prompts('scenes', 'football_field')
        assert forget == [
            'An image of a football field scene',
            'Photograph of a football field scene; high definition',
            'A picture of a football field scene full of people',
            'A picture of a football field scene during the night',
        ]
        assert retain == [
            'An image of the moon',
            'Photograph of the moon; high definition',
            'A picture of the moon full of people',
            'A picture of the moon during the night',
        ]

    def test_an_unknown_task_is_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            prompts.forget_prompts('objects', 'a hammer')  # type: ignore[arg-type]


class TestTheProperties:
    TASKS_AND_ENTITIES = [
        ('people', 'Mark_Philippoussis'),
        ('breeds', 'bouvier des flandres dog'),
        ('scenes', 'football_field'),
    ]

    @pytest.mark.parametrize('task, entity', TASKS_AND_ENTITIES)
    def test_the_first_forget_prompt_is_the_generation_prompt(self, task: str, entity: str) -> None:
        forget, _ = prompts.evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert forget[0] == generation_prompt(task, entity)  # type: ignore[arg-type]

    @pytest.mark.parametrize('task, entity', TASKS_AND_ENTITIES)
    def test_the_retain_side_is_the_substitute_concept(self, task: str, entity: str) -> None:
        _, retain = prompts.evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert retain[0] == f'An image of {substitute_concept(task)}'  # type: ignore[arg-type]
        for prompt in retain:
            assert substitute_concept(task) in prompt  # type: ignore[arg-type]

    def test_the_retain_side_does_not_depend_on_which_entity_is_being_unlearned(self) -> None:
        """The defect this replaces: for breeds the retain prompts were built from the next entity
        in metadata order, so 11 of 100 breeds were measured against a neighbour sharing a word
        with the forget concept."""
        first = prompts.retain_prompts('breeds')
        _, from_one_entity = prompts.evaluation_prompts('breeds', 'bouvier des flandres dog')
        _, from_another = prompts.evaluation_prompts('breeds', 'affenpinscher dog')
        assert first == from_one_entity == from_another

    @pytest.mark.parametrize('task, entity', TASKS_AND_ENTITIES)
    def test_both_sides_carry_four_prompts(self, task: str, entity: str) -> None:
        forget, retain = prompts.evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert len(forget) == len(retain) == prompts.N_PROMPTS_PER_SIDE

    @pytest.mark.parametrize('task, entity', TASKS_AND_ENTITIES)
    def test_no_prompt_carries_the_broken_article_or_an_underscore(self, task: str, entity: str) -> None:
        """``An picture of`` was in two of the four prompts of every list, and a dead
        ``.replace('_', ' ')`` sat on a string whose underscores are already gone."""
        forget, retain = prompts.evaluation_prompts(task, entity)  # type: ignore[arg-type]
        for prompt in forget + retain:
            assert 'An picture of' not in prompt
            assert '_' not in prompt

    @pytest.mark.parametrize('task, entity', TASKS_AND_ENTITIES)
    def test_the_two_sides_differ_only_in_the_concept(self, task: str, entity: str) -> None:
        forget, retain = prompts.evaluation_prompts(task, entity)  # type: ignore[arg-type]
        concept_forget = forget[0][len('An image of '):]
        concept_retain = retain[0][len('An image of '):]
        assert [p.replace(concept_forget, concept_retain) for p in forget] == retain


class TestNeitherPipelineBuildsItsOwn:
    """A grep is not the oracle for the strings -- the tests above are -- but it is the only way to
    see that a pipeline stopped carrying its own copy."""

    @pytest.mark.parametrize('relative', _PIPELINES)
    def test_the_pipeline_imports_the_builder(self, relative: str) -> None:
        source = (_repo_root() / relative).read_text(encoding='utf-8')
        assert 'from vision_unlearning.benchmarks.I_care.prompts import' in source

    @pytest.mark.parametrize('relative', _PIPELINES)
    def test_the_pipeline_assigns_no_prompt_list_of_its_own(self, relative: str) -> None:
        source = (_repo_root() / relative).read_text(encoding='utf-8')
        assignments: List[str] = re.findall(r'^\s*example_prompts_\w+\s*=\s*\[', source, re.MULTILINE)
        assert assignments == []

    @pytest.mark.parametrize('relative', _PIPELINES)
    def test_the_pipeline_carries_no_broken_article(self, relative: str) -> None:
        source = (_repo_root() / relative).read_text(encoding='utf-8')
        assert 'An picture of' not in source
