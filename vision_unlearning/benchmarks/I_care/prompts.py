"""The prompts an unlearning session is judged by while it runs.

Four prompts for the concept being removed, four for the concept it is pushed towards, and the
validation prompt a progress image is drawn from. They exist to measure one thing -- can the model
still produce this concept -- so they vary the *phrasing* and nothing else, and the first of them is
the exact string the benchmark generates and scores its corpus with, so that the training-time
number and the benchmark number are commensurable.

There is one builder, and both pipelines call it. Before this module the two carried hand-copied
copies of the lists, which had already drifted: one gave breeds the substitute concept on the retain
side and the other gave it the next entity in metadata order, which meant "retain set clip score
difference" answered a different question for one task than for the other two.

The retain concept is the task's substitute -- ``a child``, ``a cat``, ``the moon`` -- for all three
tasks. It is a complete noun phrase already and is used verbatim: it must not be passed through the
entity-naming rule, which would read ``the moon`` as a bare scene name and return ``a the moon
scene``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from vision_unlearning.datasets.entity_names import (
    canonical_entity,
    generation_prompt,
    substitute_concept,
    type_entity_task,
)


#: The three phrasings that follow the generation prompt, per task. The first prompt of every list
#: is the generation prompt itself and is not repeated here.
#:
#: The phrasings themselves are deliberately unchanged from the ones the published numbers were
#: produced with: rewording them would make new forget and retain scores non-comparable with the
#: old ones for no measurement gain. What changed is ``An picture of`` -> ``A picture of``.
_EXTRA_PHRASINGS: Dict[str, Tuple[str, ...]] = {
    'people': (
        'Photograph of {}; high definition',
        'A picture of {} in the rain',
        'A picture of {} running',
    ),
    'breeds': (
        'Photograph of {}; high definition',
        'A picture of {} in the rain',
        'A picture of {} running',
    ),
    'scenes': (
        'Photograph of {}; high definition',
        'A picture of {} full of people',
        'A picture of {} during the night',
    ),
}

#: How many prompts each side carries. Named so a caller can assert against it.
N_PROMPTS_PER_SIDE = 4


def _prompts_for_concept(task: type_entity_task, concept: str) -> List[str]:
    """Return the four prompts for an already-final concept string."""
    try:
        extra = _EXTRA_PHRASINGS[task]
    except KeyError:
        raise NotImplementedError(f'no evaluation prompts for task {task!r}')
    return [f'An image of {concept}'] + [phrasing.format(concept) for phrasing in extra]


def forget_prompts(task: type_entity_task, entity: str) -> List[str]:
    """Return the four prompts that ask for the entity being unlearned."""
    return _prompts_for_concept(task, canonical_entity(task, entity))


def retain_prompts(task: type_entity_task) -> List[str]:
    """Return the four prompts that ask for the concept the entity is pushed towards.

    It does not depend on the entity: the substitute is a property of the task.
    """
    return _prompts_for_concept(task, substitute_concept(task))


def evaluation_prompts(task: type_entity_task, entity: str) -> Tuple[List[str], List[str]]:
    """Return ``(forget, retain)``, four prompts each, for one unlearning session.

    The prompt a progress image is drawn from is the first of the forget list, which is the
    generation prompt of the entity; there is no separate builder for it, so the two cannot drift.
    """
    return forget_prompts(task, entity), retain_prompts(task)
