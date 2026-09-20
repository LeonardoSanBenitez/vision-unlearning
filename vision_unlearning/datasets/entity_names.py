"""How an entity is spelled, in one place.

A benchmark that unlearns concepts spends most of its life turning an entity's name into a string:
the prompt an image is generated with, the entity recorded beside its embeddings, the caption a
model is conditioned on, the folder an artifact is written to, the label under a plot. When those
strings are built in several places they drift, and a drift of one character means the images, the
embeddings and the metrics no longer describe the same thing.

So there is one rule, and it lives here.

    canonical_entity('people', 'Mark_Philippoussis')  -> 'Mark Philippoussis'
    canonical_entity('breeds', 'basenji dog')         -> 'a basenji dog'
    canonical_entity('scenes', 'abbey')               -> 'an abbey scene'
    generation_prompt('scenes', 'abbey')              -> 'An image of an abbey scene'

**The rule is idempotent**, and that property is load-bearing rather than decorative. Applying it
to a name that has already been through it returns the same string, so a pipeline that transforms a
name and hands it to something that transforms it again still ends with one article and one suffix.
Without idempotence that composition produced ``an an abbey scene scene``.

The article follows pronunciation, not spelling: ``an hour``, ``a university``. English has two
small classes of exception to "a vowel letter takes *an*", and both are listed below as words
rather than as prefixes -- a prefix rule turns ``uninhabited`` into ``a uninhabited``. A word that
is in neither list falls back to the first letter, which is correct for the overwhelming majority.
"""
from __future__ import annotations

import re
from typing import Literal, Tuple

# Mirrors the local alias in ``testbed.py``: this module stays inside ``datasets`` and does not
# import the benchmark sub-package, so the dependency runs one way only.
type_entity_task = Literal['breeds', 'scenes', 'people']

#: Vowel-lettered words English pronounces with a consonant sound, so they take "a".
_A_DESPITE_VOWEL: Tuple[str, ...] = (
    'eucalyptus', 'euphonium', 'european', 'ewe',
    'once', 'one',
    'ufo', 'ukulele', 'unicorn', 'uniform', 'union', 'unit', 'united', 'universe', 'university',
    'usage', 'use', 'used', 'user', 'usual', 'utensil', 'utility', 'utopia',
)

#: Consonant-lettered words English pronounces with a vowel sound, so they take "an".
_AN_DESPITE_CONSONANT: Tuple[str, ...] = (
    'heir', 'heirloom', 'honest', 'honor', 'honorable', 'honour', 'honourable', 'hour', 'hourglass',
)

_SUBSTITUTE_CONCEPT = {
    'people': 'a child',
    'breeds': 'a cat',
    'scenes': 'the moon',
}

_SCENE_SUFFIX = ' scene'
_LEADING_ARTICLE_RE = re.compile(r'^[Aa]n? ')


def article_for(text: str) -> Literal['a', 'an']:
    """Return the English indefinite article for *text*, by pronunciation of its first word."""
    cleaned = _clean(text)
    if not cleaned:
        raise ValueError('cannot choose an article for an empty name')
    first_word = cleaned.split(' ')[0].lower()
    if first_word in _AN_DESPITE_CONSONANT:
        return 'an'
    if first_word in _A_DESPITE_VOWEL:
        return 'a'
    return 'an' if first_word[0] in 'aeiou' else 'a'


def canonical_entity(task: type_entity_task, name: str) -> str:
    """Return the one form of *name* every string in the benchmark is built from.

    People keep their name with spaces instead of underscores; breeds gain an article; scenes gain
    an article and a trailing ``scene``. Applying this to its own output changes nothing.

    @param task: which task the entity belongs to.
    @param name: the entity as the task metadata spells it, or the canonical form itself.
    @raises ValueError: the name is empty once underscores and whitespace are normalised.
    @raises NotImplementedError: the task has no naming rule.
    """
    cleaned = _clean(name)
    if not cleaned:
        raise ValueError(f'entity name is empty after normalisation: {name!r}')

    if task == 'people':
        return cleaned
    if task == 'breeds':
        return _with_article(cleaned)
    if task == 'scenes':
        return _with_article(_with_scene_suffix(cleaned))
    raise NotImplementedError(f'no naming rule for task {task!r}')


def generation_prompt(task: type_entity_task, name: str) -> str:
    """Return the prompt the benchmark generates and scores this entity's images with."""
    return f'An image of {canonical_entity(task, name)}'


def substitute_concept(task: type_entity_task) -> str:
    """Return the concept this task's unlearned entities are pushed towards."""
    try:
        return _SUBSTITUTE_CONCEPT[task]
    except KeyError:
        raise NotImplementedError(f'no substitute concept for task {task!r}')


def display_entity(raw_name: str, max_chars: int = 24) -> str:
    """Return *raw_name* without its leading article, truncated for a plot label.

    Examples::

        display_entity('a bouvier des flandres dog')  # 'bouvier des flandres dog'
        display_entity('An ice skating rink')         # 'ice skating rink'
        display_entity('George W. Bush')              # 'George W. Bush'
    """
    name = _LEADING_ARTICLE_RE.sub('', raw_name)
    if len(name) > max_chars:
        name = name[:max_chars - 1] + '…'
    return name


def _clean(name: str) -> str:
    return re.sub(r'\s+', ' ', name.replace('_', ' ')).strip()


def _strip_article(text: str) -> str:
    return _LEADING_ARTICLE_RE.sub('', text)


def _with_article(text: str) -> str:
    body = _strip_article(text)
    return f'{article_for(body)} {body}'


def _with_scene_suffix(text: str) -> str:
    body = _strip_article(text)
    if body.lower().endswith(_SCENE_SUFFIX):
        return body
    return f'{body}{_SCENE_SUFFIX}'
