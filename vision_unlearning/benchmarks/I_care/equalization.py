"""Equalization: bringing the three unlearning methods to a comparable strength, by a fixed rule.

Within a task, I-CARE compares methods at approximately the same mean reduction of the target's
own-prompt CLIP score. For method ``m`` at dial value ``h``, target ``i`` and seed ``s``::

    F[m, h, i] = mean_s(CLIP_off[i, s] - CLIP_on[m, h, i, s])

positive meaning forgetting. (The stored per-pair ``clip_diff`` is ``on - off``; it is negated only
here.) ``F_m(h)`` is that quantity averaged over the task's calibration entities. This is a
calibrated operating point for these configurations and entities. It does not assert equal erasure
per entity, equal substitution quality, or equality over training randomness, and retain-side
damage is an outcome to report, never something tuned.

Everything that decides the selection is fixed here, before any calibration session runs:

* **the ladders** -- each method's one dial, swept over a fixed list of values
  (:func:`scale_ladder` for UCE's ``erase_scale``, :func:`epoch_ladder` for the trained methods);
* **the scale reference** ``f_min`` -- half the task's mean baseline specificity over all its
  entities and seeds (:func:`scale_reference`), where an entity's specificity is its own-prompt CLIP
  score minus its mean score against the task's other prompts. It fixes the size of effect the
  comparison is made at; it does not certify erasure;
* **the screen** -- an engineering check that the scorer recognises most of the task's entities
  (:func:`competence_screen`); it informs, it never excludes an entity;
* **the selector** -- :func:`select`, deterministic, with six branches and its tie-breaks;
* **the one edge pass** -- :func:`edge_extensions` adds one value beyond each ladder end the first
  selection landed on, all methods at once; :func:`select` runs once more and that result is final;
* **confirmation** -- :func:`confirm` re-reads the frozen settings on held-out entities and reports
  relative matching there, without ever changing the calibration label.

A missing or non-finite measurement anywhere is an integrity failure (:class:`IntegrityFailure`),
never a weak result to drop.

The code names follow ``configuration.type_unlearning_algorithm``: ``'uce'``, ``'distil'`` (SPARE)
and ``'salun'``, and that is also the order in which dial values break ties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

# The method order every tie-break uses: lower dial values first, compared in this order.
METHOD_ORDER: Tuple[str, ...] = ('uce', 'distil', 'salun')

# A set of methods qualifies as matched when every member reaches the floor and the spread of their
# effects is at most this fraction of the smallest one.
RELATIVE_TOLERANCE = 0.2

# The screen: an entity is recognised when its own prompt is among CLIP's TOP_K highest-scoring
# prompts in at least MIN_SEEDS of the seeds; the task passes when at least MIN_ENTITIES are.
SCREEN_TOP_K = 5
SCREEN_MIN_SEEDS = 3
SCREEN_MIN_ENTITIES = 80

Label = Literal['matched', 'partially matched', 'unmatched', 'no method reaches the floor']


class IntegrityFailure(ValueError):
    """A measurement the selection needs is missing or not finite. Diagnose it; never drop it."""


# -- the ladders ---------------------------------------------------------------------------------


def round_half_up(value: float) -> int:
    """Round to the nearest integer, halves upwards (Python's round() rounds halves to even)."""
    return int(math.floor(value + 0.5))


def scale_ladder(centre: float, low: int = -6, high: int = 6) -> List[float]:
    """``centre * 2**k`` for ``k = low .. high``, not rounded: a closed-form dial costs nothing extra."""
    return [centre * 2.0 ** k for k in range(low, high + 1)]


def epoch_ladder(centre: int, low: int = -3, high: int = 3) -> List[int]:
    """``centre * 2**k`` for ``k = low .. high``, rounded half up and at least 1, duplicates removed."""
    values: List[int] = []
    for k in range(low, high + 1):
        value = max(1, round_half_up(centre * 2.0 ** k))
        if value not in values:
            values.append(value)
    return values


def edge_extensions(
    ladders: Mapping[str, Sequence[float]],
    selected: Mapping[str, float],
    integer_dial: Mapping[str, bool],
) -> Dict[str, float]:
    """The one value added beyond each ladder end the first selection landed on.

    For every method whose selected dial is its ladder's lowest value, half of it; highest, double.
    Integer dials (epochs) are rounded half up and kept at least 1, and a value that repeats one
    already on the ladder is skipped. Every such method is extended at once, and only once.
    """
    extensions: Dict[str, float] = {}
    for method, value in selected.items():
        ladder = sorted(ladders[method])
        if value == ladder[0]:
            candidate = value / 2
        elif value == ladder[-1]:
            candidate = value * 2
        else:
            continue
        if integer_dial[method]:
            candidate = max(1, round_half_up(candidate))
        if candidate not in ladder:
            extensions[method] = candidate
    return extensions


# -- the quantities -------------------------------------------------------------------------------


def _finite(value: Optional[float], what: str) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IntegrityFailure(f'{what} is {value!r}; a missing or non-finite measurement must be diagnosed, not dropped')
    return float(value)


def forget_effect(clip_off: Sequence[float], clip_on: Sequence[float]) -> float:
    """``F`` for one target: the mean over seeds of its own-prompt CLIP score lost, ``off - on``."""
    if len(clip_off) != len(clip_on) or not clip_off:
        raise IntegrityFailure(f'{len(clip_off)} baseline and {len(clip_on)} unlearned scores for one target')
    differences = [_finite(off, 'a baseline CLIP score') - _finite(on, 'an unlearned CLIP score')
                   for off, on in zip(clip_off, clip_on)]
    return sum(differences) / len(differences)


def specificity(own_score: float, other_scores: Sequence[float]) -> float:
    """An entity's own-prompt CLIP score minus its mean score against the task's other prompts."""
    if not other_scores:
        raise IntegrityFailure('no other prompts to compare the own-prompt score with')
    others = [_finite(score, 'a CLIP score against another prompt') for score in other_scores]
    return _finite(own_score, 'an own-prompt CLIP score') - sum(others) / len(others)


def scale_reference(specificities: Sequence[float]) -> float:
    """``f_min``: half the mean baseline specificity over every entity and seed of the task.

    Raises :class:`IntegrityFailure` when it is not finite and strictly positive, because the
    selector is then undefined or vacuous and equalization must not start for that task.
    """
    if not specificities:
        raise IntegrityFailure('no specificities to take the scale reference from')
    values = [_finite(value, 'a specificity') for value in specificities]
    f_min = 0.5 * sum(values) / len(values)
    if not (math.isfinite(f_min) and f_min > 0):
        raise IntegrityFailure(f'the scale reference is {f_min}; it must be finite and strictly positive')
    return f_min


def competence_screen(own_prompt_ranks: Mapping[str, Sequence[int]]) -> Dict[str, object]:
    """How many entities the scorer recognises: own prompt in the top 5 of the task's prompts in 3 of 4 seeds.

    ``own_prompt_ranks`` maps each entity to its own prompt's rank (1 is best) per seed. Returns the
    count, the threshold, whether the task passes, and the entities that do not. It never excludes
    an entity: a task below the threshold is investigated, and if no defect is found its baseline is
    valid but weak, reported as such.
    """
    recognised = [
        entity for entity, ranks in own_prompt_ranks.items()
        if sum(1 for rank in ranks if rank <= SCREEN_TOP_K) >= SCREEN_MIN_SEEDS
    ]
    return {
        'recognised': len(recognised),
        'entities': len(own_prompt_ranks),
        'threshold': SCREEN_MIN_ENTITIES,
        'passes': len(recognised) >= SCREEN_MIN_ENTITIES,
        'not_recognised': sorted(set(own_prompt_ranks) - set(recognised)),
    }


# -- the selector ---------------------------------------------------------------------------------


def qualifies(effects: Sequence[float], f_min: float) -> bool:
    """Every effect reaches ``f_min`` and the spread is at most 20 % of the smallest effect."""
    low, high = min(effects), max(effects)
    return low >= f_min and (high - low) <= RELATIVE_TOLERANCE * low * (1 + 1e-12)


@dataclass(frozen=True)
class Selection:
    """The selected dial value per method, its effect, the branch label, and what it was chosen from."""
    dials: Dict[str, float]
    effects: Dict[str, float]
    label: Label
    reaching: Tuple[str, ...]
    f_min: float
    at_search_boundary: Dict[str, bool] = field(default_factory=dict)

    @property
    def gaps(self) -> Dict[str, float]:
        """Each method's effect minus the smallest selected effect."""
        low = min(self.effects.values())
        return {method: effect - low for method, effect in self.effects.items()}


def _largest(ladder: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """The largest effect, ties broken by the lower dial."""
    return min(ladder, key=lambda pair: (-pair[1], pair[0]))


def _smallest(ladder: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """The smallest effect, ties broken by the lower dial."""
    return min(ladder, key=lambda pair: (pair[1], pair[0]))


def select(ladders: Mapping[str, Mapping[float, Optional[float]]], f_min: float) -> Selection:
    """Choose one dial value per method by the fixed rule, and label the result.

    ``ladders[method][dial] = F_m(dial)``, the calibration-panel mean effect at that dial value.
    ``R`` is the set of methods with some effect of at least ``f_min``; a set of methods qualifies
    when it satisfies :func:`qualifies`.

    =====================================  =============================================  ==================
    ``R``                                  choice                                         label
    =====================================  =============================================  ==================
    3, a qualifying triplet exists         largest minimum F; ties: smaller spread, dials  matched
    3, none qualifies                      smallest (max-min)/min among triplets >= f_min;  unmatched
                                           ties: larger minimum, dials
    2, a qualifying pair exists            that pair as matched; the third at its largest  partially matched
    2, none qualifies                      that pair as unmatched; the third at largest    unmatched
    1                                      it at its smallest F >= f_min; others largest   unmatched
    none                                   every method at its largest F                   no method reaches the floor
    =====================================  =============================================  ==================

    Dials break the remaining ties, lower first, compared in :data:`METHOD_ORDER`. A method's largest
    and smallest effects break their own ties by the lower dial. A missing or non-finite effect
    raises :class:`IntegrityFailure`.
    """
    f_min = _finite(f_min, 'the scale reference')
    methods = [m for m in METHOD_ORDER if m in ladders] + sorted(set(ladders) - set(METHOD_ORDER))
    pairs: Dict[str, List[Tuple[float, float]]] = {}
    for method in methods:
        if not ladders[method]:
            raise IntegrityFailure(f'{method} has no measured ladder value')
        pairs[method] = [(float(dial), _finite(effect, f'F for {method} at {dial}'))
                         for dial, effect in sorted(ladders[method].items())]

    eligible = {m: [p for p in pairs[m] if p[1] >= f_min] for m in methods}
    reaching = tuple(m for m in methods if eligible[m])
    chosen = {m: _largest(pairs[m]) for m in methods}
    label: Label = 'no method reaches the floor'

    if len(reaching) == 1:
        chosen[reaching[0]] = _smallest(eligible[reaching[0]])
        label = 'unmatched'
    elif len(reaching) >= 2:
        candidates = list(product(*(eligible[m] for m in reaching)))
        matching = [c for c in candidates if qualifies([p[1] for p in c], f_min)]

        def matched_order(candidate: Tuple[Tuple[float, float], ...]) -> Tuple[float, ...]:
            effects = [p[1] for p in candidate]
            return (-min(effects), max(effects) - min(effects)) + tuple(p[0] for p in candidate)

        def unmatched_order(candidate: Tuple[Tuple[float, float], ...]) -> Tuple[float, ...]:
            effects = [p[1] for p in candidate]
            return ((max(effects) - min(effects)) / min(effects), -min(effects)) + tuple(p[0] for p in candidate)

        best = min(matching, key=matched_order) if matching else min(candidates, key=unmatched_order)
        for method, pair in zip(reaching, best):
            chosen[method] = pair
        if matching:
            label = 'matched' if len(reaching) == len(methods) else 'partially matched'
        else:
            label = 'unmatched'

    return Selection(
        dials={m: chosen[m][0] for m in methods},
        effects={m: chosen[m][1] for m in methods},
        label=label,
        reaching=reaching,
        f_min=f_min,
    )


def final_selection(
    initial_ladders: Mapping[str, Mapping[float, Optional[float]]],
    extended_ladders: Mapping[str, Mapping[float, Optional[float]]],
    f_min: float,
) -> Selection:
    """The final selection: :func:`select` on the ladders after the one edge pass.

    ``extended_ladders`` holds every value of ``initial_ladders`` plus the measured
    :func:`edge_extensions`. Nothing is extended again: a final dial at its extended ladder's edge
    is marked ``at_search_boundary``.
    """
    for method, ladder in initial_ladders.items():
        missing = set(ladder) - set(extended_ladders.get(method, {}))
        if missing:
            raise IntegrityFailure(f'the extended ladder of {method} lost the values {sorted(missing)}')
    selection = select(extended_ladders, f_min)
    boundary = {}
    for method, dial in selection.dials.items():
        values = sorted(extended_ladders[method])
        boundary[method] = dial in (values[0], values[-1])
    return Selection(selection.dials, selection.effects, selection.label, selection.reaching,
                     selection.f_min, boundary)


# -- confirmation ---------------------------------------------------------------------------------


def confirm(calibration: Selection, confirmation_effects: Mapping[str, Optional[float]]) -> Dict[str, object]:
    """Relative matching at the frozen settings on the held-out confirmation entities.

    Reports the scale reference, the confirmation means, their gaps, whether they qualify among
    the methods that reached the floor on calibration (the relative verdict), the level shift
    against the calibration means, and whether the confirmation spread stays within the
    calibration's frozen tolerance (20 % of the calibration's smallest selected effect). Passing
    confirms relative matching on held-out entities, not the same absolute operating point.

    It never changes the calibration label. A task matched on calibration that fails here gains
    the note ``calibration-only matching``; the campaign continues at the frozen settings either way.
    """
    effects = {m: _finite(confirmation_effects.get(m), f'the confirmation mean of {m}') for m in calibration.effects}
    compared = list(calibration.reaching) or list(effects)
    values = [effects[m] for m in compared]
    relative = qualifies(values, calibration.f_min)
    calibration_low = min(calibration.effects[m] for m in compared)
    frozen = (max(values) - min(values)) <= RELATIVE_TOLERANCE * calibration_low * (1 + 1e-12)
    low = min(effects.values())
    notes: List[str] = []
    if calibration.label == 'matched' and not relative:
        notes.append('calibration-only matching')
    return {
        'f_min': calibration.f_min,
        'means': effects,
        'gaps': {m: value - low for m, value in effects.items()},
        'relative_verdict': relative,
        'level_shift': {m: effects[m] - calibration.effects[m] for m in effects},
        'frozen_tolerance_verdict': frozen,
        'calibration_label': calibration.label,
        'notes': notes,
    }


# -- diagnostics, never eligibility rules ---------------------------------------------------------


def retain_diagnostics(receiver_clip_diffs: Sequence[float], route_noise: float) -> Dict[str, float]:
    """The retain side of one setting, reported and never tuned.

    ``receiver_clip_diffs`` are the stored ``clip_diff`` values (``on - off``, negative is damage) of
    the read-out receivers; ``route_noise`` is the output difference of an unedited model between
    the baseline and session routes. A receiver is damaged when its ``clip_diff`` is below
    ``-3 * route_noise``; the bleed rate is the fraction of receivers damaged.
    """
    values = [_finite(value, 'a receiver clip_diff') for value in receiver_clip_diffs]
    threshold = -3 * _finite(route_noise, 'the route noise')
    damaged = sum(1 for value in values if value < threshold)
    return {
        'mean_receiver_clip_diff': sum(values) / len(values),
        'receivers': float(len(values)),
        'damaged': float(damaged),
        'bleed_rate': damaged / len(values),
        'damage_threshold': threshold,
    }


# -- the panels ----------------------------------------------------------------------------------

#: The seed every task's panels are drawn with, fixed before any calibration session runs.
PANEL_SEED = 20261019
CALIBRATION_SIZE = 5
CONFIRMATION_SIZE = 25
READOUT_RECEIVERS = 20


def draw_panels(task: str, entities: Sequence[str], seed: int = PANEL_SEED) -> Dict[str, List[str]]:
    """The task's calibration and confirmation entities and its read-out receivers, drawn at random.

    Drawn from all of the task's entities (in the metadata's order) with a generator seeded to
    ``seed`` and the task name, so the three tasks get independent draws and the draw can be
    repeated by anyone: five calibration entities; 25 confirmation entities from the rest; 20
    read-out receivers from the entities that are not calibration targets. The three methods share
    them. Membership says nothing about an entity's quality: no entity is drawn by a score.
    """
    import random
    if len(set(entities)) != len(entities):
        raise ValueError('the entity list holds a duplicate')
    if len(entities) < CALIBRATION_SIZE + CONFIRMATION_SIZE:
        raise ValueError(f'{len(entities)} entities cannot hold both panels')
    rng = random.Random(f'{seed}:{task}')
    calibration = rng.sample(list(entities), CALIBRATION_SIZE)
    rest = [entity for entity in entities if entity not in calibration]
    confirmation = rng.sample(rest, CONFIRMATION_SIZE)
    receivers = rng.sample(rest, READOUT_RECEIVERS)
    return {'calibration': calibration, 'confirmation': confirmation, 'readout_receivers': receivers}
