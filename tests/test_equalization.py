"""The equalization rule, on synthetic tables whose right answers are known.

Fails on the code before: ``vision_unlearning.benchmarks.I_care.equalization`` did not exist, and
the first campaign's equalization was a hand-match of forget and retain scores on one task.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional

import pytest

from vision_unlearning.benchmarks.I_care.equalization import (
    IntegrityFailure,
    competence_screen,
    confirm,
    edge_extensions,
    epoch_ladder,
    final_selection,
    forget_effect,
    qualifies,
    retain_diagnostics,
    round_half_up,
    scale_ladder,
    scale_reference,
    select,
    specificity,
)


def _single(effects: List[float]) -> Dict[str, Dict[float, Optional[float]]]:
    """One ladder value per method, in the order uce, distil, salun."""
    return {method: {1.0: effect} for method, effect in zip(['uce', 'distil', 'salun'], effects)}


# -- the ladders are exactly the plan's ----------------------------------------------------------


def test_the_uce_ladders() -> None:
    assert scale_ladder(1500) == [23.4375, 46.875, 93.75, 187.5, 375, 750, 1500, 3000, 6000, 12000, 24000, 48000, 96000]
    assert scale_ladder(1150) == [17.96875, 35.9375, 71.875, 143.75, 287.5, 575, 1150, 2300, 4600, 9200, 18400, 36800, 73600]


@pytest.mark.parametrize('centre,expected', [
    (400, [50, 100, 200, 400, 800, 1600, 3200]),   # SPARE, people
    (100, [13, 25, 50, 100, 200, 400, 800]),       # SPARE scenes and breeds; SalUn people
    (25, [3, 6, 13, 25, 50, 100, 200]),            # SalUn scenes and breeds
])
def test_the_epoch_ladders(centre: int, expected: List[int]) -> None:
    assert epoch_ladder(centre) == expected


def test_halves_round_up_not_to_even() -> None:
    assert [round_half_up(x) for x in (0.5, 1.5, 2.5, 12.5, 3.125)] == [1, 2, 3, 13, 3]


# -- the quantities -------------------------------------------------------------------------------


def test_forget_effect_is_baseline_minus_unlearned_averaged_over_seeds() -> None:
    assert forget_effect([30, 32, 34, 36], [20, 20, 20, 20]) == pytest.approx(13.0)
    assert forget_effect([30, 30], [31, 33]) == pytest.approx(-2.0)


def test_specificity_and_the_scale_reference() -> None:
    assert specificity(30, [20, 22, 24]) == pytest.approx(8.0)
    assert scale_reference([8.0, 12.0, 10.0, 10.0]) == pytest.approx(5.0)


@pytest.mark.parametrize('values', [[0.0, 0.0], [-1.0, 0.5], [float('nan'), 2.0], [float('inf'), 1.0], []])
def test_a_scale_reference_that_is_not_finite_and_positive_raises(values: List[float]) -> None:
    with pytest.raises(IntegrityFailure):
        scale_reference(values)


def test_the_screen_counts_and_never_excludes() -> None:
    ranks = {f'e{k}': [1, 2, 3, 9] for k in range(79)}
    ranks.update({'weak': [9, 9, 9, 1], 'edge': [5, 5, 5, 6]})
    result = competence_screen(ranks)
    assert result['recognised'] == 80 and result['passes'] is True
    assert result['not_recognised'] == ['weak']
    del ranks['edge']
    assert competence_screen(ranks)['passes'] is False


# -- the qualify predicate ------------------------------------------------------------------------


def test_the_boundary_qualifies_and_the_old_example_does_not() -> None:
    assert qualifies([10, 11, 12], 5)
    assert not qualifies([8, 9, 10], 5)


def test_three_means_of_one_fail_a_floor_of_five() -> None:
    assert not qualifies([1, 1, 1], 5)


# -- the six branches -----------------------------------------------------------------------------


@pytest.mark.parametrize('effects,label', [
    ([10, 11, 12], 'matched'),
    ([5, 10, 20], 'unmatched'),
    ([5, 6, 1], 'partially matched'),
    ([5, 10, 1], 'unmatched'),
    ([5, 1, 2], 'unmatched'),
    ([1, 2, 3], 'no method reaches the floor'),
])
def test_each_branch(effects: List[float], label: str) -> None:
    assert select(_single(effects), 5).label == label


def test_a_negative_effect_never_counts_as_erasure() -> None:
    selection = select(_single([-20, 6, 6]), 5)
    assert selection.reaching == ('distil', 'salun')
    assert selection.label == 'partially matched'


def test_methods_reaching_only_5_10_and_1_at_floor_5_are_unmatched() -> None:
    assert select(_single([5, 10, 1]), 5).label == 'unmatched'


def test_matched_takes_the_largest_minimum_then_smaller_spread_then_lower_dials() -> None:
    ladders = {
        'uce': {1.0: 10.0, 2.0: 12.0, 3.0: 12.0},
        'distil': {1.0: 11.0, 2.0: 12.0},
        'salun': {1.0: 12.0, 2.0: 13.0},
    }
    selection = select(ladders, 5)
    assert selection.label == 'matched'
    assert selection.effects == {'uce': 12.0, 'distil': 12.0, 'salun': 12.0}
    assert selection.dials == {'uce': 2.0, 'distil': 2.0, 'salun': 1.0}


def test_unmatched_takes_the_smallest_relative_spread_then_the_larger_minimum() -> None:
    ladders = {'uce': {1.0: 5.0, 2.0: 10.0}, 'distil': {1.0: 20.0}, 'salun': {1.0: 40.0}}
    selection = select(ladders, 5)
    assert selection.label == 'unmatched'
    assert selection.dials == {'uce': 2.0, 'distil': 1.0, 'salun': 1.0}


def test_one_method_takes_its_smallest_reaching_effect_lower_dial_first_and_no_monotonicity() -> None:
    selection = select({'uce': {1.0: 9.0, 2.0: 5.0, 3.0: 5.0}, 'distil': {1.0: 1.0}, 'salun': {1.0: 2.0}}, 5)
    assert selection.dials['uce'] == 2.0 and selection.label == 'unmatched'


def test_the_others_sit_at_their_largest_effect() -> None:
    selection = select({'uce': {1.0: 9.0}, 'distil': {1.0: 1.0, 2.0: 3.0, 3.0: 3.0}, 'salun': {1.0: 2.0}}, 5)
    assert selection.dials['distil'] == 2.0


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf')])
def test_a_missing_or_non_finite_effect_is_an_integrity_failure(bad: Optional[float]) -> None:
    with pytest.raises(IntegrityFailure):
        select({'uce': {1.0: 10.0, 2.0: bad}, 'distil': {1.0: 11.0}, 'salun': {1.0: 12.0}}, 5)
    with pytest.raises(IntegrityFailure):
        select(_single([10, 11, 12]), bad)  # type: ignore[arg-type]


# -- the one edge pass ----------------------------------------------------------------------------


def test_every_method_at_an_edge_is_extended_once_and_together() -> None:
    ladders = {'uce': scale_ladder(1500), 'distil': epoch_ladder(100), 'salun': epoch_ladder(25)}
    selected = {'uce': 23.4375, 'distil': 800, 'salun': 25}
    extensions = edge_extensions(ladders, selected, {'uce': False, 'distil': True, 'salun': True})
    assert extensions == {'uce': 23.4375 / 2, 'distil': 1600}


def test_an_extension_that_repeats_a_value_is_skipped() -> None:
    assert edge_extensions({'salun': [1, 2, 4]}, {'salun': 1}, {'salun': True}) == {}
    assert edge_extensions({'salun': [3, 6]}, {'salun': 3}, {'salun': True}) == {'salun': 2}


def test_an_empty_reaching_set_rescued_by_the_extension_is_classified_by_it() -> None:
    initial = {m: {1.0: 1.0, 2.0: 4.0} for m in ('uce', 'distil', 'salun')}
    first = select(initial, 5)
    assert first.label == 'no method reaches the floor'
    extensions = edge_extensions({m: list(v) for m, v in initial.items()}, first.dials, {m: True for m in initial})
    assert extensions == {'uce': 4.0, 'distil': 4.0, 'salun': 4.0}
    extended = {m: {**initial[m], 4.0: 6.0} for m in initial}
    final = final_selection(initial, extended, 5)
    assert final.label == 'matched'
    assert final.at_search_boundary == {'uce': True, 'distil': True, 'salun': True}


def test_the_final_selection_refuses_a_ladder_that_lost_a_value() -> None:
    with pytest.raises(IntegrityFailure, match='lost'):
        final_selection(_single([10, 11, 12]), {'uce': {2.0: 10.0}, 'distil': {1.0: 11.0}, 'salun': {1.0: 12.0}}, 5)


# -- confirmation ---------------------------------------------------------------------------------


def test_confirmation_reports_both_verdicts_and_the_level_shift() -> None:
    calibration = select(_single([10, 11, 12]), 5)
    report = confirm(calibration, {'uce': 20.0, 'distil': 23.0, 'salun': 24.0})
    assert report['relative_verdict'] is True
    assert report['frozen_tolerance_verdict'] is False
    assert report['level_shift'] == {'uce': 10.0, 'distil': 12.0, 'salun': 12.0}
    assert report['gaps'] == {'uce': 0.0, 'distil': 3.0, 'salun': 4.0}
    assert report['calibration_label'] == 'matched' and report['notes'] == []


def test_confirmation_never_changes_the_calibration_label() -> None:
    unmatched = select(_single([5, 10, 20]), 5)
    report = confirm(unmatched, {'uce': 10.0, 'distil': 11.0, 'salun': 12.0})
    assert report['relative_verdict'] is True
    assert report['calibration_label'] == 'unmatched'
    matched = select(_single([10, 11, 12]), 5)
    failed = confirm(matched, {'uce': 6.0, 'distil': 11.0, 'salun': 12.0})
    assert failed['calibration_label'] == 'matched' and failed['notes'] == ['calibration-only matching']


# -- diagnostics ----------------------------------------------------------------------------------


def test_the_retain_diagnostics() -> None:
    result = retain_diagnostics([-0.5, -4.0, 1.0, -3.1], run_to_run_spread=1.0)
    assert result['damaged'] == 2 and result['bleed_rate'] == pytest.approx(0.5)
    assert math.isclose(result['mean_receiver_clip_diff'], -6.6 / 4)


# -- the panels -----------------------------------------------------------------------------------


def test_the_panels_are_disjoint_where_they_must_be_and_repeatable() -> None:
    from vision_unlearning.benchmarks.I_care.equalization import draw_panels
    entities = [f'entity {k}' for k in range(100)]
    panels = draw_panels('people', entities)
    assert len(panels['calibration']) == 5 and len(panels['confirmation']) == 25 and len(panels['readout_receivers']) == 20
    assert not set(panels['calibration']) & set(panels['confirmation'])
    assert not set(panels['calibration']) & set(panels['readout_receivers'])
    assert draw_panels('people', entities) == panels
    assert draw_panels('scenes', entities) != panels


def test_a_duplicate_entity_is_refused() -> None:
    from vision_unlearning.benchmarks.I_care.equalization import draw_panels
    with pytest.raises(ValueError, match='duplicate'):
        draw_panels('people', ['a', 'a'] + [str(k) for k in range(40)])
