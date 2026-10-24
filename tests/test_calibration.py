"""Equalization's calibration read back from read-out files: F per setting, the means, the selection.

Fails on the code before this change: ``calibration`` does not exist, so nothing turned the read-outs of
the three ladders into the means the selector compares, and nothing asked for the edge values.
"""
import json
import math
import os
from typing import Any, Callable, Dict

import pytest

from vision_unlearning.benchmarks.I_care import calibration as cal
from vision_unlearning.benchmarks.I_care.equalization import IntegrityFailure, draw_panels, ladder
from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_pair_path, per_seed_path_of

TASK = 'people'
ENTITIES = [f'Entity_{k:02d}' for k in range(40)]
SEEDS = 4


def _write(root: str, method: str, value: float, effect: float, receiver_diff: float = -0.1) -> None:
    """The five calibration entities' read-outs at one dial: the target loses `effect` on average."""
    base = cal.readout_assets(root, TASK, method, value)
    for target in draw_panels(TASK, ENTITIES)['calibration']:
        index = ENTITIES.index(target)
        receivers = cal.readout_receivers(TASK, ENTITIES, target)
        per_pair = get_interference_per_pair_path(TASK, index, method, cal.epochs_of(method, value), base_folder=base)  # type: ignore[arg-type]
        os.makedirs(os.path.dirname(per_pair), exist_ok=True)
        rows = {r: {'clip_diff': (-effect if r == target else receiver_diff)} for r in receivers}
        with open(per_pair, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle)
        seeds = {r: {'clip_off': [30.0] * SEEDS, 'clip_on': [30.0 - (effect if r == target else 0.0)] * SEEDS} for r in receivers}
        with open(per_seed_path_of(per_pair), 'w', encoding='utf-8') as handle:
            json.dump({'receivers': seeds}, handle)


def _ladders(root: str, effect_of: Callable[[str, int], float]) -> None:
    for method in ('uce', 'distil', 'salun'):
        for k, value in enumerate(ladder(TASK, method)):
            _write(root, method, value, effect_of(method, k))


def test_the_readout_is_the_target_and_the_twenty_receivers_in_task_order() -> None:
    target = draw_panels(TASK, ENTITIES)['calibration'][0]
    receivers = cal.readout_receivers(TASK, ENTITIES, target)
    assert target in receivers and len(receivers) == 21
    assert receivers == sorted(receivers, key=ENTITIES.index)


def test_dial_names_round_trip() -> None:
    assert [cal.dial_name(v) for v in (400.0, 23.4375, 17.96875, 96000.0)] == ['400', '23.4375', '17.96875', '96000']


def test_an_incomplete_ladder_reports_its_means_and_selects_nothing(tmp_path: Any) -> None:
    root = str(tmp_path)
    _write(root, 'distil', 400, 6.0)
    result = cal.collect(root, TASK, ENTITIES, f_min=2.0)
    assert result['methods']['distil']['values']['400']['mean'] == pytest.approx(6.0)
    assert result['methods']['distil']['values']['50']['mean'] is None
    assert result['selection'] is None and not result['methods']['uce']['complete']


def test_complete_ladders_matched_in_the_middle_are_final_without_an_edge_pass(tmp_path: Any) -> None:
    root = str(tmp_path)
    # Every method saturates at 6 before the top of its ladder; ties go to the lower dial, so mid-ladder.
    _ladders(root, lambda method, k: min(float(k), 6.0) if method == 'uce' else min(2.0 * k, 6.0))
    result = cal.collect(root, TASK, ENTITIES, f_min=2.0)
    assert result['extensions'] == {} and result['final'] is True
    assert result['selection']['label'] == 'matched'
    retain = result['methods']['distil']['values']['400']['retain']
    assert retain['receivers'] == 100.0 and retain['mean_receiver_clip_diff'] == pytest.approx(-0.1)
    assert 'damaged' not in result['methods']['uce']['values']['1500']['retain']


def test_a_selection_at_a_ladder_top_asks_for_the_one_edge_value_then_is_final(tmp_path: Any) -> None:
    root = str(tmp_path)
    # SPARE only reaches the floor at its top rung, so the first selection lands on 3200 epochs.
    _ladders(root, lambda method, k: (0.5 if k < 6 else 2.5) if method == 'distil'
             else min(0.4 * k, 2.4) if method == 'uce' else min(0.5 * k, 2.4))
    first = cal.collect(root, TASK, ENTITIES, f_min=2.0)
    assert first['extensions'] == {'distil': 6400} and first['final'] is False
    _write(root, 'distil', 6400, 2.45)
    final = cal.collect(root, TASK, ENTITIES, f_min=2.0)
    assert final['final'] is True and final['selection']['dials']['distil'] == 6400
    assert final['selection']['at_search_boundary']['distil'] is True
    assert final['methods']['distil']['values']['6400']['on_ladder'] is False


def test_a_non_finite_measurement_is_an_integrity_failure(tmp_path: Any) -> None:
    root = str(tmp_path)
    _ladders(root, lambda method, k: float(k))
    _write(root, 'salun', ladder(TASK, 'salun')[2], math.nan)
    with pytest.raises(IntegrityFailure):
        cal.collect(root, TASK, ENTITIES, f_min=2.0)
