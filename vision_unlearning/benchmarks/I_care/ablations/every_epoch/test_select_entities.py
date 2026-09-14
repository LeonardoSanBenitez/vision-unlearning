"""Unit tests for the every-epoch SPARE entity-selection logic.

These exercise the pure selection functions on synthetic fixtures only -- no
GPU, no torch, no data files. Run manually (the ablation is outside the repo's
``tests/`` pytest scope):

    python -m pytest vision_unlearning/benchmarks/I_care/ablations/every_epoch/test_select_entities.py
"""
from __future__ import annotations

from typing import Dict

import pytest

from select_entities import (
    GROUP_LITTLE_HIGH,
    GROUP_LITTLE_LOW,
    GROUP_MEDIAN,
    GROUP_SIMILAR_HIGH,
    GROUP_SIMILAR_LOW,
    choose_target,
    interference_pools,
    select_receiver_groups,
    sorted_by_interference,
    target_gates,
    target_interference_score,
    two_most_two_least_similar,
)


def _linear_fixture() -> Dict[str, Dict[str, float]]:
    """20 receivers r00..r19 with strictly monotonic interference (r00 most
    interfered) and hand-placed similarity so the extreme groups are predictable.

    Returns a dict with 'clip_diff' and 'similarity' maps.
    """
    names = [f"r{i:02d}" for i in range(20)]
    clip_diff = {name: -20.0 + 2.0 * i for i, name in enumerate(names)}  # r00=-20 ... r19=+18
    similarity = {name: 65.0 for name in names}  # middle receivers near the median
    # top interference pool (r00..r03): r00,r01 most similar; r02,r03 least
    similarity["r00"], similarity["r01"], similarity["r02"], similarity["r03"] = 90.0, 88.0, 50.0, 48.0
    # bottom interference pool (r16..r19): r16,r17 most similar; r18,r19 least
    similarity["r16"], similarity["r17"], similarity["r18"], similarity["r19"] = 85.0, 83.0, 45.0, 43.0
    return {"clip_diff": clip_diff, "similarity": similarity}


def test_top_pool_is_absolutely_most_interfered() -> None:
    """The 'high interference' groups must be the absolute most-interfered
    receivers -- not merely high relative to a similarity pool."""
    fx = _linear_fixture()
    top_pool, bottom_pool = interference_pools(fx["clip_diff"])
    assert set(top_pool) == {"r00", "r01", "r02", "r03"}
    assert set(bottom_pool) == {"r16", "r17", "r18", "r19"}
    # Every top-pool receiver is more interfered (more negative) than every other receiver.
    max_top = max(fx["clip_diff"][r] for r in top_pool)
    rest = [fx["clip_diff"][r] for r in fx["clip_diff"] if r not in top_pool]
    assert max_top < min(rest)


def test_two_most_two_least_similar_within_pool() -> None:
    fx = _linear_fixture()
    top_pool, _ = interference_pools(fx["clip_diff"])
    most, least = two_most_two_least_similar(top_pool, fx["similarity"])
    assert set(most) == {"r00", "r01"}
    assert set(least) == {"r02", "r03"}


def test_select_receiver_groups_shape_and_membership() -> None:
    """All nine selected receivers are distinct; each of the four groups has two
    members and the median has one; the high-interference groups are the top pool."""
    fx = _linear_fixture()
    labels = select_receiver_groups(fx["clip_diff"], fx["similarity"])
    assert len(labels) == 9
    assert len(set(labels)) == 9  # distinct receiver names (dict keys already unique)

    counts: Dict[str, int] = {}
    for group in labels.values():
        counts[group] = counts.get(group, 0) + 1
    assert counts[GROUP_SIMILAR_HIGH] == 2
    assert counts[GROUP_LITTLE_HIGH] == 2
    assert counts[GROUP_SIMILAR_LOW] == 2
    assert counts[GROUP_LITTLE_LOW] == 2
    assert counts[GROUP_MEDIAN] == 1

    high = {r for r, g in labels.items() if g in (GROUP_SIMILAR_HIGH, GROUP_LITTLE_HIGH)}
    assert high == {"r00", "r01", "r02", "r03"}
    low = {r for r, g in labels.items() if g in (GROUP_SIMILAR_LOW, GROUP_LITTLE_LOW)}
    assert low == {"r16", "r17", "r18", "r19"}
    assert labels["r00"] == GROUP_SIMILAR_HIGH and labels["r02"] == GROUP_LITTLE_HIGH
    assert labels["r16"] == GROUP_SIMILAR_LOW and labels["r18"] == GROUP_LITTLE_LOW

    median = [r for r, g in labels.items() if g == GROUP_MEDIAN][0]
    assert median not in {"r00", "r01", "r02", "r03", "r16", "r17", "r18", "r19"}


def test_name_tie_break_is_deterministic() -> None:
    """Receivers with equal interference are ordered by name ascending."""
    clip_diff = {"b": -5.0, "a": -5.0, "c": -5.0}
    assert sorted_by_interference(clip_diff) == ["a", "b", "c"]


def test_forgotten_gate_rejects_weak_target() -> None:
    receivers = {f"r{i}": -8.0 + i for i in range(10)}  # varied, at least one untouched
    forgotten, not_collapsed = target_gates(self_clip_diff=-1.0, receiver_clip_diffs=receivers, floor=-5.0)
    assert forgotten is False
    assert not_collapsed is True  # some receiver is above -2.0
    strongly = {f"r{i}": -30.0 - i for i in range(10)}
    forgotten2, _ = target_gates(self_clip_diff=-9.0, receiver_clip_diffs=strongly, floor=-5.0)
    assert forgotten2 is True


def test_collapse_gate_rejects_noise_collapse() -> None:
    """A target whose every receiver is strongly damaged is a broken model, not
    selective forgetting -- rejected by the not-collapsed gate."""
    all_damaged = {f"r{i}": -10.0 - i for i in range(10)}  # max is -10, below -2.0
    forgotten, not_collapsed = target_gates(self_clip_diff=-20.0, receiver_clip_diffs=all_damaged, floor=-5.0)
    assert forgotten is True
    assert not_collapsed is False


def test_target_interference_score_uses_most_interfered() -> None:
    """Score is the mean of the top-fraction most-interfered (most negative)."""
    receivers = {f"r{i}": float(i) for i in range(100)}  # r0..r99 = 0..99
    receivers["r0"] = -50.0
    receivers["r1"] = -40.0
    score = target_interference_score(receivers, fraction=0.10)  # top 10 most negative
    # The 10 most-interfered are r0(-50), r1(-40), then r2..r9 (2..9).
    expected = (-50.0 - 40.0 + sum(range(2, 10))) / 10.0
    assert score == pytest.approx(expected)


def test_choose_target_picks_strongest_gated() -> None:
    candidates = [
        {"name": "weak", "index": 0, "is_gated": True, "score": -3.0, "self_clip_diff": -6.0},
        {"name": "strong", "index": 1, "is_gated": True, "score": -9.0, "self_clip_diff": -7.0},
        {"name": "ungated", "index": 2, "is_gated": False, "score": -99.0, "self_clip_diff": -1.0},
    ]
    chosen, top5 = choose_target(candidates)
    assert chosen["name"] == "strong"  # most negative score among gated; ungated excluded
    assert [c["name"] for c in top5] == ["strong", "weak"]


def test_choose_target_raises_when_none_gated() -> None:
    candidates = [{"name": "x", "index": 0, "is_gated": False, "score": -1.0, "self_clip_diff": -1.0}]
    with pytest.raises(ValueError):
        choose_target(candidates)
