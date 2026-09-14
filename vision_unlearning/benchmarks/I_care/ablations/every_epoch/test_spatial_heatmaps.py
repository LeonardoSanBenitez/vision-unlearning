"""Contract tests for the spatial-difference analysis.

The rendered heatmap grid looks plausible under every one of the mistakes that matter: an entity paired
with a neighbour's baseline, a difference taken the wrong way round, a concentration statistic that
measures magnitude instead of localisation, or a column order that ignores its own sort key. These tests
pin the three pure functions those mistakes would go through.

Not collected by the repository's pytest configuration, which is scoped to tests/; run directly:
    python -m pytest vision_unlearning/benchmarks/I_care/ablations/every_epoch/test_spatial_heatmaps.py
"""
from __future__ import annotations

import numpy as np

from spatial_heatmaps import column_order_by_change, concentration, difference_map


def _image(value: float, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float64)


def test_difference_map_is_absolute_and_channel_averaged() -> None:
    on = _image(30.0)
    off = _image(10.0)
    assert np.allclose(difference_map(on, off), 20.0)
    # Reversing the arguments must not change the map: it is a distance, not a signed difference, so a
    # brightening and a darkening of the same size count the same.
    assert np.allclose(difference_map(off, on), 20.0)


def test_difference_map_averages_over_channels_rather_than_taking_one() -> None:
    on = np.zeros((4, 4, 3), dtype=np.float64)
    on[:, :, 0] = 30.0  # only the red channel moves
    off = np.zeros((4, 4, 3), dtype=np.float64)
    assert np.allclose(difference_map(on, off), 10.0)


def test_difference_map_rejects_mismatched_shapes() -> None:
    try:
        difference_map(_image(1.0, size=8), _image(1.0, size=4))
    except AssertionError:
        return
    raise AssertionError("a difference between images of different sizes must not be computed silently")


def test_concentration_of_a_uniform_change_is_the_top_fraction() -> None:
    assert concentration(np.full((10, 10), 5.0)) == 0.1


def test_concentration_of_a_point_source_is_one() -> None:
    change = np.zeros((10, 10))
    change[3, 4] = 99.0
    assert concentration(change) == 1.0


def test_concentration_ignores_magnitude() -> None:
    change = np.zeros((10, 10))
    change[0, :5] = 1.0
    faint = concentration(change)
    assert concentration(change * 1000.0) == faint


def test_concentration_of_an_unchanged_image_is_the_uniform_value() -> None:
    assert concentration(np.zeros((10, 10))) == 0.1


def test_column_order_pins_the_target_and_sorts_receivers_by_decreasing_change() -> None:
    order = column_order_by_change(
        last_epoch_change={0: 1.0, 1: 9.0, 2: 5.0, 3: 7.0},
        names=["target", "b", "c", "d"], target_index=0,
    )
    assert order == [0, 1, 3, 2]


def test_column_order_breaks_ties_on_the_entity_name() -> None:
    order = column_order_by_change(
        last_epoch_change={0: 1.0, 1: 4.0, 2: 4.0},
        names=["target", "zebra", "aardvark"], target_index=0,
    )
    assert order == [0, 2, 1]


def test_column_order_keeps_the_target_first_even_when_it_changed_most() -> None:
    order = column_order_by_change(
        last_epoch_change={0: 99.0, 1: 4.0, 2: 8.0},
        names=["target", "b", "c"], target_index=0,
    )
    assert order[0] == 0 and order[1] == 2
