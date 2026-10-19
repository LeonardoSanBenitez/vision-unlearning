"""The per-seed values every per-pair mean is computed from, stored beside it.

Fails on the code before: only the means over seeds were written, so leave-one-seed-out, reliability
and variance analyses had nothing to read and would have to rescore every image.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from vision_unlearning.benchmarks.I_care.metadata import (
    PER_PAIR_METRICS,
    get_interference_per_pair_path,
    means_from_per_seed,
    per_seed_path_of,
    save_interference_per_pair,
    save_interference_per_pair_per_seed,
)

PER_SEED = {
    'Alice': {'brisque_diff': [1.0, 2.0, 3.0, 6.0], 'clip_diff': [-1.0, -2.0, 0.5, 0.5], 'rmse': [0.1, 0.2, 0.3, 0.4],
              'ssim': [0.9, 0.8, 0.7, 0.6], 'dino_diff': [0.5, 0.5, 0.5, 0.9],
              'clip_off': [30.0, 31.0, 32.0, 33.0], 'clip_on': [29.0, 29.0, 32.5, 33.5]},
}


def test_the_stored_means_recompute_from_the_stored_seed_values(tmp_path: Any) -> None:
    base = str(tmp_path)
    os.makedirs(os.path.join(base, 'datasets'))
    means = means_from_per_seed(PER_SEED)
    save_interference_per_pair(means, 'people', 0, 'uce', 0, base_folder=base)
    per_pair_path = get_interference_per_pair_path('people', 0, 'uce', 0, base_folder=base)
    save_interference_per_pair_per_seed(PER_SEED, [42, 43, 44, 45], per_pair_path)

    with open(per_pair_path, encoding='utf-8') as handle:
        stored_means = json.load(handle)
    with open(per_seed_path_of(per_pair_path), encoding='utf-8') as handle:
        stored = json.load(handle)
    assert stored['seeds'] == [42, 43, 44, 45]
    recomputed = means_from_per_seed(stored['receivers'])
    assert recomputed == stored_means
    assert stored_means['Alice'] == pytest.approx({'brisque_diff': 3.0, 'clip_diff': -0.5, 'rmse': 0.25, 'ssim': 0.75, 'dino_diff': 0.6})


def test_the_per_pair_file_keeps_exactly_its_fields() -> None:
    """Readers of the per-pair file see the five metrics and nothing else, as before."""
    assert set(means_from_per_seed(PER_SEED)['Alice']) == set(PER_PAIR_METRICS)


def test_the_clip_components_reproduce_clip_diff() -> None:
    values = PER_SEED['Alice']
    assert [on - off for on, off in zip(values['clip_on'], values['clip_off'])] == values['clip_diff']


def test_unequal_seed_counts_are_refused() -> None:
    broken = {'Alice': {**PER_SEED['Alice'], 'rmse': [0.1, 0.2]}}
    with pytest.raises(ValueError, match='lengths'):
        means_from_per_seed(broken)


def test_the_sibling_is_outside_every_listing_of_per_pair_files() -> None:
    assert os.path.basename(per_seed_path_of('assets/datasets/interferences_caused_by_people_0_uce_0.json')).startswith('per_seed_')
