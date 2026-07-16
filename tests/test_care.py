"""Tests for the benchmark-agnostic "CARE" notation base classes (``benchmarks/care.py``).

Neither class does anything beyond fixing a shared shape (they inherit the full storage
cascade from ``SingleFileArtifact``), so these tests only lock two things: the module is
importable without torch (lite tier), and I-CARE's concrete artifacts are structurally
`isinstance` of the generic shape they claim to implement.
"""
from __future__ import annotations

import vision_unlearning.benchmarks as benchmarks_mod
from vision_unlearning.artifact import SingleFileArtifact
from vision_unlearning.benchmarks.care import MetricEffectPerEntity, MetricEffectPerEntityPair
from vision_unlearning.benchmarks.I_care.metadata import InterferencePerEntity, InterferencePerPair


def test_care_module_is_importable_without_torch() -> None:
    """This file runs in the lite tier (no torch installed); successful collection is the
    real proof that ``benchmarks/care.py`` stays outside the torch-guarded imports."""
    assert issubclass(MetricEffectPerEntity, SingleFileArtifact)
    assert issubclass(MetricEffectPerEntityPair, SingleFileArtifact)


def test_interference_per_entity_is_a_metric_effect_per_entity() -> None:
    """I-CARE's InterferencePerEntity is the concrete instance of the generic per-entity
    metric-effect shape."""
    assert issubclass(InterferencePerEntity, MetricEffectPerEntity)


def test_interference_per_pair_is_a_metric_effect_per_entity_pair() -> None:
    """I-CARE's InterferencePerPair is the concrete instance of the generic per-entity-pair
    metric-effect shape."""
    assert issubclass(InterferencePerPair, MetricEffectPerEntityPair)


def test_benchmarks_init_exports_only_the_agnostic_base_classes() -> None:
    """`vision_unlearning.benchmarks` must export the shared CARE base classes only — not
    I-CARE's (or any other single benchmark's) names — so a second benchmark's own
    `import *` cannot collide with I-CARE's public surface."""
    assert set(benchmarks_mod.__all__) == {"MetricEffectPerEntity", "MetricEffectPerEntityPair"}
    # I-CARE-specific names (e.g. a Result Template class) must not leak into this namespace.
    assert not hasattr(benchmarks_mod, "ResultTemplate")
