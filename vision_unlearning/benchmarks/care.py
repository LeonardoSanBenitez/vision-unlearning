"""Benchmark-agnostic notation ("CARE") shared across I-CARE and future benchmarks.

Every benchmark under `vision_unlearning.benchmarks` measures the effect of an intervention
(e.g. unlearning) on entities, and every one so far has needed the same two storage shapes: a
metric value attached to a single entity, and a metric value attached to an ordered pair of
entities (e.g. emitter/receiver). I-CARE's own artifacts (`InterferencePerEntity`,
`InterferencePerPair`, in `I_care/metadata.py`) are the concrete I-CARE instances of these two
shapes — "interference" is I-CARE's name for the effect it measures. A future benchmark
measuring a different effect (e.g. accuracy retention) would define its own concrete artifact
with its own name, while inheriting the shared storage cascade and structural identity from
here, so that "this is a per-entity metric-effect artifact" is a checkable `isinstance`
relationship rather than a naming convention.

Kept deliberately thin: no per-metric classes, and no shared column-naming logic. Naming
conventions for individual metrics/columns are genuinely benchmark-specific (see
`choose_metric_column_interference_per_entity` in `I_care/metadata.py` for I-CARE's own), so
they stay with each benchmark's concrete subclass rather than being forced into this shared
base. This module only fixes the one shape the storage classes already have in common.
"""
from __future__ import annotations

from vision_unlearning.artifact import SingleFileArtifact


class MetricEffectPerEntity(SingleFileArtifact):
    """A metric value attached to each entity in a task (I-CARE calls this `m_e`).

    Always subclassed by a concrete, benchmark-specific artifact — e.g. I-CARE's
    `InterferencePerEntity` — which supplies the storage path and computation. Never
    instantiated directly.
    """


class MetricEffectPerEntityPair(SingleFileArtifact):
    """A metric value attached to an ordered pair of entities in a task (I-CARE calls this
    `m_p`; the pair's ordering is the emitter/receiver distinction).

    Always subclassed by a concrete, benchmark-specific artifact — e.g. I-CARE's
    `InterferencePerPair` — which supplies the storage path and computation. Never
    instantiated directly.
    """
