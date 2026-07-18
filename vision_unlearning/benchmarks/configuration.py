"""Benchmark-agnostic configuration record types shared across every benchmark.

Each benchmark under `vision_unlearning.benchmarks` describes its metrics, its embedding
functions, and its unlearning methods with small `(software name, display name, ...)` records.
The record *shapes* are identical across benchmarks — only the concrete instances (which
metrics exist, what they are called) are benchmark-specific. Those instances, and every
registry/mapping derived from them, stay in each benchmark's own `configuration.py`
(e.g. `I_care/configuration.py`'s `MP_REGISTRY`/`S_REGISTRY`/`ALGORITHM_REGISTRY`); only the
shared shapes live here, next to `benchmarks/care.py`'s shared storage shapes.

Kept deliberately thin: no registries and no instances, exactly as `care.py` holds no concrete
artifacts. A sibling benchmark imports these record types from here rather than from any other
benchmark, so benchmarks never import each other.
"""
from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel

# The up/down arrow denotes a metric's HEALTHY / higher-quality direction. What a given arrow
# means concretely ("↑" vs "more effect") is a per-benchmark interpretation defined where the
# record is instantiated (e.g. I-CARE's direction-convention comment above its registries).
type_direction = Literal["↑", "↓"]


class MetricWithDirectionSpec(BaseModel):
    """A `(software name, display name, direction)` record for a metric that has a meaningful
    "healthy"/"high-quality" direction (e.g. a per-pair effect metric or a similarity metric).

    ``name_pretty`` is ``None`` for a metric that is computed and typed but intentionally not
    exposed for display/selection. ``direction`` uses the shared ``type_direction`` arrow.
    """
    name: str
    name_pretty: Optional[str] = None
    direction: type_direction


class LSpec(BaseModel):
    """A `(software name, display name)` record for a latent embedding function."""
    name: str
    name_pretty: str


class UnlearningAlgorithmSpec(BaseModel):
    """A `(software name, display name)` record for an unlearning method."""
    name: str
    name_pretty: str
