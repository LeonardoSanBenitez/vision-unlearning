"""`vision_unlearning.benchmarks` — the top-level namespace shared by every benchmark.

This package exports only the benchmark-agnostic base classes (`benchmarks/care.py`), not any
single benchmark's names. Each benchmark keeps its own full export in its own `__init__.py`
(e.g. `vision_unlearning.benchmarks.I_care`) so that multiple benchmarks can coexist under this
namespace without their public names colliding.
"""
from vision_unlearning.benchmarks.care import MetricEffectPerEntity, MetricEffectPerEntityPair

__all__ = [
    "MetricEffectPerEntity",
    "MetricEffectPerEntityPair",
]
