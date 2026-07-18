"""Tests for the benchmark-agnostic base modules promoted to the shared `benchmarks/` level:
`benchmarks/result_template.py` (the `ResultTemplate` base class) and `benchmarks/configuration.py`
(configuration record shapes).

These lock two contracts:

1. The promoted names import from their new shared location, and I-CARE's own
   `result_templates` / `configuration` modules still expose them (the re-import shims keep every
   existing importer working). I-CARE's own `ResultTemplateMatrix` stays in `I_care`, but still
   inherits the shared `ResultTemplate` base.
2. Importing `benchmarks/result_template.py` pulls in neither `torch` nor any single benchmark's
   package — this is what keeps it inside the lite test tier and structurally enforces the
   no-sibling-import contract (a shared base must not depend on a concrete benchmark).

This file runs in the lite tier (no torch installed); successful collection is itself part of
proof #2.
"""
from __future__ import annotations

import subprocess
import sys

from vision_unlearning.artifact import SingleFileArtifact
from vision_unlearning.benchmarks.result_template import ResultTemplate
from vision_unlearning.benchmarks.configuration import (
    type_direction,
    MetricWithDirectionSpec,
    LSpec,
    UnlearningAlgorithmSpec,
)


def test_result_template_base_class_imports_from_shared_location() -> None:
    assert issubclass(ResultTemplate, SingleFileArtifact)


def test_icare_result_template_matrix_stays_in_icare_but_inherits_the_shared_base() -> None:
    """`ResultTemplateMatrix` is deliberately NOT promoted — it stays an I-CARE class — but its
    base is the shared `ResultTemplate`."""
    from vision_unlearning.benchmarks.I_care.result_templates import ResultTemplateMatrix
    assert issubclass(ResultTemplateMatrix, ResultTemplate)


def test_configuration_record_shapes_import_from_shared_location() -> None:
    # type_direction is the Literal["↑", "↓"] alias.
    assert "↑" in type_direction.__args__ and "↓" in type_direction.__args__
    spec = MetricWithDirectionSpec(name="clip_diff", name_pretty="Delta Clip", direction="↑")
    assert spec.name_pretty == "Delta Clip"
    # name_pretty is optional on MetricWithDirectionSpec (a non-displayed diagnostic metric)...
    assert MetricWithDirectionSpec(name="weight_overlap", direction="↑").name_pretty is None
    # ...but required on the two (name, name_pretty) records.
    assert LSpec(name="dino_embedding", name_pretty="DINOv2 Embedding").name_pretty == "DINOv2 Embedding"
    assert UnlearningAlgorithmSpec(name="distil", name_pretty="spare").name_pretty == "spare"


def test_icare_reimports_resolve_to_the_shared_definitions() -> None:
    """The re-import shims mean I-CARE's own modules still expose the exact same objects, so every
    `from ...I_care.result_templates import ResultTemplate` keeps resolving."""
    from vision_unlearning.benchmarks.I_care import result_templates as icare_rt
    from vision_unlearning.benchmarks.I_care import configuration as icare_cfg

    assert icare_rt.ResultTemplate is ResultTemplate
    assert icare_cfg.MetricWithDirectionSpec is MetricWithDirectionSpec
    assert icare_cfg.type_direction is type_direction


def test_shared_result_template_imports_no_torch_and_no_benchmark_package() -> None:
    """Importing the shared Result Template module must not pull in torch or any single
    benchmark's package. Run in a subprocess for a clean sys.modules state (this is meaningful
    in the heavy tier, where torch IS installed and would leak in if imported transitively)."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import vision_unlearning.benchmarks.result_template as m; "
            "import sys; "
            "assert 'torch' not in sys.modules, 'result_template imported torch'; "
            "leaked = [k for k in sys.modules if k.startswith('vision_unlearning.benchmarks.I_care')]; "
            "assert not leaked, f'result_template imported a benchmark package: {leaked}'"
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"shared result_template import leaked torch or a benchmark package.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
