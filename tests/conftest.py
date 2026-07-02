"""
Pytest configuration for vision-unlearning tests.
"""
import importlib.util

# Test files that import torch/diffusers at module level (directly or transitively).
# These fail at *collection* time in the lite tier (no torch installed), before any
# marker is ever evaluated -- so a `heavy` marker cannot guard this boundary, only an
# explicit collect_ignore list can. See CONTRIBUTING.md Section 6 and
# PLAN-TASK-2026-07-01-TestTooling.md Workstream B2.
#
# A NEW heavy test file that is not added here will fail import in the lite CI job
# (.github/workflows/lite.yml) with a ModuleNotFoundError, forcing the author to either
# classify it here or keep it torch-free. That failure is the intended rot-protection
# behaviour, not a bug.
_HEAVY_TEST_FILES = [
    "test_data_generation.py",
    "test_gradient_weighting.py",
    "test_model_management.py",
    "test_testbed.py",
    "test_metrics/test_image.py",
    "test_metrics/test_image_and_image.py",
    "test_metrics/test_image_and_text.py",
    "test_unlearner/test_fade.py",
    "test_unlearner/test_gpu_unlearning.py",
]

if importlib.util.find_spec("torch") is None:
    collect_ignore = _HEAVY_TEST_FILES
    print(
        f"heavy tests skipped: torch not installed ({len(_HEAVY_TEST_FILES)} files; "
        "see CONTRIBUTING.md Section 6 for the lite/heavy test tier)"
    )


def pytest_configure(config: "object") -> None:
    """Register custom markers.

    ``gpu`` marks tests that require a CUDA/ROCm device. They are excluded
    from the default ``make test`` run and run explicitly via
    ``pytest -m gpu``.
    """
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "gpu: test requires a GPU (CUDA/ROCm); excluded from default run",
    )
