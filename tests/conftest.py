"""
Pytest configuration for vision-unlearning tests.
"""
import importlib.util
from typing import Iterator

import pytest

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
    "test_epoch_checkpoint_hook.py",
    "test_data_generation.py",
    "test_gradient_weighting.py",
    "test_model_management.py",
    "test_testbed.py",
    "test_metrics/test_image.py",
    "test_metrics/test_image_and_image.py",
    "test_metrics/test_image_and_text.py",
    "test_unlearner/test_fade.py",
    "test_unlearner/test_lora_trainer.py",
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


@pytest.fixture(autouse=True)
def _no_real_huggingface_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block real HuggingFace network calls by default, for every test in the suite.

    Every ``SingleFileArtifact`` subclass (``ResultTemplate``, ``InterferencePerEntity``,
    ``InterferencePerPair``, ``MetadataFiltered``, ``Similarity``, ``BaselineEmbeddings``,
    ``EntityEmbeddings``) resolves ``huggingface_dataset_file_exists`` through
    ``vision_unlearning.artifact`` (imported at module level there), so patching it here once
    makes "not found on HuggingFace" the default outcome for the whole suite, unless a test
    explicitly overrides it with its own ``monkeypatch.setattr(artifact_mod,
    "huggingface_dataset_file_exists", ...)`` (a test-level patch always wins over this one,
    since it runs after this fixture in the same test).

    Why this exists as a *global*, not a per-file, fixture: a test that forgets to write a local
    fixture file (or add its own mock) previously fell through silently to a real network call
    against the production HuggingFace repo -- slow, and observed in practice to occasionally
    fail from transient network conditions inside CI/Docker, producing a false test failure
    unrelated to the code under test, and a real (if small) contributor to hitting HuggingFace
    rate limits. A single test file adding its own local safety net (as `test_embedding_rts.py`
    did before this fixture existed) only protects that one file; the same silent gap can and did
    reappear in a different file. This fixture makes "no real network" the suite's default
    property instead of a convention every new test file has to remember.

    Scope note: this covers the single-file artifact cascade only. `GeneratedDataset`
    (`datasets/testbed.py`) resolves its HuggingFace calls via a *deferred* (function-body)
    import of `vision_unlearning.integrations.huggingface`, and its `_exists_remote` is
    deliberately left unguarded on network errors (a transient failure must surface rather than
    silently trigger an expensive full-dataset regeneration) -- forcing it to a blanket "not
    found" here would fight that documented design, so it is out of scope; the heavy-tier
    `test_testbed.py` suite already mocks that path per-test where needed.
    """
    import vision_unlearning.artifact as _artifact_mod
    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False)
    yield
