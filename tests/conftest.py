"""
Pytest configuration for vision-unlearning tests.
"""


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
