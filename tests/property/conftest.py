"""
Hypothesis settings profiles for property-based tests.

Two profiles are registered:

- ``dev`` (default): fast local iteration, no deadline changes.
- ``ci``: relaxed deadline (``deadline=None``, to avoid runner-speed flakiness) and a
  moderate example count.

Selected via the ``HYPOTHESIS_PROFILE`` environment variable, set explicitly in the CI
workflows that run pytest (``.github/workflows/lite.yml`` and
``.github/workflows/all_version_pytest.yml``) and in the Docker ``make test`` target.
Locally, with the variable unset, this defaults to ``dev``.
"""
import os

from hypothesis import settings

settings.register_profile("dev", max_examples=100)
settings.register_profile("ci", max_examples=200, deadline=None)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
