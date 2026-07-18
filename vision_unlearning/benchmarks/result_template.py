"""Benchmark-agnostic Result Template base class shared across every benchmark.

A Result Template is the canonical analysis unit: a class that computes, persists, and
visualizes one kind of quantitative result. ``ResultTemplate`` is benchmark-agnostic — it carries
no metric semantics, no file-path conventions, and no benchmark-specific vocabulary — so it
belongs at the shared `benchmarks/` level, next to `benchmarks/care.py`, rather than inside any
single benchmark. A concrete benchmark (e.g. I-CARE) subclasses it to build its own Result
Template catalogue.

``ResultTemplate`` adds the result-file conventions (a JSON per class × parameter set, a
``results/{name}/{params}.json`` remote path, and a ``{"result": ...}`` payload shape) on top of
the ``SingleFileArtifact`` storage cascade (local -> HuggingFace -> compute-from-scratch).

Kept inside the lite test tier on purpose: the only non-stdlib imports are matplotlib and
``SingleFileArtifact``. Importing this module must never pull in ``torch`` or any single
benchmark's package, which structurally enforces the no-sibling-import contract.
"""
from __future__ import annotations

import io
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vision_unlearning.artifact import SingleFileArtifact


class ResultTemplate(SingleFileArtifact):


    def _serialize_parameters(self) -> str:
        raise NotImplementedError()

    def _get_data_path_remote(self) -> str:
        # Forward slashes always: this is a path inside the HuggingFace repository, not a
        # filesystem path. os.path.join would produce backslashes on Windows hosts, which
        # HF rejects (the file then looks nonexistent and compute() falls through to
        # _compute_from_scratch).
        return f"results/{self.__class__.__name__.replace('ResultTemplate', '')}/{self._serialize_parameters()}.json"

    @classmethod
    def _fig_to_bytes(cls, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    def _compute_from_scratch(self) -> dict | list:
        raise NotImplementedError()

    def _validate(self, data: Any) -> None:
        assert type(data) == dict, f"Expected a dict in the json file, but got {type(data)}"
        assert 'result' in data, f"Expected 'result' key in the json file, but got {list(data.keys())}"
        assert type(data['result']) in [dict, list], f"Expected 'result' to be a dict or list, but got {type(data['result'])}"

    def compute(self) -> dict:
        """Resolve this result from local disk, HuggingFace, or by computing it from scratch.

        The storage cascade (local -> remote -> from-scratch, with optional persist and
        upload) lives in :class:`~vision_unlearning.artifact.SingleFileArtifact`; this method
        only pins the return type.
        """
        return cast(dict, self._resolve())
