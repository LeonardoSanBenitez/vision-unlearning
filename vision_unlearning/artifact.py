"""Common storage/cascade contract for the artifacts produced by the benchmarks.

Several benchmark artifacts (Result Template results, per-entity interference summaries,
generated image datasets, ...) share the exact same *lifecycle*:

    1. If the artifact is already available locally, load and return it.
    2. Otherwise, if it exists on HuggingFace, download it, then load and return it.
    3. Otherwise, produce it from scratch, optionally persist it locally, and optionally
       upload it to HuggingFace.

Only the *storage granularity* differs between artifacts: a Result Template result is a
single JSON file, whereas a generated dataset is a folder of many images checked for
completeness. :class:`Artifact` captures the shared cascade as a template method
(:meth:`Artifact._resolve`) over a small set of storage hooks that each concrete artifact
implements, so the file-shaped and folder-shaped artifacts share one implementation of the
lifecycle instead of triplicating it.

This module deliberately depends only on ``pydantic`` and
``vision_unlearning.integrations.huggingface`` (both import-time free of torch/diffusers), so
it can be imported by the torch-free "lite" analysis path. It must stay outside the
torch-guarded imports in ``vision_unlearning/__init__.py``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from pydantic import BaseModel

from vision_unlearning.integrations.huggingface import (
    get_hf_token_from_env,
    huggingface_dataset_file_download,
    huggingface_dataset_file_exists,
    huggingface_dataset_file_upload,
)
from vision_unlearning.utils.logger import get_logger

logger = get_logger('artifact')

# The single HuggingFace dataset repository that currently hosts every I-CARE artifact.
# Kept here so the default lives in one place; a later workstream parameterises it per
# benchmark.
DEFAULT_REMOTE_REPOSITORY_NAME = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'


class Artifact(BaseModel):
    """Base class standardising the local/remote storage cascade of a benchmark artifact.

    A concrete artifact implements the storage hooks below and exposes its own typed
    ``compute(...)`` that delegates to :meth:`_resolve`. The hooks are deliberately nullary
    (apart from the HuggingFace token) so that the cascade is uniform regardless of whether
    the concrete computation needs runtime arguments: an artifact whose computation depends
    on runtime arguments (e.g. a generated dataset that needs seeds and prompts) stashes
    those arguments before calling :meth:`_resolve` and reads them back inside its hook
    overrides.
    """

    recompute_if_exists: bool = False
    save_outputs: bool = True
    upload_if_recomputed: bool = False
    base_folder: str = 'assets'
    remote_repository_name: str = DEFAULT_REMOTE_REPOSITORY_NAME

    # ------------------------------------------------------------------
    # Storage hooks — overridden by concrete artifacts
    # ------------------------------------------------------------------
    def _exists_local(self) -> bool:
        """Return True when the artifact is fully available in ``base_folder``."""
        raise NotImplementedError()

    def _exists_remote(self, hf_token: Optional[str]) -> bool:
        """Return True when the artifact exists in the HuggingFace repository."""
        raise NotImplementedError()

    def _pull_remote(self, hf_token: Optional[str]) -> None:
        """Download the artifact from HuggingFace into ``base_folder``."""
        raise NotImplementedError()

    def _load_local(self) -> Any:
        """Load and return the artifact that is already available locally."""
        raise NotImplementedError()

    def _produce_from_scratch(self) -> Any:
        """Produce the artifact from scratch and return it.

        This is the cascade's "compute" step. It is nullary so the template method stays
        uniform; a concrete artifact whose real computation takes arguments bridges this
        hook to its argument-taking method (see ``GeneratedDataset``).
        """
        raise NotImplementedError()

    def _persist_local(self, data: Any) -> None:
        """Persist a freshly produced artifact to ``base_folder``.

        No-op for artifacts that are already written to disk as a side effect of producing
        them (e.g. image generation writes files directly).
        """
        raise NotImplementedError()

    def _push_remote(self, hf_token: Optional[str]) -> None:
        """Upload the locally persisted artifact to HuggingFace."""
        raise NotImplementedError()

    def _validate(self, data: Any) -> None:
        """Assert invariants on the resolved artifact. Default: no invariants."""
        return None

    # ------------------------------------------------------------------
    # Template method — the shared cascade
    # ------------------------------------------------------------------
    def _resolve(self) -> Any:
        """Run the local -> remote -> from-scratch cascade and return the artifact.

        Returns the artifact loaded from the first source that has it, producing (and
        optionally persisting/uploading) it only when neither the local nor the remote copy
        is available.
        """
        hf_token: Optional[str] = get_hf_token_from_env()
        data: Any
        if not self.recompute_if_exists and self._exists_local():  # Local
            data = self._load_local()
        elif not self.recompute_if_exists and self._exists_remote(hf_token):  # Remote
            self._pull_remote(hf_token)
            assert self._exists_local(), (
                f"HuggingFace download for {type(self).__name__} completed but the artifact "
                f"is still not available locally."
            )
            data = self._load_local()
        else:  # Produce from scratch
            data = self._produce_from_scratch()
            if self.save_outputs:
                self._persist_local(data)
            if self.upload_if_recomputed:
                assert self.save_outputs, (
                    "upload_if_recomputed=True requires save_outputs=True "
                    "(no file to upload when save_outputs=False)."
                )
                assert hf_token, (
                    "upload_if_recomputed=True but HF_TOKEN is not set. "
                    "Set HF_TOKEN environment variable before calling compute()."
                )
                logger.info(
                    "Uploading recomputed %s to HuggingFace", type(self).__name__
                )
                self._push_remote(hf_token)
        self._validate(data)
        return data


class SingleFileArtifact(Artifact):
    """An :class:`Artifact` stored as exactly one JSON file.

    Implements every storage hook over a single file, driven by the file's local and remote
    paths. Concrete subclasses provide :meth:`_get_data_path_remote` (the path inside the
    HuggingFace repository, forward-slashed) and :meth:`_compute_from_scratch` (the actual
    computation), plus optionally :meth:`_validate`.

    ``_exists_remote`` swallows any network/authentication error and returns ``False`` so a
    transient failure falls through to a local recompute rather than raising — appropriate
    here because recomputing a single JSON file is cheap.
    """

    def _get_data_path_remote(self) -> str:
        raise NotImplementedError()

    def _get_data_path_local(self) -> str:
        return os.path.join(self.base_folder, self._get_data_path_remote())

    def _compute_from_scratch(self) -> Any:
        raise NotImplementedError()

    # ---- storage hooks over one JSON file ----
    def _exists_local(self) -> bool:
        return os.path.exists(self._get_data_path_local())

    def _exists_remote(self, hf_token: Optional[str]) -> bool:
        try:
            return huggingface_dataset_file_exists(
                self.remote_repository_name,
                self._get_data_path_remote(),
                token=hf_token,
            )
        except Exception as exc:
            logger.debug(
                "HF existence check failed for %s (falling through to local compute): %s",
                self._get_data_path_remote(), exc,
            )
            return False

    def _pull_remote(self, hf_token: Optional[str]) -> None:
        huggingface_dataset_file_download(
            folder_datasets=self.base_folder,
            dataset_repository=self.remote_repository_name,
            file_path=self._get_data_path_remote(),
            # None (not "") when unset: an empty string becomes an illegal
            # 'Authorization: Bearer ' header and breaks unauthenticated
            # downloads from public repositories.
            token=hf_token,
        )

    def _load_local(self) -> Any:
        with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
            return json.load(f)

    def _produce_from_scratch(self) -> Any:
        return self._compute_from_scratch()

    def _persist_local(self, data: Any) -> None:
        os.makedirs(os.path.dirname(self._get_data_path_local()), exist_ok=True)
        with open(self._get_data_path_local(), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _push_remote(self, hf_token: Optional[str]) -> None:
        # _resolve only calls _push_remote after asserting a token is present (uploads
        # require authentication); re-assert here to narrow the type for the upload call.
        assert hf_token is not None
        huggingface_dataset_file_upload(
            file_path=self._get_data_path_local(),
            dataset_repository=self.remote_repository_name,
            dataset_path=self._get_data_path_remote(),
            token=hf_token,
        )
