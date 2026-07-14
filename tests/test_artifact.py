"""Tests for the Artifact storage/cascade base class.

These exercise the local -> HuggingFace -> from-scratch cascade of ``vision_unlearning.artifact``
via two minimal subclasses:

- ``_ToyFileArtifact`` — a single JSON file, the shape used by Result Templates and the
  per-entity interference summary (via :class:`SingleFileArtifact`).
- ``_ToyFolderArtifact`` — a folder written directly by the "produce" step, the shape used by
  generated image datasets, with the runtime arguments stashed as ``PrivateAttr`` so the
  shared hooks stay nullary.

Neither subclass needs torch, so this module runs in the lite test tier and doubles as the
check that ``vision_unlearning.artifact`` is importable without the heavy dependency stack. If
that import ever started requiring torch, the lite CI job (which installs no torch) would fail
to collect this file.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional, cast

import pytest
from pydantic import PrivateAttr

import vision_unlearning.artifact as artifact_mod
from vision_unlearning.artifact import Artifact, SingleFileArtifact


# ---------------------------------------------------------------------------
# Toy single-file artifact (Result Template / InterferencePerEntity shape)
# ---------------------------------------------------------------------------
class _ToyFileArtifact(SingleFileArtifact):
    name: str = "toy"
    _scratch_called: bool = PrivateAttr(default=False)

    def _get_data_path_remote(self) -> str:
        return f"toy/{self.name}.json"

    def _compute_from_scratch(self) -> dict:
        self._scratch_called = True
        return {"result": {"produced": True, "name": self.name}}

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict)
        assert "result" in data


# ---------------------------------------------------------------------------
# Toy folder artifact (GeneratedDataset shape)
# ---------------------------------------------------------------------------
class _ToyFolderArtifact(Artifact):
    name: str = "folder"
    remote_available: bool = False  # test knob standing in for a HuggingFace check
    _pending: List[str] = PrivateAttr(default_factory=list)
    _produced: bool = PrivateAttr(default=False)
    _pulled: bool = PrivateAttr(default=False)
    _pushed: bool = PrivateAttr(default=False)

    def build(self, items: List[str]) -> str:
        """Typed public entry point; stashes runtime args and runs the cascade."""
        self._pending = list(items)
        return cast(str, self._resolve())

    @property
    def folder(self) -> str:
        return os.path.join(self.base_folder, self.name)

    def _write_items(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        for item in self._pending:
            with open(os.path.join(self.folder, f"{item}.txt"), "w", encoding="utf-8") as f:
                f.write(item)

    def _exists_local(self) -> bool:
        if not self._pending:
            return False
        return all(
            os.path.exists(os.path.join(self.folder, f"{item}.txt"))
            for item in self._pending
        )

    def _exists_remote(self, hf_token: Optional[str]) -> bool:
        return self.remote_available

    def _pull_remote(self, hf_token: Optional[str]) -> None:
        self._pulled = True
        self._write_items()  # simulate a download producing the folder contents

    def _load_local(self) -> str:
        return self.folder

    def _produce_from_scratch(self) -> str:
        self._produced = True
        self._write_items()  # producing writes the files directly, like image generation
        return self.folder

    def _persist_local(self, data: Any) -> None:
        return None  # no-op: _produce_from_scratch already wrote the folder

    def _push_remote(self, hf_token: Optional[str]) -> None:
        self._pushed = True


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------
def test_artifact_module_is_importable_without_torch() -> None:
    """The base module lives outside the torch-guarded imports.

    This file runs in the lite tier (no torch installed), so its successful collection is the
    real proof; the assertions below are a sanity check on the public surface.
    """
    assert issubclass(SingleFileArtifact, Artifact)
    assert hasattr(Artifact, "_resolve")


# ---------------------------------------------------------------------------
# Single-file cascade
# ---------------------------------------------------------------------------
class TestSingleFileCascade:
    def test_local_hit_returns_without_remote_or_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path))
        local = art._get_data_path_local()
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            json.dump({"result": {"from": "disk"}}, f)

        def _boom(*a: Any, **k: Any) -> bool:
            raise AssertionError("remote must not be consulted on a local hit")

        monkeypatch.setattr(artifact_mod, "huggingface_dataset_file_exists", _boom)

        data = art._resolve()
        assert data["result"] == {"from": "disk"}
        assert art._scratch_called is False

    def test_remote_hit_downloads_and_skips_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path))
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: True
        )

        def fake_download(**kw: Any) -> None:
            local = art._get_data_path_local()
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w", encoding="utf-8") as f:
                json.dump({"result": {"from": "remote"}}, f)

        monkeypatch.setattr(artifact_mod, "huggingface_dataset_file_download", fake_download)

        data = art._resolve()
        assert data["result"] == {"from": "remote"}
        assert art._scratch_called is False

    def test_scratch_persists_when_save_outputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path), save_outputs=True)
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        data = art._resolve()
        assert data["result"]["produced"] is True
        assert art._scratch_called is True
        assert os.path.exists(art._get_data_path_local())

    def test_scratch_does_not_persist_when_save_outputs_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path), save_outputs=False)
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        art._resolve()
        assert not os.path.exists(art._get_data_path_local())

    def test_network_error_falls_through_to_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A raising existence check must be swallowed and fall through to a local recompute."""
        art = _ToyFileArtifact(base_folder=str(tmp_path))

        def _raise(*a: Any, **k: Any) -> bool:
            raise ConnectionError("HuggingFace unreachable")

        monkeypatch.setattr(artifact_mod, "huggingface_dataset_file_exists", _raise)
        data = art._resolve()
        assert art._scratch_called is True
        assert data["result"]["produced"] is True

    def test_recompute_if_exists_ignores_local(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path), recompute_if_exists=True)
        local = art._get_data_path_local()
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            json.dump({"result": {"from": "stale disk"}}, f)
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        data = art._resolve()
        assert data["result"]["produced"] is True
        assert art._scratch_called is True

    def test_upload_after_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(
            base_folder=str(tmp_path), upload_if_recomputed=True
        )
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        monkeypatch.setattr(artifact_mod, "get_hf_token_from_env", lambda: "fake_token")
        upload_calls: List[Any] = []
        monkeypatch.setattr(
            artifact_mod,
            "huggingface_dataset_file_upload",
            lambda **kw: upload_calls.append(kw),
        )
        art._resolve()
        assert len(upload_calls) == 1
        assert upload_calls[0]["dataset_path"] == art._get_data_path_remote()

    def test_upload_requires_save_outputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(
            base_folder=str(tmp_path), upload_if_recomputed=True, save_outputs=False
        )
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        monkeypatch.setattr(artifact_mod, "get_hf_token_from_env", lambda: "fake_token")
        with pytest.raises(AssertionError, match="save_outputs"):
            art._resolve()

    def test_upload_requires_hf_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        art = _ToyFileArtifact(base_folder=str(tmp_path), upload_if_recomputed=True)
        monkeypatch.setattr(
            artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False
        )
        monkeypatch.setattr(artifact_mod, "get_hf_token_from_env", lambda: None)
        with pytest.raises(AssertionError, match="HF_TOKEN"):
            art._resolve()


# ---------------------------------------------------------------------------
# Folder cascade (proves the base contract spans multi-file artifacts too)
# ---------------------------------------------------------------------------
class TestFolderCascade:
    def test_local_hit(self, tmp_path: Any) -> None:
        art = _ToyFolderArtifact(base_folder=str(tmp_path))
        # Pre-populate the folder as if a previous run completed it.
        os.makedirs(art.folder, exist_ok=True)
        for item in ["a", "b"]:
            with open(os.path.join(art.folder, f"{item}.txt"), "w", encoding="utf-8") as f:
                f.write(item)
        result = art.build(["a", "b"])
        assert result == art.folder
        assert art._produced is False
        assert art._pulled is False

    def test_remote_hit(self, tmp_path: Any) -> None:
        art = _ToyFolderArtifact(base_folder=str(tmp_path), remote_available=True)
        result = art.build(["a", "b"])
        assert result == art.folder
        assert art._pulled is True
        assert art._produced is False
        assert os.path.exists(os.path.join(art.folder, "a.txt"))

    def test_scratch(self, tmp_path: Any) -> None:
        art = _ToyFolderArtifact(base_folder=str(tmp_path), remote_available=False)
        result = art.build(["a", "b", "c"])
        assert result == art.folder
        assert art._produced is True
        assert art._pulled is False
        assert all(
            os.path.exists(os.path.join(art.folder, f"{i}.txt")) for i in ["a", "b", "c"]
        )

    def test_upload_after_scratch(self, tmp_path: Any) -> None:
        art = _ToyFolderArtifact(
            base_folder=str(tmp_path), remote_available=False, upload_if_recomputed=True
        )
        # get_hf_token_from_env is looked up in the artifact module; a real env token or a
        # patched one both work — here the default env may be empty, so patch it.
        import unittest.mock as _mock
        with _mock.patch.object(artifact_mod, "get_hf_token_from_env", lambda: "fake_token"):
            art.build(["a"])
        assert art._pushed is True
