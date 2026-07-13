"""Tests for pipeline_08_run_all_rts (pure logic — no network, no GPU, no torch).

``_should_skip`` and ``_compute_and_report`` only rely on the duck-typed interface a
Result Template exposes (``_get_data_path_local``, ``_get_data_path_remote``,
``remote_repository_name``, ``compute``), so a lightweight fake stands in for the real
(pydantic, torch-adjacent) ``ResultTemplate`` subclasses.
"""
import logging
from typing import Any, Callable, FrozenSet, List, Optional, cast

import pytest

from vision_unlearning.benchmarks.I_care.pipeline_08_run_all_rts import (
    _compute_and_report,
    _should_skip,
    parse_args,
    resolve_rt_names,
)


class _FakeRT:
    """Minimal stand-in for a ``ResultTemplate`` subclass.

    Exposes exactly the surface ``_should_skip``/``_compute_and_report`` use, so tests
    do not need to instantiate a real (heavy) Result Template.
    """

    def __init__(
        self,
        local_path: str,
        remote_path: str = "results/Fake/x.json",
        compute_fn: Optional[Callable[[], dict]] = None,
        remote_repository_name: str = "some/repo",
    ) -> None:
        self._local_path = local_path
        self._remote_path = remote_path
        self.remote_repository_name = remote_repository_name
        self._compute_fn = compute_fn or (lambda: {"result": {}})
        self.compute_called = False

    def _get_data_path_local(self) -> str:
        return self._local_path

    def _get_data_path_remote(self) -> str:
        return self._remote_path

    def compute(self) -> dict:
        self.compute_called = True
        return self._compute_fn()


class TestShouldSkip:
    def test_returns_false_when_absent_locally_and_remotely(self, tmp_path: Any) -> None:
        rt = _FakeRT(local_path=str(tmp_path / "missing.json"))
        assert _should_skip(cast(Any, rt), hf_files=frozenset(), upload_if_recomputed=False) is False

    def test_returns_true_when_present_locally(self, tmp_path: Any) -> None:
        local = tmp_path / "present.json"
        local.write_text("{}")
        rt = _FakeRT(local_path=str(local))
        assert _should_skip(cast(Any, rt), hf_files=frozenset(), upload_if_recomputed=False) is True

    def test_returns_true_when_present_on_hf(self, tmp_path: Any) -> None:
        rt = _FakeRT(local_path=str(tmp_path / "missing.json"), remote_path="results/Fake/x.json")
        hf_files: FrozenSet[str] = frozenset({"results/Fake/x.json"})
        assert _should_skip(cast(Any, rt), hf_files=hf_files, upload_if_recomputed=False) is True

    def test_local_hit_does_not_upload_without_hf_token(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """upload_if_recomputed=True but HF_TOKEN unset: skip is reported, no upload attempted."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        local = tmp_path / "present.json"
        local.write_text("{}")
        rt = _FakeRT(local_path=str(local), remote_path="results/Fake/x.json")
        result = _should_skip(cast(Any, rt), hf_files=frozenset(), upload_if_recomputed=True)
        assert result is True

    def test_local_hit_uploads_when_not_yet_on_hf_and_token_set(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        uploaded: List[dict] = []
        monkeypatch.setattr(
            "vision_unlearning.integrations.huggingface.huggingface_dataset_file_upload",
            lambda **kw: uploaded.append(kw),
        )
        local = tmp_path / "present.json"
        local.write_text("{}")
        rt = _FakeRT(local_path=str(local), remote_path="results/Fake/x.json")
        result = _should_skip(cast(Any, rt), hf_files=frozenset(), upload_if_recomputed=True)
        assert result is True
        assert len(uploaded) == 1
        assert uploaded[0]["dataset_path"] == "results/Fake/x.json"

    def test_local_hit_skips_upload_when_already_on_hf(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        uploaded: List[dict] = []
        monkeypatch.setattr(
            "vision_unlearning.integrations.huggingface.huggingface_dataset_file_upload",
            lambda **kw: uploaded.append(kw),
        )
        local = tmp_path / "present.json"
        local.write_text("{}")
        rt = _FakeRT(local_path=str(local), remote_path="results/Fake/x.json")
        result = _should_skip(
            cast(Any, rt), hf_files=frozenset({"results/Fake/x.json"}), upload_if_recomputed=True
        )
        assert result is True
        assert uploaded == []


class TestComputeAndReport:
    def test_local_hit_skips_compute(self, tmp_path: Any, capsys: Any) -> None:
        local = tmp_path / "present.json"
        local.write_text("{}")
        rt = _FakeRT(local_path=str(local))
        _compute_and_report(cast(Any, rt), frozenset(), False, "context")
        assert rt.compute_called is False
        assert capsys.readouterr().out == ""

    def test_hf_hit_skips_compute(self, tmp_path: Any) -> None:
        rt = _FakeRT(local_path=str(tmp_path / "missing.json"), remote_path="results/Fake/x.json")
        _compute_and_report(cast(Any, rt), frozenset({"results/Fake/x.json"}), False, "context")
        assert rt.compute_called is False

    def test_computes_and_prints_marker_on_success(self, tmp_path: Any, capsys: Any) -> None:
        rt = _FakeRT(local_path=str(tmp_path / "missing.json"))
        _compute_and_report(cast(Any, rt), frozenset(), False, "context")
        assert rt.compute_called is True
        assert capsys.readouterr().out == "."

    def test_logs_and_continues_on_compute_failure(
        self, tmp_path: Any, caplog: Any
    ) -> None:
        def _boom() -> dict:
            raise RuntimeError("boom")

        rt = _FakeRT(local_path=str(tmp_path / "missing.json"), compute_fn=_boom)
        with caplog.at_level(logging.WARNING):
            _compute_and_report(cast(Any, rt), frozenset(), False, "my-context")
        assert "my-context failed: boom" in caplog.text

    def test_recomputes_when_upload_if_recomputed_but_not_yet_on_hf(
        self, tmp_path: Any
    ) -> None:
        """upload_if_recomputed alone (no local, no HF hit) does not shortcut compute()."""
        rt = _FakeRT(local_path=str(tmp_path / "missing.json"))
        _compute_and_report(cast(Any, rt), frozenset(), True, "context")
        assert rt.compute_called is True


class TestResolveRtNames:
    def test_matches_interference_visual_summary(self) -> None:
        assert resolve_rt_names(["InterferenceVisualSummary"]) == ["interferencevisualsummary"]

    def test_all_keyword_returns_full_registered_list(self) -> None:
        from vision_unlearning.benchmarks.I_care.pipeline_08_run_all_rts import ALL_RT_NAMES
        assert resolve_rt_names(["all"]) == ALL_RT_NAMES
        assert "interferencevisualsummary" in ALL_RT_NAMES
        assert "unlearningvisualsummary" not in ALL_RT_NAMES

    def test_unmatched_name_returns_empty_and_warns(self, caplog: Any) -> None:
        with caplog.at_level(logging.WARNING):
            result = resolve_rt_names(["NotARealRT"])
        assert result == []
        assert "did not match any known RT" in caplog.text


class TestMpArgument:
    def test_defaults_to_all_five_metrics(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("sys.argv", ["pipeline_08_run_all_rts.py"])
        args = parse_args()
        assert set(args.mp) == {"brisque_diff", "clip_diff", "rmse", "ssim", "dino_diff"}

    def test_restricts_to_requested_metric(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("sys.argv", ["pipeline_08_run_all_rts.py", "--mp", "clip_diff"])
        args = parse_args()
        assert args.mp == ["clip_diff"]

    def test_accepts_multiple_metrics(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "sys.argv", ["pipeline_08_run_all_rts.py", "--mp", "clip_diff", "ssim"]
        )
        args = parse_args()
        assert args.mp == ["clip_diff", "ssim"]

    def test_rejects_unknown_metric(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "sys.argv", ["pipeline_08_run_all_rts.py", "--mp", "not_a_real_metric"]
        )
        with pytest.raises(SystemExit):
            parse_args()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
