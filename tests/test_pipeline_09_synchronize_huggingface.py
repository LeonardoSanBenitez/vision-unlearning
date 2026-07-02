"""Tests for pipeline_09_synchronize_huggingface (pure logic — no network, no GPU).

All side effects (remote listing, uploads, sleeping) are injected as fakes.
"""
import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care.pipeline_09_synchronize_huggingface import (
    RemoteSnapshot,
    SyncItem,
    annotate_local_existence,
    annotate_remote_existence,
    build_report,
    enumerate_items,
    STAGES,
    synchronize,
    to_hf_path,
    upload_with_retry,
    write_status_json,
)


METADATA_BY_TASK: Dict[str, List[Dict[str, Any]]] = {
    "people": [{"name": "George_W_Bush"}, {"name": "Colin_Powell"}],
    "scenes": [{"name": "abbey"}, {"name": "airport"}],
}


def _items(stages: List[str], tasks: List[str] = ["people"], methods: List[str] = ["distil"], **kwargs: Any) -> List[SyncItem]:
    return enumerate_items(
        stages=stages,
        tasks=tasks,
        methods=methods,
        index_start=kwargs.get("index_start", 0),
        max_identities=kwargs.get("max_identities", 2),
        base_folder=kwargs.get("base_folder", "assets"),
        metadata_by_task=METADATA_BY_TASK,
    )


class TestToHfPath:
    def test_replaces_os_separators_with_forward_slashes(self) -> None:
        assert to_hf_path(os.path.join("datasets", "x.json")) == "datasets/x.json"


class TestEnumerateItems:
    def test_counts_per_stage(self) -> None:
        items = _items(list(STAGES), tasks=["people"], methods=["distil", "uce"])
        by_stage = {s: [i for i in items if i.stage == s] for s in STAGES}
        assert len(by_stage["metadata"]) == 1
        assert len(by_stage["act-fingerprints"]) == 1
        assert len(by_stage["interference-per-entity"]) == 1
        assert len(by_stage["models"]) == 4                 # 2 entities x 2 methods
        assert len(by_stage["generated-datasets"]) == 4
        assert len(by_stage["embeddings"]) == 5             # 1 baseline + 2 entities x 2 methods
        assert len(by_stage["interference-per-pair"]) == 4

    def test_epochs_resolved_from_configuration(self) -> None:
        items = _items(["embeddings"], methods=["distil"])
        entity = [i for i in items if i.name != "original"][0]
        assert entity.remote_path.endswith("_distil_400.json")  # people/distil = 400 epochs
        items_uce = _items(["embeddings"], methods=["uce"])
        entity_uce = [i for i in items_uce if i.name != "original"][0]
        assert entity_uce.remote_path.endswith("_uce_000.json")

    def test_entity_names_resolved_to_hf_convention(self) -> None:
        items = _items(["embeddings"])
        names = {i.name for i in items}
        assert "George W Bush" in names          # underscores -> spaces for people
        items_scenes = _items(["embeddings"], tasks=["scenes"])
        assert "an abbey scene" in {i.name for i in items_scenes}

    def test_models_use_raw_metadata_name(self) -> None:
        items = _items(["models"])
        assert {i.name for i in items} == {"George_W_Bush", "Colin_Powell"}

    def test_baseline_embedding_is_method_agnostic_and_unique_per_task(self) -> None:
        items = _items(["embeddings"], methods=["distil", "uce"])
        baselines = [i for i in items if i.name == "original"]
        assert len(baselines) == 1
        assert baselines[0].remote_path == "datasets/embeddings_people_original.json"
        assert baselines[0].method is None

    def test_remote_paths_use_forward_slashes(self) -> None:
        items = _items(list(STAGES), tasks=["people"], methods=["distil"])
        for item in items:
            assert "\\" not in item.remote_path, item.remote_path

    def test_interference_per_pair_paths(self) -> None:
        items = _items(["interference-per-pair"], methods=["distil"])
        assert items[0].remote_path == "datasets/interferences_caused_by_people_0_distil_400.json"
        assert items[1].remote_path == "datasets/interferences_caused_by_people_1_distil_400.json"

    def test_max_identities_and_index_start(self) -> None:
        items = _items(["models"], index_start=1, max_identities=5)
        assert [i.name for i in items] == ["Colin_Powell"]  # clipped to metadata length


class TestRemoteSnapshot:
    def test_file_and_folder_membership(self) -> None:
        snapshot = RemoteSnapshot(files={"datasets/a.json"}, folders={"models/x"})
        file_item = SyncItem(stage="embeddings", task="people", name="a", kind="file",
                             local_path="l", remote_path="datasets/a.json")
        folder_item = SyncItem(stage="models", task="people", name="x", kind="folder",
                               local_path="l", remote_path="models/x")
        missing_item = SyncItem(stage="embeddings", task="people", name="b", kind="file",
                                local_path="l", remote_path="datasets/b.json")
        # A folder path is never matched against files, and vice versa.
        folder_named_like_file = SyncItem(stage="models", task="people", name="a", kind="folder",
                                          local_path="l", remote_path="datasets/a.json")
        assert snapshot.exists(file_item)
        assert snapshot.exists(folder_item)
        assert not snapshot.exists(missing_item)
        assert not snapshot.exists(folder_named_like_file)

    def test_annotate_remote_existence(self) -> None:
        snapshot = RemoteSnapshot(files={"datasets/a.json"})
        items = [
            SyncItem(stage="embeddings", task="people", name="a", kind="file",
                     local_path="l", remote_path="datasets/a.json"),
            SyncItem(stage="embeddings", task="people", name="b", kind="file",
                     local_path="l", remote_path="datasets/b.json"),
        ]
        annotate_remote_existence(items, snapshot)
        assert items[0].remote_exists_before is True
        assert items[1].remote_exists_before is False


class TestAnnotateLocalExistence:
    def test_plain_file_stages_use_path_existence(self, tmp_path: Any) -> None:
        existing = tmp_path / "datasets" / "x.json"
        existing.parent.mkdir(parents=True)
        existing.write_text("{}")
        items = [
            SyncItem(stage="embeddings", task="people", name="x", kind="file",
                     local_path=str(existing), remote_path="datasets/x.json"),
            SyncItem(stage="embeddings", task="people", name="y", kind="file",
                     local_path=str(tmp_path / "datasets" / "y.json"), remote_path="datasets/y.json"),
        ]
        annotate_local_existence(items, METADATA_BY_TASK, seeds=[42], base_folder=str(tmp_path))
        assert items[0].local_exists is True
        assert items[1].local_exists is False


class TestUploadWithRetry:
    def test_success_first_try(self) -> None:
        calls: List[int] = []
        assert upload_with_retry(lambda: calls.append(1), "d", 3, 1.0, sleep_fn=lambda s: None) is None
        assert len(calls) == 1

    def test_retries_with_exponential_backoff_then_succeeds(self) -> None:
        attempts: List[int] = []
        waits: List[float] = []

        def flaky() -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("rate limited")

        assert upload_with_retry(flaky, "d", 5, 10.0, sleep_fn=waits.append) is None
        assert len(attempts) == 3
        assert waits == [10.0, 20.0]

    def test_exhausts_retries_and_returns_error(self) -> None:
        waits: List[float] = []

        def always_fails() -> None:
            raise RuntimeError("boom")

        error = upload_with_retry(always_fails, "d", 3, 1.0, sleep_fn=waits.append)
        assert error == "boom"
        assert waits == [1.0, 2.0]  # no sleep after the final attempt


def _sync_item(kind: str, local_exists: bool, remote_exists: bool, remote_path: str) -> SyncItem:
    item = SyncItem(stage="embeddings", task="people", name=remote_path, kind=kind,  # type: ignore[arg-type]
                    local_path=os.path.join("assets", remote_path.replace("/", os.sep)),
                    remote_path=remote_path)
    item.local_exists = local_exists
    item.remote_exists_before = remote_exists
    return item


class TestSynchronize:
    def test_uploads_only_local_only_items(self) -> None:
        uploaded_files: List[str] = []
        uploaded_folders: List[str] = []
        items = [
            _sync_item("file", True, False, "datasets/upload_me.json"),
            _sync_item("file", True, True, "datasets/already_there.json"),
            _sync_item("file", False, False, "datasets/missing_everywhere.json"),
            _sync_item("folder", True, False, "models/upload_me_folder"),
        ]
        snapshot = RemoteSnapshot()
        synchronize(
            items, snapshot, upload=True, repository="r", token="t", base_folder="assets",
            max_retries=1, retry_base_seconds=0.0,
            upload_file_fn=lambda **kw: uploaded_files.append(kw["dataset_path"]),
            upload_folder_fn=lambda **kw: uploaded_folders.append(kw["path_in_repo"]),
            sleep_fn=lambda s: None,
        )
        assert uploaded_files == ["datasets/upload_me.json"]
        assert uploaded_folders == ["models/upload_me_folder"]
        assert items[0].uploaded_now is True
        assert items[1].uploaded_now is False
        assert items[2].uploaded_now is False
        assert items[3].uploaded_now is True
        assert "datasets/upload_me.json" in snapshot.files
        assert "models/upload_me_folder" in snapshot.folders

    def test_no_upload_mode_uploads_nothing(self) -> None:
        calls: List[str] = []
        items = [_sync_item("file", True, False, "datasets/upload_me.json")]
        synchronize(
            items, RemoteSnapshot(), upload=False, repository="r", token="t", base_folder="assets",
            max_retries=1, retry_base_seconds=0.0,
            upload_file_fn=lambda **kw: calls.append("called"),
            upload_folder_fn=lambda **kw: calls.append("called"),
            sleep_fn=lambda s: None,
        )
        assert calls == []
        assert items[0].uploaded_now is False

    def test_upload_failure_is_recorded_and_run_continues(self) -> None:
        uploaded: List[str] = []

        def failing_then_ok(**kw: Any) -> None:
            if kw["dataset_path"] == "datasets/bad.json":
                raise RuntimeError("permanent failure")
            uploaded.append(kw["dataset_path"])

        items = [
            _sync_item("file", True, False, "datasets/bad.json"),
            _sync_item("file", True, False, "datasets/good.json"),
        ]
        synchronize(
            items, RemoteSnapshot(), upload=True, repository="r", token="t", base_folder="assets",
            max_retries=2, retry_base_seconds=0.0,
            upload_file_fn=failing_then_ok,
            upload_folder_fn=lambda **kw: None,
            sleep_fn=lambda s: None,
        )
        assert items[0].uploaded_now is False
        assert items[0].upload_error == "permanent failure"
        assert items[1].uploaded_now is True
        assert uploaded == ["datasets/good.json"]


class TestReporting:
    def test_build_report_counts(self) -> None:
        items = [
            _sync_item("file", True, False, "datasets/a.json"),
            _sync_item("file", True, True, "datasets/b.json"),
            _sync_item("file", False, False, "datasets/c.json"),
        ]
        items[0].uploaded_now = True
        report = build_report(items)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["expected"] == 3
        assert row["local"] == 2
        assert row["remote_before"] == 1
        assert row["uploaded_now"] == 1
        assert row["remote_after"] == 2
        assert row["missing_everywhere"] == 1
        assert row["upload_failures"] == 0

    def test_write_status_json(self, tmp_path: Any) -> None:
        items = [
            _sync_item("file", True, False, "datasets/not_uploaded.json"),
            _sync_item("file", False, False, "datasets/missing.json"),
        ]
        items.append(_sync_item("file", True, False, "datasets/failed.json"))
        items[-1].upload_error = "boom"
        status_path = str(tmp_path / "status.json")
        write_status_json(items, build_report(items), status_path, repository="r")
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        assert status["repository"] == "r"
        assert "datasets/missing.json" in status["missing_everywhere"]
        assert "datasets/not_uploaded.json" in status["local_only_not_uploaded"]
        assert status["upload_failures"] == {"datasets/failed.json": "boom"}
        assert isinstance(status["summary"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
