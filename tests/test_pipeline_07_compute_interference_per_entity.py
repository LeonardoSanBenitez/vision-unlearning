"""Tests for pipeline_07's per-entity error isolation and run ledger (no network, no torch)."""
import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import pipeline_07_compute_interference_per_entity as p07
from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger


_TASK = "people"
_METHOD = "uce"
_EPOCHS = 0
_ENTITIES = ["Alice", "Bob"]


def _fake_metadata(*_a: Any, **_k: Any) -> List[Dict[str, Any]]:
    return [{"name": name} for name in _ENTITIES]


def _setup_base_folder(tmp_path: Any) -> str:
    """Create the minimal on-disk layout compute_for_task expects to reach its per-entity loop."""
    base_folder = str(tmp_path)
    datasets_dir = os.path.join(base_folder, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # Baseline embedding file (compute_for_task raises FileNotFoundError without it).
    baseline_path = os.path.join(datasets_dir, f"embeddings_{_TASK}_original.json")
    with open(baseline_path, "w", encoding="utf-8") as fh:
        json.dump({"embeddings": []}, fh)

    # One dummy per-pair file per entity so the "file missing" skip branch is not hit --
    # get_interference_per_pair itself is monkeypatched per-test, so this file's content
    # is irrelevant, only its existence is checked.
    for index in range(len(_ENTITIES)):
        path = p07.get_interference_per_pair_path(
            _TASK, index, _METHOD, _EPOCHS, base_folder=base_folder,
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({}, fh)

    return base_folder


class TestPerEntityErrorIsolation:
    def test_one_entity_failing_does_not_abort_the_others(
        self, tmp_path: Any, monkeypatch: Any, caplog: Any,
    ) -> None:
        """A real, pre-existing bug class this guards against: before this change, any
        exception raised while summarising a single entity crashed compute_for_task for
        every entity and every (method, epochs) combination in the same call."""
        base_folder = _setup_base_folder(tmp_path)
        monkeypatch.setattr(p07, "get_metadata_filtered", _fake_metadata)
        monkeypatch.setattr(
            p07, "get_interference_per_pair_inverse",
            lambda *a, **k: {
                "Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
                "Bob": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
            },
        )

        def _fake_get_interference_per_pair(task: str, index: int, *a: Any, **k: Any) -> Dict[str, Any]:
            if index == 1:
                raise RuntimeError("simulated corrupt per-pair file")
            return {"Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0}}

        monkeypatch.setattr(p07, "get_interference_per_pair", _fake_get_interference_per_pair)

        # compute_for_task must complete (not raise) even though index=1 fails internally.
        p07.compute_for_task(
            task=_TASK, methods=[_METHOD], num_train_epochs_list=[_EPOCHS],
            index_start=0, max_identities=len(_ENTITIES),
            embedding_assets_folder=os.path.join(base_folder, "datasets"),
            base_folder=base_folder,
        )

        result_path = p07.get_interference_per_entity_path(_TASK, base_folder=base_folder)
        with open(result_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data) == len(_ENTITIES)  # both rows present
        assert f"metric_{_METHOD}_{_EPOCHS}_emitter_average_clip_diff (↑)" in data[0]  # Alice succeeded
        assert f"metric_{_METHOD}_{_EPOCHS}_emitter_average_clip_diff (↑)" not in data[1]  # Bob's failed, no columns
        assert "InterferencePerEntity failed for" in caplog.text

    def test_ledger_records_ok_skipped_and_failed(self, tmp_path: Any, monkeypatch: Any) -> None:
        base_folder = _setup_base_folder(tmp_path)
        monkeypatch.setattr(p07, "get_metadata_filtered", _fake_metadata)
        monkeypatch.setattr(
            p07, "get_interference_per_pair_inverse",
            lambda *a, **k: {
                "Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
                "Bob": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
            },
        )

        def _fake_get_interference_per_pair(task: str, index: int, *a: Any, **k: Any) -> Dict[str, Any]:
            if index == 1:
                raise RuntimeError("simulated corrupt per-pair file")
            return {"Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0}}

        monkeypatch.setattr(p07, "get_interference_per_pair", _fake_get_interference_per_pair)

        # Remove entity 0's per-pair file so a third status (skipped) is exercised too, via a
        # second (method, epochs) combination.
        ledger_path = os.path.join(base_folder, "ledger.jsonl")
        ledger = RunLedger(ledger_path)

        p07.compute_for_task(
            task=_TASK, methods=[_METHOD], num_train_epochs_list=[_EPOCHS],
            index_start=0, max_identities=len(_ENTITIES),
            embedding_assets_folder=os.path.join(base_folder, "datasets"),
            base_folder=base_folder,
            ledger=ledger,
        )
        ledger.close()

        with open(ledger_path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        statuses = {r["context"]: r["status"] for r in records}
        assert statuses[f"InterferencePerEntity for {_TASK}/{_METHOD}/{_EPOCHS}/index=0"] == "ok"
        assert statuses[f"InterferencePerEntity for {_TASK}/{_METHOD}/{_EPOCHS}/index=1"] == "failed"
        failed_record = next(r for r in records if r["status"] == "failed")
        assert failed_record["exception_type"] == "RuntimeError"
        assert failed_record["message"] == "simulated corrupt per-pair file"

    def test_missing_per_pair_file_is_recorded_as_skipped(self, tmp_path: Any, monkeypatch: Any) -> None:
        base_folder = _setup_base_folder(tmp_path)
        monkeypatch.setattr(p07, "get_metadata_filtered", _fake_metadata)

        # Delete entity 1's per-pair file so the pre-existing skip branch fires.
        missing_path = p07.get_interference_per_pair_path(_TASK, 1, _METHOD, _EPOCHS, base_folder=base_folder)
        os.remove(missing_path)

        def _fake_get_interference_per_pair(task: str, index: int, *a: Any, **k: Any) -> Dict[str, Any]:
            return {"Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0}}

        monkeypatch.setattr(p07, "get_interference_per_pair", _fake_get_interference_per_pair)
        monkeypatch.setattr(
            p07, "get_interference_per_pair_inverse",
            lambda *a, **k: {
                "Alice": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
                "Bob": {"brisque_diff": 0.0, "clip_diff": 0.0, "rmse": 0.0, "ssim": 1.0},
            },
        )

        ledger_path = os.path.join(base_folder, "ledger.jsonl")
        ledger = RunLedger(ledger_path)

        p07.compute_for_task(
            task=_TASK, methods=[_METHOD], num_train_epochs_list=[_EPOCHS],
            index_start=0, max_identities=len(_ENTITIES),
            embedding_assets_folder=os.path.join(base_folder, "datasets"),
            base_folder=base_folder,
            ledger=ledger,
        )
        ledger.close()

        with open(ledger_path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        statuses = {r["context"]: r["status"] for r in records}
        assert statuses[f"InterferencePerEntity for {_TASK}/{_METHOD}/{_EPOCHS}/index=1"] == "skipped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
