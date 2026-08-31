"""Tests for the pipeline_08 execution ledger (pure logic -- no network, no torch)."""
import json
from typing import Any

from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger, summarize


class TestRunLedger:
    def test_record_writes_one_json_line_per_call(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        ledger.record("RTOne for a/b/c", status="ok")
        ledger.record("RTOne for a/b/d", status="skipped")
        ledger.close()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 2
        assert lines[0]["status"] == "ok"
        assert lines[0]["context"] == "RTOne for a/b/c"
        assert lines[1]["status"] == "skipped"

    def test_record_includes_timestamp_and_commit_sha(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        ledger.record("RTOne for a/b/c", status="ok")
        ledger.close()

        with open(path, "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert "timestamp" in record and record["timestamp"]
        assert "commit_sha" in record and record["commit_sha"]

    def test_failed_status_carries_exception_type_and_message(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        ledger.record(
            "RTOne for a/b/c", status="failed",
            exception_type="ValueError", message="bad input",
        )
        ledger.close()

        with open(path, "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert record["exception_type"] == "ValueError"
        assert record["message"] == "bad input"

    def test_count_property_tracks_number_of_records(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        assert ledger.count == 0
        ledger.record("RTOne for a", status="ok")
        ledger.record("RTOne for b", status="ok")
        assert ledger.count == 2
        ledger.close()

    def test_creates_parent_directory_if_missing(self, tmp_path: Any) -> None:
        path = str(tmp_path / "nested" / "dir" / "ledger.jsonl")
        ledger = RunLedger(path)
        ledger.record("RTOne for a", status="ok")
        ledger.close()
        assert (tmp_path / "nested" / "dir" / "ledger.jsonl").exists()

    def test_appends_across_multiple_ledger_instances(self, tmp_path: Any) -> None:
        """A ledger opened twice (e.g. two pipeline_08 invocations) accumulates, never
        overwrites -- this is what makes the ledger a true run history, not a snapshot."""
        path = str(tmp_path / "ledger.jsonl")
        first = RunLedger(path)
        first.record("RTOne for a", status="ok")
        first.close()

        second = RunLedger(path)
        second.record("RTOne for b", status="ok")
        second.close()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 2


class TestSummarize:
    def test_missing_ledger_reports_absence_not_an_error(self, tmp_path: Any) -> None:
        result = summarize(str(tmp_path / "does_not_exist.jsonl"))
        assert "No ledger found" in result

    def test_groups_counts_by_rt_name_and_status(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        ledger.record("RTOne for a/b/c", status="ok")
        ledger.record("RTOne for a/b/d", status="ok")
        ledger.record("RTOne for a/b/e", status="skipped")
        ledger.record("RTTwo for x/y/z", status="failed", exception_type="ValueError", message="boom")
        ledger.close()

        result = summarize(path)
        assert "RTOne" in result
        assert "RTTwo" in result
        assert "Total records: 4" in result

    def test_groups_identical_failure_reasons_with_a_count(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        ledger = RunLedger(path)
        for _ in range(3):
            ledger.record(
                "RTOne for a/b/c", status="failed",
                exception_type="ValueError", message="same reason every time",
            )
        ledger.record(
            "RTOne for a/b/d", status="failed",
            exception_type="KeyError", message="a different reason",
        )
        ledger.close()

        result = summarize(path)
        assert "3x  ValueError: same reason every time" in result
        assert "1x  KeyError: a different reason" in result

    def test_reports_commit_shas_seen(self, tmp_path: Any) -> None:
        path = str(tmp_path / "ledger.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "context": "RTOne for a", "status": "ok", "exception_type": None,
                "message": None, "timestamp": "2026-01-01T00:00:00+00:00",
                "commit_sha": "abc123",
            }) + "\n")
        result = summarize(path)
        assert "abc123" in result
