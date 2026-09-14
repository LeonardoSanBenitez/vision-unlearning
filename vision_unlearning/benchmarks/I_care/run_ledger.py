"""Append-only execution ledger for ``pipeline_08_run_all_rts.py``.

Every attempted Result Template computation (skipped / succeeded / failed) is written as
one JSON line, immediately, to a local log file. This is a debugging and traceability aid
only: it records what a run actually did, and why a given attempt failed, which is
otherwise only visible in transient stdout / process logs. It is not a Result Template, it
produces no data any RT reads, and it is not one of the artifacts described in the paper --
see the module docstring of ``pipeline_08_run_all_rts.py`` for the actual RT catalogue.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import DefaultDict, Dict, Optional, TextIO


def get_commit_sha(repo_dir: Optional[str] = None) -> str:
    """Return the current git commit SHA, or ``"unknown"`` if it cannot be determined.

    A count of how many Result Templates ran, or how many failed, is not meaningful
    without knowing which version of the code produced it, so every ledger record is
    tagged with the commit the process was running under.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class RunLedger:
    """Appends one JSON record per Result-Template-computation attempt to ``path``.

    Each record is written and flushed immediately, so the ledger reflects real progress
    even if the process is interrupted mid-run -- a partially written ledger is still a
    truthful account of what happened before the interruption, unlike a summary that is
    only assembled in memory at the end of a run.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._commit_sha = get_commit_sha()
        self._count = 0
        self._file: TextIO = open(path, "a", encoding="utf-8")

    def record(
        self,
        context: str,
        status: str,
        exception_type: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Write one ledger line.

        Args:
            context: human-readable description of what was attempted -- the Result
                Template name followed by " for " and its parameters, e.g.
                "MetricMetricAlignment for sd1.4/people/distil/clip_diff/rmse".
            status: one of "ok", "skipped", "failed".
            exception_type: exception class name; only set when ``status == "failed"``.
            message: exception message; only set when ``status == "failed"``.
        """
        record = {
            "context": context,
            "status": status,
            "exception_type": exception_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit_sha": self._commit_sha,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        self._file.close()

    @property
    def count(self) -> int:
        return self._count


def summarize(ledger_path: str) -> str:
    """Read a ledger file and return a grouped, human-readable text summary.

    Groups record counts by (Result Template name, status) -- the name is the text before
    " for " in each record's ``context`` -- and, for failures, groups and counts identical
    (exception_type, message) pairs, so a failure that recurred across many parameter
    combinations shows up once with a count instead of as dozens of identical lines.
    """
    if not os.path.exists(ledger_path):
        return f"No ledger found at {ledger_path}."

    status_counts: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {"ok": 0, "skipped": 0, "failed": 0}
    )
    failure_reasons: Counter = Counter()
    commit_shas: set = set()
    total = 0

    with open(ledger_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            rt_name = record["context"].split(" for ", 1)[0]
            status = record["status"]
            if status in status_counts[rt_name]:
                status_counts[rt_name][status] += 1
            commit_shas.add(record.get("commit_sha", "unknown"))
            if status == "failed":
                reason = f"{record.get('exception_type')}: {record.get('message')}"
                failure_reasons[reason] += 1

    lines = [
        f"Run ledger summary: {ledger_path}",
        f"Total records: {total}",
        f"Commit SHA(s) seen: {sorted(commit_shas)}",
        "",
        f"{'Result Template':<45} {'ok':>6} {'skipped':>9} {'failed':>8}",
    ]
    for rt_name in sorted(status_counts):
        counts = status_counts[rt_name]
        lines.append(
            f"{rt_name:<45} {counts['ok']:>6} {counts['skipped']:>9} {counts['failed']:>8}"
        )

    if failure_reasons:
        lines.append("")
        lines.append("Failure reasons (grouped, most frequent first):")
        for reason, count in failure_reasons.most_common():
            lines.append(f"  {count:>4}x  {reason}")

    return "\n".join(lines)
