"""Append-only execution ledgers: ``pipeline_08_run_all_rts.py``'s, and a campaign's.

Every attempted Result Template computation (skipped / succeeded / failed) is written as
one JSON line, immediately, to a local log file. This is a debugging and traceability aid
only: it records what a run actually did, and why a given attempt failed, which is
otherwise only visible in transient stdout / process logs. It is not a Result Template, it
produces no data any RT reads, and it is not one of the artifacts described in the paper --
see the module docstring of ``pipeline_08_run_all_rts.py`` for the actual RT catalogue.

The same append-only mechanism records a data-producing campaign: each attempt of each session is
one line whose ``fields`` hold what makes it checkable -- the session key, the attempt number, the
commit and the hashes of the run's ``pins.json`` and environment freeze, the card, the digests of
the outputs it wrote, its counts and pairing verdict, and whether it was a calibration run. Failed
attempts stay in the file; an attempt whose output was later replaced is marked by a ``superseded``
record, never deleted. :func:`check` reads such a ledger and reports everything that would make the
campaign's record untrustworthy; ``python -m vision_unlearning.benchmarks.I_care.run_ledger check
--help`` runs it.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import sys
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, TextIO


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
        fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write one ledger line.

        Args:
            context: human-readable description of what was attempted -- the Result
                Template name followed by " for " and its parameters, e.g.
                "MetricMetricAlignment for sd1.4/people/distil/clip_diff/rmse".
            status: one of "ok", "skipped", "failed", and for a campaign also "superseded" and
                "drift".
            exception_type: exception class name; only set when ``status == "failed"``.
            message: exception message; only set when ``status == "failed"``.
            fields: structured fields of a campaign record (see :func:`check`); written under
                ``"fields"`` only when given, so a record without them is exactly as before.
        """
        record: Dict[str, Any] = {
            "context": context,
            "status": status,
            "exception_type": exception_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit_sha": self._commit_sha,
        }
        if fields is not None:
            record["fields"] = fields
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


# ---------------------------------------------------------------------------
# The campaign check
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(ledger_path: str) -> List[Dict[str, Any]]:
    """Every record of a ledger, in file order."""
    with open(ledger_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(
    ledger_path: str,
    expected_sessions: Iterable[str],
    base_folder: Optional[str] = None,
) -> List[str]:
    """Every reason a campaign ledger cannot be trusted; an empty list means it can.

    A campaign record is one with ``fields``; ``fields["session"]`` names its session and
    ``fields["attempt"]`` numbers its attempt. Reported:

    * **an expected session with no live ok attempt** -- live meaning not superseded;
    * **two live ok attempts of one session whose output digests differ** -- a session ran twice and
      the record does not say which output stands; supersede one;
    * **a live ok output that no longer matches its digest**, when `base_folder` is given:
      ``fields["outputs"]`` maps paths relative to it to sha256, and every file is re-hashed;
    * **a live ok attempt whose commit, pins.json hash or freeze hash differs** from the campaign's
      first live ok attempt without a ``drift`` record admitting it. A drift record's fields hold the
      ``"commit"``, ``"pins_sha256"`` and ``"freeze_sha256"`` it admits, and the reason.

    A ``superseded`` record's ``fields["supersedes"]`` is the attempt it retires, of the same
    session.
    """
    records = [r for r in read_records(ledger_path) if "fields" in r]
    problems: List[str] = []
    retired = {
        (r["fields"]["session"], r["fields"]["supersedes"])
        for r in records if r["status"] == "superseded"
    }
    admitted: Dict[str, set] = {"commit": set(), "pins_sha256": set(), "freeze_sha256": set()}
    for r in records:
        if r["status"] == "drift":
            for key in admitted:
                if r["fields"].get(key) is not None:
                    admitted[key].add(r["fields"][key])

    live: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        f = r["fields"]
        if r["status"] == "ok" and (f["session"], f["attempt"]) not in retired:
            live[f["session"]].append(r)

    for session in expected_sessions:
        if not live.get(session):
            problems.append(f"missing: {session} has no live ok attempt")

    reference: Optional[Dict[str, Any]] = None
    for session, attempts in sorted(live.items()):
        digests = {json.dumps(a["fields"].get("outputs", {}), sort_keys=True) for a in attempts}
        if len(digests) > 1:
            numbers = [a["fields"]["attempt"] for a in attempts]
            problems.append(f"duplicate: {session} has {len(attempts)} live ok attempts {numbers} with different outputs")
        for attempt in attempts:
            f = attempt["fields"]
            if reference is None:
                reference = f
            for key in ("commit", "pins_sha256", "freeze_sha256"):
                if f.get(key) != reference.get(key) and f.get(key) not in admitted[key]:
                    problems.append(
                        f"drift: {session} attempt {f['attempt']} ran with {key} {f.get(key)} against the "
                        f"campaign's {reference.get(key)}, and no drift record admits it"
                    )
            if base_folder is not None:
                for relative, expected in sorted(f.get("outputs", {}).items()):
                    path = os.path.join(base_folder, relative)
                    if not os.path.isfile(path):
                        problems.append(f"stale: {session} attempt {f['attempt']} output {relative} is gone")
                    elif _sha256(path) != expected:
                        problems.append(f"stale: {session} attempt {f['attempt']} output {relative} no longer matches its digest")
    return problems


#: The routes whose outputs must equal the baseline's for a drift to be admitted; the short-final-batch
#: route is held to pairing only, its last batch having another shape than the baseline's.
DRIFT_EXACT_ROUTES = ('route_partial_weights', 'route_adapter')
DRIFT_PAIRED_ROUTES = DRIFT_EXACT_ROUTES + ('route_partial_weights_short',)


def admit_drift(
    ledger_path: str,
    commit: str,
    reason: str,
    routes_path: str,
    pins_sha256: Optional[str] = None,
    freeze_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a ``drift`` record admitting `commit` (and new pinned inputs, if given) into a campaign.

    The evidence is the route-agreement result (``ablations/regeneration_rehearsal/route_agreement.py``)
    run at that commit against the campaign's baseline: every route in :data:`DRIFT_PAIRED_ROUTES` is
    paired, and every route in :data:`DRIFT_EXACT_ROUTES` reproduces the baseline's images exactly.
    Raises ValueError, writing nothing, otherwise. Returns the record's fields.
    """
    with open(routes_path, encoding="utf-8") as handle:
        routes = json.load(handle)
    problems = [f"{route}: {routes.get(route, {}).get('pairing', 'absent')}" for route in DRIFT_PAIRED_ROUTES
                if routes.get(route, {}).get("pairing") != "paired"]
    problems += [f"{route}: mean absolute difference {routes[route].get('mean_abs')} to the baseline" for route in DRIFT_EXACT_ROUTES
                 if route in routes and routes[route].get("mean_abs") != 0.0]
    if problems:
        raise ValueError(f"{commit} cannot be admitted: {'; '.join(problems)}")
    fields: Dict[str, Any] = {
        "session": "campaign", "attempt": 0, "commit": commit, "pins_sha256": pins_sha256,
        "freeze_sha256": freeze_sha256, "reason": reason, "routes_sha256": _sha256(routes_path),
        "routes": {route: {"pairing": routes[route]["pairing"], "mean_abs": routes[route].get("mean_abs")}
                   for route in DRIFT_PAIRED_ROUTES},
    }
    ledger = RunLedger(ledger_path)
    ledger.record(f"drift admitted: {commit}", "drift", fields=fields)
    ledger.close()
    return fields


def _main(argv: Optional[List[str]] = None) -> int:
    description = (
        "check: read a campaign ledger and report every reason its record cannot be trusted -- a "
        "missing expected session, a session with two live ok attempts whose outputs differ, an "
        "output that no longer matches its recorded sha256, or an attempt run at another commit or "
        "with other pinned inputs than the campaign's without a drift record. Prints one line per "
        "problem, then a one-line summary. Exit status: 0 nothing found, 1 problems found, 2 bad "
        "arguments."
    )
    parser = argparse.ArgumentParser(prog="python -m vision_unlearning.benchmarks.I_care.run_ledger", description=description)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("check", help="check a campaign ledger", description=description)
    run.add_argument("--ledger", required=True, help="the ledger file (JSON lines)")
    run.add_argument("--expected", required=True,
                     help="a file with one expected session key per line, spelled as fields['session'] is")
    run.add_argument("--base-folder", default=None,
                     help="re-hash every live ok output below this folder; omitted, outputs are not re-hashed")
    drift = sub.add_parser(
        "drift", help="admit a later commit or pinned inputs into a campaign",
        description="Append a drift record admitting a commit (and, if given, new pins.json and freeze hashes), "
                    "only when the route-agreement result run at that commit shows every route paired and the "
                    "full-batch routes reproducing the baseline exactly. Exit status: 0 admitted, 1 refused "
                    "(nothing written), 2 bad arguments.")
    drift.add_argument("--ledger", required=True)
    drift.add_argument("--commit", required=True, help="the commit the campaign is to run from now on")
    drift.add_argument("--reason", required=True, help="why the commit changed, and what its diff touches")
    drift.add_argument("--routes", required=True, help="routes.json written by route_agreement.py at that commit")
    drift.add_argument("--pins-sha256", default=None)
    drift.add_argument("--freeze-sha256", default=None)
    args = parser.parse_args(argv)
    if args.command == "drift":
        try:
            fields = admit_drift(args.ledger, args.commit, args.reason, args.routes, args.pins_sha256, args.freeze_sha256)
        except ValueError as exc:
            print(f"refused: {exc}")
            return 1
        print(f"admitted: {fields['commit']} ({json.dumps(fields['routes'])})")
        return 0
    with open(args.expected, encoding="utf-8") as handle:
        expected = [line.strip() for line in handle if line.strip()]
    problems = check(args.ledger, expected, args.base_folder)
    for problem in problems:
        print(problem)
    records = [r for r in read_records(args.ledger) if "fields" in r]
    statuses = Counter(r["status"] for r in records)
    print(f"{len(records)} campaign records {dict(sorted(statuses.items()))}; {len(expected)} expected sessions; "
          f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(_main())
