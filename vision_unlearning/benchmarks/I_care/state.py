"""State-of-the-world census for the I-CARE benchmark.

Answers "what is computed / uploaded / valid?" cheaply, by reading a record instead of
re-deriving it by brute force. It replaces two habits that keep producing wrong answers:
a per-file HuggingFace probe over the whole repository (minutes, rate-limited), and a
hand-typed coverage note that drifts from the truth.

How it stays honest (see ``CONTRIBUTING_ICARE.md`` section 12):

- **The expected set is generated, not hand-listed.** It is the cartesian product of the
  declared ontology (``configuration.py``: tasks x methods x entities, plus the per-task and
  per-metric artifacts), resolved through the real ``Artifact`` classes. Nothing here
  re-implements a path or a filename — every path comes from the same artifact object the
  pipeline itself resolves, so the census cannot drift from the code that writes the files.
- **Local presence is ``os.path.exists`` (instant); remote presence is a membership test
  against a cached full-repository listing** — ONE ``list_repo_files`` call, cached to disk
  with its date, paid once. Every subsequent query is a set lookup, never a per-file network
  round trip. This is the whole point: the expensive probe becomes a record.
- **Content validity runs each artifact's own ``_validate``** over the locally present file.
  This is the only check that distinguishes "present" from "valid" — e.g. an embedding file
  that exists but carries a truncated entity label. Presence alone is structurally blind to
  that class of damage.

This module is a benchmark *auditor*, not a data consumer: it deliberately checks presence
and validity through the artifacts' non-resolving audit API (``exists_local`` /
``is_in_listing`` / ``validate_local``) instead of ``.compute()``/``.exists()``, because
resolving each of ~1800 cells would be exactly the network storm it exists to replace. For
the same reason it is intentionally NOT in ``tests/test_artifact_discipline.py``'s scanned
library list — the "always resolve through the cascade" rule is for data consumers.

Command-line usage (from the I-CARE package directory)::

    python state.py refresh-remote            # one HuggingFace listing -> local cache
    python state.py report --remote --validate
    python state.py query --kind entity_embeddings --task scenes --missing-remote
    python state.py state-md                  # regenerate the committed STATE.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from vision_unlearning.artifact import (
    DEFAULT_REMOTE_REPOSITORY_NAME,
    SingleFileArtifact,
)
from vision_unlearning.integrations.huggingface import (
    get_hf_token_from_env,
    huggingface_dataset_list_files,
)
from vision_unlearning.benchmarks.I_care.configuration import (
    type_model,
    type_s,
    type_task,
    type_unlearning_algorithm,
    unlearning_algorithm_to_epochs,
)
from vision_unlearning.benchmarks.I_care.metadata import (
    BaselineEmbeddings,
    EntityEmbeddings,
    InterferencePerEntity,
    InterferencePerPair,
)
from vision_unlearning.benchmarks.I_care.similarity import Similarity
from vision_unlearning.datasets.testbed import MetadataFiltered, get_target_overwrite

# The ontology dimensions the census enumerates over. Kept as explicit lists (rather than
# derived from get_args) so the file reads as the grid it is; a test locks them against the
# configuration Literals so they cannot silently drift.
TASKS: List[type_task] = ["people", "scenes", "breeds"]
METHODS: List[type_unlearning_algorithm] = ["distil", "munba", "uce"]
# weight_overlap is excluded on purpose: it is a scenes/distil-only diagnostic with no GUI
# display name, never produced as a canonical similarity_{s}_{task}.json for every task.
SIMILARITY_METRICS: List[type_s] = ["clip", "jacc", "dino", "act"]

_STATE_DIR_NAME = "state"
_LISTING_CACHE_FILENAME = "hf_listing.json"
_CENSUS_CACHE_FILENAME = "census.json"


@dataclass
class ExpectedArtifact:
    """One cell of the expected grid, bound to the real artifact object that addresses it."""

    kind: str
    task: str
    method: Optional[str]
    entity: Optional[str]
    index: Optional[int]
    similarity_metric: Optional[str]
    artifact: SingleFileArtifact


@dataclass
class ArtifactState:
    """The audited state of one expected artifact. Serialised into the census JSON."""

    kind: str
    task: str
    method: Optional[str]
    entity: Optional[str]
    index: Optional[int]
    similarity_metric: Optional[str]
    local_path: str
    remote_path: str
    present_local: bool
    present_remote: Optional[bool]  # None when no remote listing was supplied
    valid: Optional[bool]  # None when not checked or not present locally
    validation_error: Optional[str]


def expected_artifacts(
    base_folder: str = "assets",
    model: type_model = "sd1.4",
) -> List[ExpectedArtifact]:
    """Enumerate every JSON artifact the I-CARE demonstration is expected to contain.

    The per-entity grid (embeddings, per-pair interference) is enumerated from the task's
    ``MetadataFiltered`` — the same ordered entity list, and therefore the same emitter
    ``index``, that the pipeline itself uses — so the expected set matches what the pipeline
    would produce rather than a parallel hand-list. When a task's metadata is not available
    locally its per-entity grid cannot be enumerated; the metadata row itself still reports
    as missing, which is the honest signal that the task cannot be censused yet.
    """
    out: List[ExpectedArtifact] = []
    for task in TASKS:
        out.append(ExpectedArtifact(
            "metadata", task, None, None, None, None,
            MetadataFiltered(task=task, base_folder=base_folder),
        ))
        out.append(ExpectedArtifact(
            "baseline_embeddings", task, None, None, None, None,
            BaselineEmbeddings(task=task, base_folder=base_folder, model=model),
        ))
        out.append(ExpectedArtifact(
            "interference_per_entity", task, None, None, None, None,
            InterferencePerEntity(task=task, base_folder=base_folder, model=model),
        ))
        for similarity_metric in SIMILARITY_METRICS:
            out.append(ExpectedArtifact(
                "similarity", task, None, None, None, similarity_metric,
                Similarity(
                    task=task, similarity_metric=similarity_metric,
                    base_folder=base_folder, model=model,
                ),
            ))

        metadata_artifact = MetadataFiltered(task=task, base_folder=base_folder)
        if not metadata_artifact.exists_local():
            # Cannot enumerate the per-entity grid without the entity list; the metadata row
            # above already reports the gap.
            continue
        names = [str(entry["name"]) for entry in metadata_artifact.compute()]
        for method in METHODS:
            epochs = unlearning_algorithm_to_epochs[task][method]
            for index, name in enumerate(names):
                hf_entity = get_target_overwrite(task, method, name)[0]
                out.append(ExpectedArtifact(
                    "entity_embeddings", task, method, name, index, None,
                    EntityEmbeddings(
                        task=task, hf_entity=hf_entity, unlearning_algorithm=method,
                        base_folder=base_folder, model=model,
                    ),
                ))
                out.append(ExpectedArtifact(
                    "interference_per_pair", task, method, name, index, None,
                    InterferencePerPair(
                        task=task, index=index, method=method, num_train_epochs=epochs,
                        base_folder=base_folder, model=model,
                    ),
                ))
    return out


def build_census(
    base_folder: str = "assets",
    model: type_model = "sd1.4",
    remote_listing: Optional[Set[str]] = None,
    validate: bool = False,
) -> List[ArtifactState]:
    """Audit every expected artifact for local presence, remote presence, and validity.

    ``remote_listing`` is a set of repository-relative file paths (from
    :func:`load_cached_listing`); when ``None`` the remote column is left unknown. ``validate``
    additionally loads each locally present file and runs the artifact's own ``_validate``.
    """
    rows: List[ArtifactState] = []
    for expected in expected_artifacts(base_folder=base_folder, model=model):
        artifact = expected.artifact
        present_local = artifact.exists_local()
        present_remote: Optional[bool] = (
            artifact.is_in_listing(remote_listing) if remote_listing is not None else None
        )
        valid: Optional[bool] = None
        validation_error: Optional[str] = None
        if validate and present_local:
            try:
                artifact.validate_local()
                valid = True
            except Exception as exc:  # noqa: BLE001 — a failed invariant is a reportable state, not a crash
                valid = False
                validation_error = f"{type(exc).__name__}: {exc}"
        rows.append(ArtifactState(
            kind=expected.kind,
            task=expected.task,
            method=expected.method,
            entity=expected.entity,
            index=expected.index,
            similarity_metric=expected.similarity_metric,
            # Read only to tell a human WHERE the file is/should be; the audit logic above
            # never opens these paths (that is what the artifact's own methods are for).
            local_path=artifact._get_data_path_local(),
            remote_path=artifact._get_data_path_remote(),
            present_local=present_local,
            present_remote=present_remote,
            valid=valid,
            validation_error=validation_error,
        ))
    return rows


# ---------------------------------------------------------------------------
# Remote listing cache — the one expensive probe, paid once and written down
# ---------------------------------------------------------------------------
def _state_dir(base_folder: str) -> str:
    return os.path.join(base_folder, _STATE_DIR_NAME)


def _listing_cache_path(base_folder: str) -> str:
    return os.path.join(_state_dir(base_folder), _LISTING_CACHE_FILENAME)


def refresh_remote_listing(
    base_folder: str = "assets",
    remote_repository_name: str = DEFAULT_REMOTE_REPOSITORY_NAME,
) -> str:
    """List the whole HuggingFace repository once and cache it (with its date) to disk.

    Returns the cache path. This is the only place the census touches the network, and it is
    the deliberate "brute force, but only to build the record" exception: run it when the
    remote has changed, then every query reads the cache.
    """
    files = huggingface_dataset_list_files(remote_repository_name, get_hf_token_from_env())
    payload = {
        "remote_repository_name": remote_repository_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "files": sorted(files),
    }
    cache_path = _listing_cache_path(base_folder)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return cache_path


def load_cached_listing(base_folder: str = "assets") -> Tuple[Set[str], str]:
    """Return the cached remote listing as a set, plus the date it was fetched.

    Raises ``FileNotFoundError`` (with a clear instruction) when the cache is absent, so a
    caller never silently treats "no cache" as "nothing on the remote".
    """
    cache_path = _listing_cache_path(base_folder)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No remote-listing cache at {cache_path}. Run `python state.py refresh-remote` "
            "first (one HuggingFace listing, then cached)."
        )
    with open(cache_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return set(payload["files"]), str(payload.get("fetched_at", "unknown"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
@dataclass
class _GroupCounts:
    expected: int = 0
    local: int = 0
    remote: int = 0
    valid: int = 0
    invalid: int = 0


def _group(rows: List[ArtifactState]) -> Dict[Tuple[str, str, str], _GroupCounts]:
    """Aggregate rows by (kind, task, method-or-similarity-metric)."""
    groups: Dict[Tuple[str, str, str], _GroupCounts] = defaultdict(_GroupCounts)
    for row in rows:
        subdimension = row.method or row.similarity_metric or "-"
        counts = groups[(row.kind, row.task, subdimension)]
        counts.expected += 1
        if row.present_local:
            counts.local += 1
        if row.present_remote:
            counts.remote += 1
        if row.valid is True:
            counts.valid += 1
        if row.valid is False:
            counts.invalid += 1
    return groups


def _summary_row(
    kind: str, task: str, subdimension: str, local_cell: str,
    remote_cell: Optional[str], invalid_cell: Optional[str],
) -> str:
    line = f"{kind:<24}{task:<8}{subdimension:<8}{local_cell:>11}"
    if remote_cell is not None:
        line += f" {remote_cell:>11}"
    if invalid_cell is not None:
        line += f" {invalid_cell:>8}"
    return line


def summarize_text(rows: List[ArtifactState], remote_known: bool, validated: bool) -> str:
    """One-screen coverage summary grouped by (kind, task, sub-dimension)."""
    groups = _group(rows)
    header = _summary_row(
        "kind", "task", "sub", "local",
        "remote" if remote_known else None,
        "invalid" if validated else None,
    )
    lines: List[str] = [header, "-" * len(header)]
    for (kind, task, subdimension) in sorted(groups):
        counts = groups[(kind, task, subdimension)]
        lines.append(_summary_row(
            kind, task, subdimension,
            f"{counts.local}/{counts.expected}",
            f"{counts.remote}/{counts.expected}" if remote_known else None,
            str(counts.invalid) if validated else None,
        ))
    totals = _totals(rows)
    lines.append("-" * len(header))
    lines.append(
        f"TOTAL expected={totals['expected']}  local={totals['local']}"
        + (f"  remote={totals['remote']}" if remote_known else "")
        + (f"  invalid={totals['invalid']}" if validated else "")
    )
    return "\n".join(lines)


def _totals(rows: List[ArtifactState]) -> Dict[str, int]:
    return {
        "expected": len(rows),
        "local": sum(1 for r in rows if r.present_local),
        "remote": sum(1 for r in rows if r.present_remote),
        "invalid": sum(1 for r in rows if r.valid is False),
    }


def census_to_json(rows: List[ArtifactState]) -> List[Dict[str, Any]]:
    return [asdict(row) for row in rows]


def generate_state_md(
    rows: List[ArtifactState],
    remote_known: bool,
    validated: bool,
    listing_date: Optional[str],
    base_folder: str,
) -> str:
    """Render the committed one-screen STATE.md: pointers and coverage, no narrative."""
    groups = _group(rows)
    totals = _totals(rows)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: List[str] = []
    out.append("# I-CARE — state of the world")
    out.append("")
    out.append(
        "Generated record. Do not hand-edit — regenerate with "
        "`python state.py state-md` (from the I-CARE package directory)."
    )
    out.append("")
    out.append(f"- Generated: `{now}`")
    out.append(f"- Local base folder: `{base_folder}`")
    if remote_known:
        out.append(f"- Remote listing fetched: `{listing_date}` "
                   "(refresh with `python state.py refresh-remote`)")
    else:
        out.append("- Remote listing: not fetched (run `python state.py refresh-remote`)")
    out.append(
        f"- Totals: expected **{totals['expected']}**, local **{totals['local']}**"
        + (f", remote **{totals['remote']}**" if remote_known else "")
        + (f", invalid **{totals['invalid']}**" if validated else "")
    )
    out.append("")
    header = "| kind | task | sub | local | " + ("remote | " if remote_known else "") \
        + ("invalid |" if validated else "")
    sep = "|------|------|-----|-------|" + ("--------|" if remote_known else "") \
        + ("---------|" if validated else "")
    out.append(header)
    out.append(sep)
    for (kind, task, subdimension) in sorted(groups):
        counts = groups[(kind, task, subdimension)]
        row = f"| {kind} | {task} | {subdimension} | {counts.local}/{counts.expected} | "
        if remote_known:
            row += f"{counts.remote}/{counts.expected} | "
        if validated:
            row += f"{counts.invalid} |"
        out.append(row)
    out.append("")
    gaps = _known_gaps(groups, remote_known, validated)
    out.append("## Known gaps")
    out.append("")
    if gaps:
        out.extend(gaps)
    else:
        out.append("None detected at the audited granularity.")
    out.append("")
    return "\n".join(out)


def _known_gaps(
    groups: Dict[Tuple[str, str, str], _GroupCounts],
    remote_known: bool,
    validated: bool,
) -> List[str]:
    """Bullet each (kind, task, sub) that is short on the shared-truth axis (remote, else local)."""
    gaps: List[str] = []
    for (kind, task, subdimension) in sorted(groups):
        counts = groups[(kind, task, subdimension)]
        have = counts.remote if remote_known else counts.local
        axis = "remote" if remote_known else "local"
        if have < counts.expected:
            gaps.append(f"- `{kind}` / {task} / {subdimension}: {have}/{counts.expected} {axis}")
        if validated and counts.invalid:
            gaps.append(f"- `{kind}` / {task} / {subdimension}: **{counts.invalid} invalid** (present but failed _validate)")
    return gaps


def filter_rows(
    rows: List[ArtifactState],
    kind: Optional[str] = None,
    task: Optional[str] = None,
    method: Optional[str] = None,
    missing_local: bool = False,
    missing_remote: bool = False,
    invalid_only: bool = False,
) -> List[ArtifactState]:
    """Return the subset of rows matching the given filters (used by the ``query`` command)."""
    result: List[ArtifactState] = []
    for row in rows:
        if kind is not None and row.kind != kind:
            continue
        if task is not None and row.task != task:
            continue
        if method is not None and row.method != method:
            continue
        if missing_local and row.present_local:
            continue
        if missing_remote and row.present_remote is not False:
            continue
        if invalid_only and row.valid is not False:
            continue
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _default_state_md_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "STATE.md")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-folder", default="assets", help="Local assets folder (default: assets).")
    parser.add_argument(
        "--repo", default=DEFAULT_REMOTE_REPOSITORY_NAME,
        help="HuggingFace dataset repository to census against.",
    )


def _load_listing_or_none(base_folder: str, use_remote: bool) -> Tuple[Optional[Set[str]], Optional[str]]:
    if not use_remote:
        return None, None
    listing, date = load_cached_listing(base_folder)
    return listing, date


def _cmd_refresh_remote(args: argparse.Namespace) -> int:
    cache_path = refresh_remote_listing(base_folder=args.base_folder, remote_repository_name=args.repo)
    listing, date = load_cached_listing(args.base_folder)
    print(f"Cached {len(listing)} remote paths to {cache_path} (fetched {date}).")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    listing, _ = _load_listing_or_none(args.base_folder, args.remote)
    rows = build_census(base_folder=args.base_folder, remote_listing=listing, validate=args.validate)
    print(summarize_text(rows, remote_known=args.remote, validated=args.validate))
    if args.write_json:
        cache = os.path.join(_state_dir(args.base_folder), _CENSUS_CACHE_FILENAME)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(census_to_json(rows), handle)
        print(f"\nCensus JSON written to {cache}")
    return 0


def _cmd_state_md(args: argparse.Namespace) -> int:
    listing: Optional[Set[str]]
    date: Optional[str]
    try:
        listing, date = load_cached_listing(args.base_folder)
        remote_known = True
    except FileNotFoundError:
        listing, date, remote_known = None, None, False
    rows = build_census(base_folder=args.base_folder, remote_listing=listing, validate=args.validate)
    text = generate_state_md(
        rows, remote_known=remote_known, validated=args.validate,
        listing_date=date, base_folder=args.base_folder,
    )
    output = args.output or _default_state_md_path()
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Wrote {output}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    listing, _ = _load_listing_or_none(args.base_folder, args.remote or args.missing_remote)
    rows = build_census(
        base_folder=args.base_folder, remote_listing=listing,
        validate=args.validate or args.invalid_only,
    )
    matches = filter_rows(
        rows, kind=args.kind, task=args.task, method=args.method,
        missing_local=args.missing_local, missing_remote=args.missing_remote,
        invalid_only=args.invalid_only,
    )
    for row in matches:
        marks: List[str] = ["local" if row.present_local else "NO-local"]
        if row.present_remote is not None:
            marks.append("remote" if row.present_remote else "NO-remote")
        if row.valid is False:
            marks.append(f"INVALID({row.validation_error})")
        print(f"{row.kind:<22} {row.task:<7} {str(row.method or row.similarity_metric or '-'):<7} "
              f"{str(row.entity or ''):<28} {'  '.join(marks)}  -> {row.remote_path}")
    print(f"\n{len(matches)} match(es).")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser("refresh-remote", help="List the HuggingFace repo once and cache it.")
    _add_common(p_refresh)

    p_report = sub.add_parser("report", help="Print a coverage summary; optionally write the census JSON.")
    _add_common(p_report)
    p_report.add_argument("--remote", action="store_true", help="Include the remote column (needs a cached listing).")
    p_report.add_argument("--validate", action="store_true", help="Also run each artifact's _validate over local files.")
    p_report.add_argument("--write-json", action="store_true", help="Write the full census to the state cache.")

    p_state = sub.add_parser("state-md", help="Regenerate the committed STATE.md.")
    _add_common(p_state)
    p_state.add_argument("--validate", action="store_true", help="Include the invalid column (slower).")
    p_state.add_argument("--output", default=None, help="Output path (default: STATE.md next to this module).")

    p_query = sub.add_parser("query", help="Print artifacts matching filters.")
    _add_common(p_query)
    p_query.add_argument("--kind")
    p_query.add_argument("--task")
    p_query.add_argument("--method")
    p_query.add_argument("--remote", action="store_true", help="Resolve remote presence (needs a cached listing).")
    p_query.add_argument("--validate", action="store_true")
    p_query.add_argument("--missing-local", action="store_true")
    p_query.add_argument("--missing-remote", action="store_true")
    p_query.add_argument("--invalid-only", action="store_true")

    args = parser.parse_args(argv)

    dispatch: Dict[str, Callable[[argparse.Namespace], int]] = {
        "refresh-remote": _cmd_refresh_remote,
        "report": _cmd_report,
        "state-md": _cmd_state_md,
        "query": _cmd_query,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
