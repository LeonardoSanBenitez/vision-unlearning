"""pipeline_09 — Synchronize I-CARE artifacts with the HuggingFace dataset repository.

For each selected stage, enumerates the expected artifacts (task x method x entity),
checks local and remote existence, optionally uploads anything that is present locally
but absent remotely, and prints/saves a completion-status report.

Remote existence is resolved against a snapshot of the repository listing fetched once
at startup (a few paginated tree requests), not per-artifact HTTP checks — the full
artifact set is ~5,000 items and per-item requests would hit rate limits.

Stages (each maps to one artifact type; see the path table in CONTRIBUTING_ICARE.md §4):
    metadata                 metadata_{task}_2_enriched_filtered.json
    models                   models/{task}_{target}_{method}_{epochs:03d}/  (folder)
    generated-datasets       datasets/generated_{task}_{target_hf}_{method}_{epochs:03d}/  (folder)
    embeddings               datasets/embeddings_*.json  (baseline + per-entity)
    act-fingerprints         datasets/act_fingerprints_{task}_sd1.4.json
    interference-per-pair    datasets/interferences_caused_by_{task}_{index}_{method}_{epochs}.json
    interference-per-entity  interference_per_entity_{task}.json

Upload semantics: an artifact is uploaded only if it exists locally AND is absent
remotely. Nothing is ever deleted or overwritten remotely. Uploads are retried with
exponential backoff (rate limits); a failed item is recorded and the run continues.
Re-running is idempotent (a fresh remote snapshot is fetched each run).

Folder-type artifacts (models, generated datasets) are checked at folder granularity:
a folder present remotely counts as uploaded; per-file completeness inside remote
folders is not audited.

Usage
-----
    python pipeline_09_synchronize_huggingface.py --no-upload            # status report only
    python pipeline_09_synchronize_huggingface.py --stages embeddings act-fingerprints
    python pipeline_09_synchronize_huggingface.py --stages all --tasks people --methods distil
    python pipeline_09_synchronize_huggingface.py --help

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/ (with the
vision_unlearning package importable — installed, or PYTHONPATH set to the repo root).
Requires: no GPU; network access to huggingface.co. HF_TOKEN must be set in the
environment (or in a .env file, same lookup as pipeline_05) unless --no-upload is used
on a public repository.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, cast

import pandas as pd
from pydantic import BaseModel

from vision_unlearning.benchmarks.I_care.configuration import unlearning_algorithm_to_epochs
from vision_unlearning.benchmarks.I_care.metadata import (
    get_embedding_hf_path,
    get_embedding_output_path,
    get_interference_per_entity_path,
    get_interference_per_pair_path,
)
from vision_unlearning.datasets.testbed import (
    exists_unlearned_dataset,
    exists_unlearned_model,
    get_generated_dataset_folder,
    get_metadata_filtered,
    get_metadata_filtered_path,
    get_target_overwrite,
    get_unlearned_model_folder,
)
from vision_unlearning.integrations.huggingface import (
    huggingface_dataset_file_upload,
    huggingface_dataset_upload,
)
from vision_unlearning.utils.mechanistic_interpretability import get_act_fingerprints_path

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("synchronize_huggingface")


HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
SD_MODEL = "sd1.4"
GENERATE_DATASET_SEEDS = [42, 43, 44, 45]

_type_task = Literal["scenes", "objects", "breeds", "people"]
_type_method = Literal["munba", "uce", "distil"]

STAGES: List[str] = [
    "metadata",
    "models",
    "generated-datasets",
    "embeddings",
    "act-fingerprints",
    "interference-per-pair",
    "interference-per-entity",
]

ALL_TASKS: List[str] = ["breeds", "scenes", "people"]
ALL_METHODS: List[str] = ["uce", "munba", "distil"]


def to_hf_path(path: str) -> str:
    """Normalize a local-style relative path to an HF repo path (forward slashes)."""
    return path.replace(os.sep, "/")


class SyncItem(BaseModel):
    """One expected artifact: where it lives locally, where it belongs remotely, and its sync state."""
    stage: str
    task: str
    method: Optional[str] = None  # None for task-level (method-agnostic) artifacts
    name: str                     # entity name or artifact identifier (for reporting)
    kind: Literal["file", "folder"]
    local_path: str
    remote_path: str
    local_exists: bool = False
    remote_exists_before: bool = False
    uploaded_now: bool = False
    upload_error: Optional[str] = None

    @property
    def remote_exists_after(self) -> bool:
        return self.remote_exists_before or self.uploaded_now

    @property
    def missing_everywhere(self) -> bool:
        return not self.local_exists and not self.remote_exists_before


class RemoteSnapshot(BaseModel):
    """File and folder paths present in the remote repository (within the listed prefixes)."""
    files: Set[str] = set()
    folders: Set[str] = set()

    def exists(self, item: SyncItem) -> bool:
        if item.kind == "file":
            return item.remote_path in self.files
        return item.remote_path in self.folders


def fetch_remote_snapshot(
    repository: str,
    token: Optional[str],
    prefixes: Iterable[str] = ("", "datasets", "models"),
) -> RemoteSnapshot:
    """List the remote repository once per prefix (non-recursive, paginated).

    A prefix that does not exist remotely contributes nothing (no error).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=token)
    snapshot = RemoteSnapshot()
    for prefix in prefixes:
        try:
            n_before = len(snapshot.files) + len(snapshot.folders)
            for entry in api.list_repo_tree(
                repository,
                path_in_repo=prefix or None,
                repo_type="dataset",
                recursive=False,
            ):
                if isinstance(entry, RepoFile):
                    snapshot.files.add(entry.path)
                else:
                    snapshot.folders.add(entry.path)
            n_added = len(snapshot.files) + len(snapshot.folders) - n_before
            logger.info("Remote listing '%s/': %d entries", prefix, n_added)
        except EntryNotFoundError:
            logger.warning("Remote prefix not found (treated as empty): '%s/'", prefix)
    return snapshot


# ---------------------------------------------------------------------------
# Enumeration of expected artifacts (pure path logic — no filesystem, no network)
# ---------------------------------------------------------------------------

def enumerate_items(
    stages: List[str],
    tasks: List[str],
    methods: List[str],
    index_start: int,
    max_identities: int,
    base_folder: str,
    metadata_by_task: Dict[str, List[Dict[str, Any]]],
) -> List[SyncItem]:
    """Build the full expected-artifact list for the selected stages/tasks/methods.

    Epochs come from configuration.unlearning_algorithm_to_epochs. Entity names are
    resolved per the conventions in CONTRIBUTING_ICARE.md §6: models use the raw
    metadata name; generated datasets and embeddings use the HF-resolved name from
    get_target_overwrite()[0].
    """
    items: List[SyncItem] = []
    for task in tasks:
        task_t = cast(_type_task, task)
        entities = metadata_by_task[task]
        indices = range(index_start, min(index_start + max_identities, len(entities)))

        if "metadata" in stages:
            items.append(SyncItem(
                stage="metadata", task=task, name=f"metadata_{task}",
                kind="file",
                local_path=get_metadata_filtered_path(task_t, base_folder=base_folder),
                remote_path=to_hf_path(get_metadata_filtered_path(task_t, base_folder="")),
            ))

        if "act-fingerprints" in stages:
            items.append(SyncItem(
                stage="act-fingerprints", task=task, name=f"act_fingerprints_{task}",
                kind="file",
                local_path=get_act_fingerprints_path(task, SD_MODEL, base_folder=base_folder),
                remote_path=to_hf_path(get_act_fingerprints_path(task, SD_MODEL, base_folder="")),
            ))

        if "interference-per-entity" in stages:
            items.append(SyncItem(
                stage="interference-per-entity", task=task, name=f"interference_per_entity_{task}",
                kind="file",
                local_path=get_interference_per_entity_path(task_t, base_folder=base_folder),
                remote_path=to_hf_path(get_interference_per_entity_path(task_t, base_folder="")),
            ))

        if "embeddings" in stages:
            # Baseline embedding: method-agnostic (one per task), see pipeline_05 docstring.
            items.append(SyncItem(
                stage="embeddings", task=task, name="original",
                kind="file",
                local_path=get_embedding_output_path(task, "original", "", 0, base_folder=base_folder),
                remote_path=get_embedding_hf_path(task, "original", "", 0),
            ))

        for method in methods:
            method_t = cast(_type_method, method)
            epochs: int = unlearning_algorithm_to_epochs[task][method]

            for index in indices:
                target: str = entities[index]["name"]
                target_hf: str = get_target_overwrite(task_t, method_t, target)[0]

                if "models" in stages:
                    items.append(SyncItem(
                        stage="models", task=task, method=method, name=target,
                        kind="folder",
                        local_path=get_unlearned_model_folder(task_t, method_t, epochs, target, base_folder=base_folder),
                        remote_path=to_hf_path(get_unlearned_model_folder(task_t, method_t, epochs, target, base_folder="")),
                    ))

                if "generated-datasets" in stages:
                    items.append(SyncItem(
                        stage="generated-datasets", task=task, method=method, name=target_hf,
                        kind="folder",
                        local_path=get_generated_dataset_folder(task_t, method_t, epochs, target_hf, base_folder=base_folder),
                        remote_path=to_hf_path(get_generated_dataset_folder(task_t, method_t, epochs, target_hf, base_folder="")),
                    ))

                if "embeddings" in stages:
                    items.append(SyncItem(
                        stage="embeddings", task=task, method=method, name=target_hf,
                        kind="file",
                        local_path=get_embedding_output_path(task, target_hf, method, epochs, base_folder=base_folder),
                        remote_path=get_embedding_hf_path(task, target_hf, method, epochs),
                    ))

                if "interference-per-pair" in stages:
                    items.append(SyncItem(
                        stage="interference-per-pair", task=task, method=method, name=f"{task}_{index}",
                        kind="file",
                        local_path=get_interference_per_pair_path(task_t, index, method_t, epochs, base_folder=base_folder),
                        remote_path=to_hf_path(get_interference_per_pair_path(task_t, index, method_t, epochs, base_folder="")),
                    ))
    return items


# ---------------------------------------------------------------------------
# Local / remote existence annotation
# ---------------------------------------------------------------------------

def annotate_local_existence(
    items: List[SyncItem],
    metadata_by_task: Dict[str, List[Dict[str, Any]]],
    seeds: List[int],
    base_folder: str,
) -> None:
    """Set local_exists on every item, using the same checks as the pipeline scripts.

    models: weight file inside the folder (exists_unlearned_model semantics).
    generated-datasets: all expected on_* images present (exists_unlearned_dataset).
    everything else: plain file existence.
    """
    prompts_by_task: Dict[str, List[str]] = {}
    for item in items:
        if item.stage == "models":
            # exists_unlearned_model checks the method-specific weight filename.
            assert item.method is not None
            task_t = cast(_type_task, item.task)
            method_t = cast(_type_method, item.method)
            epochs = unlearning_algorithm_to_epochs[item.task][item.method]
            item.local_exists = exists_unlearned_model(task_t, method_t, epochs, item.name, base_folder=base_folder)
        elif item.stage == "generated-datasets":
            if item.task not in prompts_by_task:
                task_t = cast(_type_task, item.task)
                prompts_by_task[item.task] = [
                    f"An image of {get_target_overwrite(task_t, cast(_type_method, 'distil'), m['name'])[0]}"
                    for m in metadata_by_task[item.task]
                ]
            item.local_exists = exists_unlearned_dataset(item.local_path, seeds, prompts_by_task[item.task])
        else:
            item.local_exists = os.path.exists(item.local_path)


def annotate_remote_existence(items: List[SyncItem], snapshot: RemoteSnapshot) -> None:
    for item in items:
        item.remote_exists_before = snapshot.exists(item)


# ---------------------------------------------------------------------------
# Upload with retry
# ---------------------------------------------------------------------------

def upload_with_retry(
    upload_fn: Callable[[], None],
    description: str,
    max_retries: int,
    retry_base_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Optional[str]:
    """Call upload_fn, retrying with exponential backoff. Returns None on success,
    or the last error message after exhausting retries."""
    last_error = ""
    for attempt in range(max_retries):
        try:
            upload_fn()
            return None
        except Exception as exc:
            last_error = str(exc)
            wait = retry_base_seconds * (2 ** attempt)
            logger.warning(
                "Upload failed (%s), attempt %d/%d: %s",
                description, attempt + 1, max_retries, exc,
            )
            if attempt < max_retries - 1:
                logger.info("Retrying in %.0f seconds ...", wait)
                sleep_fn(wait)
    return last_error


def synchronize(
    items: List[SyncItem],
    snapshot: RemoteSnapshot,
    upload: bool,
    repository: str,
    token: str,
    base_folder: str,
    max_retries: int,
    retry_base_seconds: float,
    upload_file_fn: Callable[..., None] = huggingface_dataset_file_upload,
    upload_folder_fn: Callable[..., None] = huggingface_dataset_upload,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Upload every item that is local-only (present locally, absent remotely).

    Mutates items in place (uploaded_now / upload_error) and adds successful uploads
    to the snapshot so a re-enumerated item in the same run sees them.
    """
    to_upload = [i for i in items if upload and i.local_exists and not i.remote_exists_before]
    logger.info("Items to upload: %d", len(to_upload))
    for n, item in enumerate(to_upload, start=1):
        logger.info("[%d/%d] Uploading %s: %s", n, len(to_upload), item.kind, item.remote_path)
        fn: Callable[[], None]
        if item.kind == "file":
            fn = partial(
                upload_file_fn,
                file_path=item.local_path,
                dataset_repository=repository,
                dataset_path=item.remote_path,
                token=token,
            )
        else:
            fn = partial(
                upload_folder_fn,
                folder_datasets=base_folder,
                dataset_repository=repository,
                dataset_config=os.path.relpath(item.local_path, base_folder),
                token=token,
                path_in_repo=item.remote_path,
            )
        error = upload_with_retry(fn, item.remote_path, max_retries, retry_base_seconds, sleep_fn=sleep_fn)
        if error is None:
            item.uploaded_now = True
            if item.kind == "file":
                snapshot.files.add(item.remote_path)
            else:
                snapshot.folders.add(item.remote_path)
        else:
            item.upload_error = error
            logger.error("Giving up on %s: %s", item.remote_path, error)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(items: List[SyncItem]) -> pd.DataFrame:
    """Per (stage, task, method) summary of expected / local / remote / uploaded / missing counts."""
    rows: List[Dict[str, Any]] = []
    keys = sorted({(i.stage, i.task, i.method or "-") for i in items})
    for stage, task, method in keys:
        group = [i for i in items if i.stage == stage and i.task == task and (i.method or "-") == method]
        rows.append({
            "stage": stage,
            "task": task,
            "method": method,
            "expected": len(group),
            "local": sum(i.local_exists for i in group),
            "remote_before": sum(i.remote_exists_before for i in group),
            "uploaded_now": sum(i.uploaded_now for i in group),
            "remote_after": sum(i.remote_exists_after for i in group),
            "missing_everywhere": sum(i.missing_everywhere for i in group),
            "upload_failures": sum(i.upload_error is not None for i in group),
        })
    return pd.DataFrame(rows)


def write_status_json(items: List[SyncItem], report: pd.DataFrame, status_path: str, repository: str) -> None:
    """Persist the run outcome: summary table + explicit problem lists."""
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": repository,
        "summary": report.to_dict(orient="records"),
        "missing_everywhere": [i.remote_path for i in items if i.missing_everywhere],
        "local_only_not_uploaded": [
            i.remote_path for i in items
            if i.local_exists and not i.remote_exists_before and not i.uploaded_now
        ],
        "upload_failures": {i.remote_path: i.upload_error for i in items if i.upload_error is not None},
    }
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    logger.info("Status written: %s", status_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    try:
        import dotenv
    except ImportError:  # optional: HF_TOKEN can come from the plain environment
        dotenv = None  # type: ignore[assignment]

    parser = argparse.ArgumentParser(
        description="Synchronize I-CARE artifacts with the HuggingFace dataset repository.",
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES + ["all"], default=["all"],
                        help="Artifact types to synchronize.")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--index-start", type=int, default=0,
                        help="Index of first entity in metadata to process.")
    parser.add_argument("--max-identities", type=int, default=100,
                        help="Number of entities to process per task.")
    parser.add_argument("--no-upload", action="store_true", default=False,
                        help="Check-only mode: report local/remote completion status, upload nothing.")
    parser.add_argument("--base-folder", default="assets",
                        help="Local base folder for data storage.")
    parser.add_argument("--status-path", default=os.path.join("assets", "hf_sync_status.json"),
                        help="Where to write the JSON status report.")
    parser.add_argument("--repository", default=HF_REPO)
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Upload retry attempts per item (exponential backoff).")
    parser.add_argument("--retry-base-seconds", type=float, default=60.0,
                        help="Backoff base: wait retry_base_seconds * 2^attempt between retries.")
    parser.add_argument("--seeds", nargs="+", type=int, default=GENERATE_DATASET_SEEDS,
                        help="Generation seeds (for the generated-datasets local completeness check).")
    args = parser.parse_args()

    stages: List[str] = STAGES if "all" in args.stages else args.stages
    upload: bool = not args.no_upload

    # --- Token (same lookup as pipeline_05) ---
    _env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", "forgety", ".env"),
    ]
    for _ep in _env_paths:
        if dotenv is not None and os.path.exists(_ep):
            dotenv.load_dotenv(_ep)
            break
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    if upload:
        assert len(HF_TOKEN) > 0, (
            "HF_TOKEN not found. Set HF_TOKEN in the environment or in a .env file, "
            "or pass --no-upload for a check-only run on a public repository."
        )

    logger.info("Stages: %s | tasks: %s | methods: %s | upload: %s",
                stages, args.tasks, args.methods, upload)

    metadata_by_task: Dict[str, List[Dict[str, Any]]] = {
        task: get_metadata_filtered(cast(_type_task, task), base_folder=args.base_folder)
        for task in args.tasks
    }

    items = enumerate_items(
        stages=stages,
        tasks=args.tasks,
        methods=args.methods,
        index_start=args.index_start,
        max_identities=args.max_identities,
        base_folder=args.base_folder,
        metadata_by_task=metadata_by_task,
    )
    logger.info("Expected artifacts enumerated: %d", len(items))

    annotate_local_existence(items, metadata_by_task, seeds=args.seeds, base_folder=args.base_folder)
    snapshot = fetch_remote_snapshot(args.repository, token=HF_TOKEN or None)
    annotate_remote_existence(items, snapshot)

    synchronize(
        items=items,
        snapshot=snapshot,
        upload=upload,
        repository=args.repository,
        token=HF_TOKEN,
        base_folder=args.base_folder,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )

    report = build_report(items)
    print("-" * 100)
    print("Completion status (counts per stage x task x method):")
    print(report.to_string(index=False))
    missing = [i for i in items if i.missing_everywhere]
    if missing:
        print("-" * 100)
        print(f"Missing everywhere (not local, not remote — never computed or cluster-only): {len(missing)}")
        for i in missing[:50]:
            print(f"  {i.remote_path}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more (full list in the status JSON)")
    write_status_json(items, report, args.status_path, args.repository)
