"""pipeline_05 — Compute DINOv2 embeddings for generated image datasets.

For each entity in a task, runs DINOv2-ViT-S/14 on the generated images (LoRA-ON
and LoRA-OFF) and saves per-image embeddings to:

    assets/datasets/embeddings_{task}_{lora_state}_{method}_{epochs:03d}.json

HF_TOKEN must be set in the environment (or in a .env file in the working directory)
for HuggingFace uploads.

Usage
-----
    python pipeline_05_compute_embeddings.py --task people --method distil
    python pipeline_05_compute_embeddings.py --task people --method uce --max-identities 20
    python pipeline_05_compute_embeddings.py --dry-run --task people --method distil
    python pipeline_05_compute_embeddings.py --help

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
Requires: GPU with torch+torchvision (DINOv2 inference).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Literal, Optional

from vision_unlearning.datasets.testbed import (
    get_target_preprocessed,
    get_target_overwrite,
    get_generated_dataset_folder,
    get_generated_dataset_file,
    get_shared_baseline_folder,
    get_metadata_filtered,
    GeneratedDataset,
)
from vision_unlearning.integrations.huggingface import (
    huggingface_dataset_file_upload,
    huggingface_dataset_file_exists,
    huggingface_dataset_file_download,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("compute_embeddings")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
EMBEDDING_MODEL = "dinov2_vits14"
EMBEDDING_DIM = 384
BASE_FOLDER = "assets"
PROGRESS_PATH = os.path.join(BASE_FOLDER, "embedding_progress.json")


# ---------------------------------------------------------------------------
# Progress tracking helpers (pure — safe to import / test without GPU)
# ---------------------------------------------------------------------------

def load_progress(progress_path: str = PROGRESS_PATH) -> Dict[str, str]:
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {}


def save_progress(progress: Dict[str, str], progress_path: str = PROGRESS_PATH) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Path helpers (pure — safe to import / test without GPU)
# ---------------------------------------------------------------------------

def get_embedding_output_path(
    task: str,
    target_preprocessed: str,
    method: str,
    num_train_epochs: int,
    base_folder: str = "assets",
) -> str:
    """Local path for the output JSON.

    Baseline embeddings (target_preprocessed == 'original') use a method-agnostic
    name because they embed generated_{task}_baseline/ which has no method or epoch.
    Per-entity embeddings include method and epoch.
    """
    if target_preprocessed == "original":
        filename = f"embeddings_{task}_original.json"
    else:
        filename = f"embeddings_{task}_{target_preprocessed}_{method}_{num_train_epochs:03d}.json"
    return os.path.join(base_folder, "datasets", filename)


def get_embedding_hf_path(
    task: str,
    target_preprocessed: str,
    method: str,
    num_train_epochs: int,
) -> str:
    """HF repo path (no leading slash) for the output JSON."""
    if target_preprocessed == "original":
        return f"datasets/embeddings_{task}_original.json"
    return f"datasets/embeddings_{task}_{target_preprocessed}_{method}_{num_train_epochs:03d}.json"


# ---------------------------------------------------------------------------
# Skip logic (injectable hf_file_exists_fn — testable without network)
# ---------------------------------------------------------------------------

def should_skip(
    local_path: str,
    hf_path: str,
    replace_if_exists: bool,
    upload_to_hf: bool,
    hf_file_exists_fn: Callable[[str, str, str], bool] = huggingface_dataset_file_exists,
    hf_repo: str = HF_REPO,
    hf_token: str = "",
) -> bool:
    """Return True if we should skip this entity (already done).

    Args:
        local_path: Local output JSON path.
        hf_path: HF repo path.
        replace_if_exists: If True, always recompute.
        upload_to_hf: If True, also check HF for existing file.
        hf_file_exists_fn: Injectable fn(repo, path, token) -> bool for HF check.
        hf_repo: HF dataset repository ID.
        hf_token: HF auth token.
    """
    if replace_if_exists:
        return False
    if os.path.exists(local_path):
        logger.info("Skipping (local exists): %s", local_path)
        return True
    if upload_to_hf and hf_file_exists_fn(hf_repo, hf_path, hf_token):
        logger.info("Skipping (HF exists): %s", hf_path)
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset completeness check (pure — safe to import / test without GPU)
# ---------------------------------------------------------------------------

def dataset_folder_is_complete(
    folder: str,
    seeds: List[int],
    prompts: List[str],
    lora_state: Literal["on", "off"],
) -> bool:
    """Return True if ALL expected image files for lora_state exist in folder.

    Expected files: one per (seed × prompt) for the given lora_state.
    Filename format: ``{lora_state}_{seed:02d}_{prompt}.png``

    Args:
        folder: Local directory that should contain the images.
        seeds: List of generation seeds to check.
        prompts: Full prompt strings (e.g. "An image of Colin Powell").
        lora_state: 'on' or 'off' — only files for this state are checked.
    """
    for seed in seeds:
        for prompt in prompts:
            filename = get_generated_dataset_file(lora_state, seed, prompt)
            if not os.path.exists(os.path.join(folder, filename)):
                return False
    return True


# ---------------------------------------------------------------------------
# HF dataset folder existence check (injectable — testable without network)
# ---------------------------------------------------------------------------

def hf_dataset_folder_exists(
    hf_repo: str,
    dataset_config: str,
    hf_token: str,
    hf_prefix: str = "datasets",
) -> bool:
    """Return True if the HF dataset folder has at least one file.

    Uses the HF tree API (same approach as huggingface_dataset_download).
    Returns False on any network/auth error so the caller can skip gracefully.

    Args:
        hf_repo: HF dataset repository ID.
        dataset_config: Folder name within hf_prefix on HF
                        (e.g. "generated_people_George W Bush_distil_400").
        hf_token: HF auth token.
        hf_prefix: Prefix path within the HF repo (default "datasets").
    """
    try:
        import requests as _requests  # local import — not needed at module load time
        hf_path = f"{hf_prefix}/{dataset_config}" if hf_prefix else dataset_config
        tree_url = (
            f"https://huggingface.co/api/datasets/{hf_repo}/tree/main/{hf_path}"
        )
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        r = _requests.get(tree_url, headers=headers, timeout=15)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        entries = r.json()
        return isinstance(entries, list) and len(entries) > 0
    except Exception as exc:
        logger.warning(
            "hf_dataset_folder_exists: error checking %s/%s: %s",
            hf_repo, dataset_config, exc,
        )
        return False


# ---------------------------------------------------------------------------
# Dry-run manifest helpers (pure — safe to import / test without GPU)
# ---------------------------------------------------------------------------

ManifestEntry = Dict[str, str]  # keys: entity, lora_state, status


def get_manifest_entry(
    task: str,
    method: str,
    num_train_epochs: int,
    target_hf_name: str,
    lora_state: Literal["on", "off"],
    base_folder: str,
    hf_repo: str,
    hf_token: str,
    seeds: List[int],
    prompts: List[str],
    hf_folder_exists_fn: Callable[..., bool] = hf_dataset_folder_exists,
) -> str:
    """Classify a single (entity, lora_state) combination for the dry-run manifest.

    Returns one of:
        'done'             — output embedding JSON exists locally
        'local'            — dataset folder exists locally (but no embedding JSON yet)
        'local_incomplete' — dataset folder exists but is missing expected images
        'hf_only'          — dataset folder absent locally but present on HF
        'absent'           — dataset folder absent everywhere
    """
    output_path = get_embedding_output_path(
        task, target_hf_name, method, num_train_epochs, base_folder=base_folder
    )
    if os.path.exists(output_path):
        return "done"

    dataset_folder = get_generated_dataset_folder(
        task, method, num_train_epochs, target_hf_name, base_folder=base_folder  # type: ignore[arg-type]
    )

    if os.path.exists(dataset_folder):
        complete = dataset_folder_is_complete(dataset_folder, seeds, prompts, lora_state)
        return "local" if complete else "local_incomplete"

    # Not local — check HF
    dataset_config_name = os.path.basename(dataset_folder)
    if hf_folder_exists_fn(hf_repo, dataset_config_name, hf_token):
        return "hf_only"

    return "absent"


def build_manifest(
    task: str,
    method: str,
    num_train_epochs: int,
    index_start: int,
    max_identities: int,
    metadata_filtered: List[Dict[str, Any]],
    lora_states: List[Literal["on", "off"]],
    base_folder: str,
    hf_repo: str,
    hf_token: str,
    seeds: List[int],
    prompts: List[str],
    hf_folder_exists_fn: Callable[..., bool] = hf_dataset_folder_exists,
) -> List[ManifestEntry]:
    """Scan all entities in range and return a manifest list.

    Each entry has keys: entity, lora_state, status.
    """
    entries: List[ManifestEntry] = []
    for index in range(index_start, index_start + max_identities):
        target: str = metadata_filtered[index]["name"]
        target_hf_name: str = metadata_filtered[index].get("_hf_name", target)
        for ls in lora_states:
            ls_typed: Literal["on", "off"] = ls  # type: ignore[assignment]
            status = get_manifest_entry(
                task=task,
                method=method,
                num_train_epochs=num_train_epochs,
                target_hf_name=target_hf_name,
                lora_state=ls_typed,
                base_folder=base_folder,
                hf_repo=hf_repo,
                hf_token=hf_token,
                seeds=seeds,
                prompts=prompts,
                hf_folder_exists_fn=hf_folder_exists_fn,
            )
            entries.append(
                {
                    "entity": target_hf_name,
                    "lora_state": ls,
                    "status": status,
                }
            )
    return entries


# ---------------------------------------------------------------------------
# Core embedding function (lives in vision_unlearning — imported here)
# ---------------------------------------------------------------------------
from vision_unlearning.benchmarks.I_care.embeddings import embed_forgetting_session  # noqa: E402

# ---------------------------------------------------------------------------
# HF dataset download helper (implementation lives in utils_analysis)
# ---------------------------------------------------------------------------
from utils_analysis import huggingface_dataset_download  # noqa: E402 — after sys.path setup


# ---------------------------------------------------------------------------
# Session runner (injectable skip / upload fns — testable without network)
# ---------------------------------------------------------------------------

def run_session(
    forgotten_entity: str,
    target_preprocessed: str,
    lora_state: Literal["on", "off"],
    output_path: str,
    hf_path: str,
    progress_key: str,
    dataset_folder: str,
    progress: Dict[str, str],
    task: str,
    method: str,
    num_train_epochs: int,
    generate_dataset_seeds: List[int],
    prompts: List[str],
    metadata_filtered: List[Dict[str, Any]],
    replace_if_exists: bool,
    upload_to_hf: bool,
    hf_token: str,
    progress_path: str = PROGRESS_PATH,
    embed_image_fn: Optional[Callable[[str], List[float]]] = None,
    hf_file_exists_fn: Callable[[str, str, str], bool] = huggingface_dataset_file_exists,
    hf_upload_fn: Optional[Callable[..., None]] = None,
    hf_download_fn: Optional[Callable[..., None]] = None,
    hf_folder_exists_fn: Callable[..., bool] = hf_dataset_folder_exists,
) -> None:
    """
    Embed all images for one forgetting session (one entity x one lora_state),
    save JSON with new schema, optionally upload to HF.

    All external side effects (HF check, embed, upload, download, folder check)
    are injectable for testing.

    Graceful skips (no exception, progress marked accordingly):
        - 'skipped'                  — output already exists (normal skip)
        - 'skipped_no_dataset'       — dataset absent both locally and on HF
        - 'skipped_incomplete_dataset' — dataset present locally but missing
                                         expected images for this lora_state
    """
    if should_skip(
        output_path,
        hf_path,
        replace_if_exists,
        upload_to_hf,
        hf_file_exists_fn=hf_file_exists_fn,
        hf_repo=HF_REPO,
        hf_token=hf_token,
    ):
        progress[progress_key] = "skipped"
        save_progress(progress, progress_path)
        return

    try:
        if not os.path.exists(dataset_folder):
            dataset_config_name = os.path.basename(dataset_folder)

            # Pre-flight: check if the dataset folder exists on HF before attempting
            # a download that would fail or produce a broken local folder.
            if not hf_folder_exists_fn(HF_REPO, dataset_config_name, hf_token):
                logger.warning(
                    "Dataset absent from HF (skipping): %s", dataset_config_name
                )
                progress[progress_key] = "skipped_no_dataset"
                save_progress(progress, progress_path)
                return

            logger.info("Downloading dataset: %s", dataset_config_name)
            if hf_download_fn is not None:
                hf_download_fn(dataset_config_name)
            else:
                huggingface_dataset_download(
                    folder_datasets=os.path.join(BASE_FOLDER, "datasets"),
                    dataset_repository=HF_REPO,
                    dataset_config=dataset_config_name,
                    token=hf_token,
                )

        # Completeness check: verify expected images exist for this lora_state.
        if not dataset_folder_is_complete(dataset_folder, generate_dataset_seeds, prompts, lora_state):
            dataset_config_name = os.path.basename(dataset_folder)
            logger.warning(
                "Dataset incomplete for lora_state=%s (skipping): %s",
                lora_state, dataset_config_name,
            )
            progress[progress_key] = "skipped_incomplete_dataset"
            save_progress(progress, progress_path)
            return

        records = embed_forgetting_session(
            dataset_folder=dataset_folder,
            seeds=generate_dataset_seeds,
            prompts=prompts,
            metadata_filtered=metadata_filtered,
            lora_state=lora_state,
            task=task,
            embed_image_fn=embed_image_fn,
        )

        output: Dict[str, Any] = {
            "metadata": {
                "task": task,
                "forgotten_entity": forgotten_entity,
                "method": method,
                "num_train_epochs": num_train_epochs,
                "lora_state": lora_state,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
            },
            "embeddings": records,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f)
        logger.info(
            "Saved JSON: %s (%d records)", output_path, len(records)
        )

        if upload_to_hf:
            if hf_upload_fn is not None:
                hf_upload_fn(output_path, hf_path)
            else:
                huggingface_dataset_file_upload(
                    file_path=output_path,
                    dataset_repository=HF_REPO,
                    dataset_path=hf_path,
                    token=hf_token,
                )
            logger.info("Uploaded to HF: %s", hf_path)

        progress[progress_key] = "done"
        save_progress(progress, progress_path)

    except Exception as exc:
        logger.error(
            "Session failed [%s / %s]: %s", forgotten_entity, lora_state, exc,
            exc_info=True,
        )
        progress[progress_key] = "failed"
        save_progress(progress, progress_path)
        raise


# ---------------------------------------------------------------------------
# Status file helper (pure — safe to import / test without GPU)
# ---------------------------------------------------------------------------

def write_run_status(
    method: str,
    task: str,
    index_start: int,
    index: int,
    max_identities: int,
    progress: Dict[str, str],
    base_folder: str = "assets",
) -> None:
    """Write a human-readable status summary to assets/run_status_{method}.txt.

    Called every 10 entities processed. Survives computer restarts — the file
    is written incrementally so progress is visible even if the run is killed.

    Args:
        method: Unlearning method (distil, uce, munba).
        task: Task name (people, objects, etc.).
        index_start: First entity index in this run.
        index: Current entity index (0-based, inclusive — last entity processed).
        max_identities: Total entities requested in this run.
        progress: Current progress dict (from load_progress).
        base_folder: Local base folder for data storage.
    """
    import datetime

    status_path = os.path.join(base_folder, f"run_status_{method}.txt")
    method_keys = [k for k in progress if f"/{method}/" in k and task in k]
    counts: Dict[str, int] = {}
    for v in progress.values():
        counts[v] = counts.get(v, 0) + 1
    method_counts: Dict[str, int] = {}
    for k in method_keys:
        v = progress[k]
        method_counts[v] = method_counts.get(v, 0) + 1

    processed = index - index_start + 1
    lines = [
        f"run_status_{method} — written {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"method={method}  task={task}  range=[{index_start}, {index_start + max_identities - 1}]",
        f"entities processed so far: {processed} / {max_identities}",
        f"",
        f"method-level counts (this session's method only):",
    ]
    for status_val, cnt in sorted(method_counts.items()):
        lines.append(f"  {status_val:<30s} {cnt}")
    lines += [
        f"",
        f"all-methods counts (full progress.json):",
    ]
    for status_val, cnt in sorted(counts.items()):
        lines.append(f"  {status_val:<30s} {cnt}")

    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Status written: %s", status_path)


# ---------------------------------------------------------------------------
# Main entry point — DINOv2 model only loaded here (not at import time)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import dotenv
    import requests  # noqa: F401 — needed by huggingface_dataset_download

    # --- CLI ---
    parser = argparse.ArgumentParser(
        description="Compute DINOv2 embeddings for generated image datasets."
    )
    parser.add_argument("--task", choices=["scenes", "objects", "breeds", "people"],
                        default="people")
    parser.add_argument("--method", choices=["munba", "uce", "distil"], default="distil")
    parser.add_argument("--max-identities", type=int, default=2,
                        help="Number of entities to process.")
    parser.add_argument("--index-start", type=int, default=0,
                        help="Index of first entity in metadata to process.")
    parser.add_argument("--num-train-epochs", type=int, default=None,
                        help="Training epochs override. Defaults: distil=400, munba=200, uce=0.")
    parser.add_argument("--lora-state", choices=["on", "off"], default="on",
                        help="LoRA state for unlearned model runs.")
    parser.add_argument("--replace-if-exists", action="store_true", default=False,
                        help="Re-embed even if output already exists.")
    parser.add_argument("--no-upload", action="store_true", default=False,
                        help="Skip uploading to HuggingFace.")
    parser.add_argument("--delete-dataset", action="store_true", default=False,
                        help="Delete local dataset folder after computing embeddings. Default is to KEEP the dataset.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45],
                        help="Generation seeds to process.")
    parser.add_argument("--base-folder", default="assets",
                        help="Local base folder for data storage.")
    parser.add_argument("--device", default=None,
                        help="Torch device for DINOv2 inference (e.g. 'cpu', 'cuda'). "
                             "Defaults to 'cuda' if available, else 'cpu'.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Scan entities in range without computing any embeddings. "
            "For each entity × lora_state, reports: done / local / local_incomplete / "
            "hf_only / absent. Writes result to "
            "assets/embedding_manifest_{task}_{method}_{epochs}.json and exits."
        ),
    )
    args = parser.parse_args()

    task: Literal["scenes", "objects", "breeds", "people"] = args.task  # type: ignore[assignment]
    method: Literal["munba", "uce", "distil"] = args.method  # type: ignore[assignment]
    max_identities: int = args.max_identities
    index_start: int = args.index_start
    lora_state: Literal["on", "off"] = args.lora_state  # type: ignore[assignment]
    replace_if_exists: bool = args.replace_if_exists
    upload_to_hf: bool = not args.no_upload
    generate_dataset_seeds: List[int] = args.seeds
    BASE_FOLDER = args.base_folder
    PROGRESS_PATH = os.path.join(BASE_FOLDER, "embedding_progress.json")
    force_device: Optional[str] = args.device

    # Default epochs per method
    _default_epochs = {"distil": 400, "munba": 200, "uce": 0}
    num_train_epochs: int = args.num_train_epochs if args.num_train_epochs is not None \
        else _default_epochs[method]

    # --- Token ---
    _env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", "forgety", ".env"),
    ]
    for _ep in _env_paths:
        if os.path.exists(_ep):
            dotenv.load_dotenv(_ep)
            break

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    assert len(HF_TOKEN) > 0, (
        "HF_TOKEN not found. Set HF_TOKEN in the environment or in a .env file "
        "in the working directory."
    )

    os.makedirs(os.path.join(BASE_FOLDER, "datasets"), exist_ok=True)

    # --- Metadata ---
    metadata_local_path = os.path.join(
        BASE_FOLDER, f"metadata_{task}_2_enriched_filtered.json"
    )
    metadata_hf_path = f"metadata_{task}_2_enriched_filtered.json"

    if not os.path.exists(metadata_local_path):
        logger.info("Downloading metadata from HF: %s", metadata_hf_path)
        huggingface_dataset_file_download(
            folder_datasets=BASE_FOLDER,
            dataset_repository=HF_REPO,
            file_path=metadata_hf_path,
            token=HF_TOKEN,
        )

    metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(
        task, base_folder=BASE_FOLDER
    )
    logger.info("Loaded %d entities from metadata.", len(metadata_filtered))

    # Build prompt list using HF-compatible names (spaces, not underscores).
    # get_target_overwrite()[0] returns e.g. "George W Bush" for people task.
    # This matches the prompted_entity and prompt values in existing embedding JSONs.
    prompts: List[str] = [
        f"An image of {get_target_overwrite(task, method, m['name'])[0]}"
        for m in metadata_filtered
    ]

    # Also build an HF-name list for use as metadata_filtered mock in embed_forgetting_session.
    # embed_forgetting_session uses metadata_filtered[i]["name"] to get prompted_entity.
    # We need the HF-space name there, so override the metadata name mapping.
    metadata_for_embed: List[Dict[str, Any]] = [
        {**m, "name": get_target_overwrite(task, method, m["name"])[0]}
        for m in metadata_filtered
    ]

    # Build an augmented metadata list with the resolved HF name for manifest use.
    metadata_with_hf_name: List[Dict[str, Any]] = [
        {**m, "_hf_name": get_target_overwrite(task, method, m["name"])[0]}
        for m in metadata_filtered
    ]

    # -----------------------------------------------------------------------
    # --dry-run: build manifest and exit (no DINOv2 loading, no computation)
    # -----------------------------------------------------------------------
    if args.dry_run:
        lora_states_for_manifest: List[Literal["on", "off"]] = ["on", "off"]
        logger.info(
            "DRY RUN: scanning %d entities (index %d..%d) for %s/%s/%d ...",
            max_identities, index_start, index_start + max_identities - 1,
            task, method, num_train_epochs,
        )
        manifest_entries = build_manifest(
            task=task,
            method=method,
            num_train_epochs=num_train_epochs,
            index_start=index_start,
            max_identities=max_identities,
            metadata_filtered=metadata_with_hf_name,
            lora_states=lora_states_for_manifest,
            base_folder=BASE_FOLDER,
            hf_repo=HF_REPO,
            hf_token=HF_TOKEN,
            seeds=generate_dataset_seeds,
            prompts=prompts,
        )
        manifest_path = os.path.join(
            BASE_FOLDER,
            f"embedding_manifest_{task}_{method}_{num_train_epochs:03d}.json",
        )
        os.makedirs(BASE_FOLDER, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as _mf:
            json.dump(manifest_entries, _mf, indent=2)
        logger.info("Manifest written: %s (%d entries)", manifest_path, len(manifest_entries))

        # Print summary counts
        from collections import Counter
        status_counts = Counter(e["status"] for e in manifest_entries)
        for status, count in sorted(status_counts.items()):
            logger.info("  %-28s %d", status, count)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Normal run: load DINOv2 and compute embeddings
    # -----------------------------------------------------------------------

    # --- DINOv2 setup (only here — not at module import) ---
    import torch
    import torchvision.transforms as T
    from PIL import Image

    if force_device is not None:
        device: str = force_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    logger.info("Loading DINOv2 %s ...", EMBEDDING_MODEL)
    dino_model = torch.hub.load("facebookresearch/dinov2", EMBEDDING_MODEL)
    dino_model = dino_model.to(device)  # type: ignore[attr-defined]
    dino_model.eval()  # type: ignore[attr-defined]
    logger.info("DINOv2 loaded.")

    dino_transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),  # type: ignore[attr-defined]
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def embed_image(image_path: str) -> List[float]:
        """
        Load a PNG from disk and return a 384-dim DINOv2 embedding as plain Python list.
        Single-image inference (no batching).
        TODO: refactor into batched DataLoader for throughput.
        """
        img = Image.open(image_path).convert("RGB")
        tensor = dino_transform(img).unsqueeze(0).to(device)  # type: ignore[attr-defined]
        with torch.no_grad():
            feat = dino_model(tensor)  # type: ignore[operator]
        return feat.squeeze().tolist()  # type: ignore[attr-defined]

    # --- Main loop ---
    progress = load_progress(PROGRESS_PATH)

    for index in range(index_start, index_start + max_identities):
        target: str = metadata_filtered[index]["name"]
        # get_target_overwrite()[0] gives the HF-compatible name (spaces, not underscores).
        # HF dataset folders use: generated_people_George W Bush_distil_400 (with spaces).
        # get_target_preprocessed returns underscores for people — NOT the HF convention.
        target_hf_name: str = get_target_overwrite(task, method, target)[0]

        progress_key = f"{task}/{method}/{target_hf_name}"
        logger.info(
            "--- [%d/%d] %s ---",
            index + 1, index_start + max_identities, target_hf_name,
        )

        ds_entity = GeneratedDataset(  # type: ignore[arg-type]
            task=task,  # type: ignore[arg-type]
            target=target_hf_name, method=method, num_train_epochs=num_train_epochs,
        )
        dataset_folder = ds_entity.folder_path

        # Download dataset once — shared by both lora-OFF baseline and lora-ON pass.
        # Guard: check HF first so a missing dataset does not crash here.
        if not os.path.exists(dataset_folder):
            dataset_config_name = os.path.basename(dataset_folder)
            if not hf_dataset_folder_exists(HF_REPO, dataset_config_name, HF_TOKEN):
                logger.warning(
                    "Dataset absent from HF, skipping entity: %s", dataset_config_name
                )
                progress[progress_key] = "skipped_no_dataset"
                save_progress(progress, PROGRESS_PATH)
                # Write status for absent entity before continuing (honours every-10 interval)
                _entities_processed_absent = index - index_start + 1
                _is_last_absent = _entities_processed_absent == max_identities
                if _entities_processed_absent % 10 == 0 or _is_last_absent:
                    write_run_status(
                        method=method,
                        task=task,
                        index_start=index_start,
                        index=index,
                        max_identities=max_identities,
                        progress=progress,
                        base_folder=BASE_FOLDER,
                    )
                continue
            logger.info("Downloading dataset: %s", dataset_config_name)
            huggingface_dataset_download(
                folder_datasets=os.path.join(BASE_FOLDER, "datasets"),
                dataset_repository=HF_REPO,
                dataset_config=dataset_config_name,
                token=HF_TOKEN,
            )
        else:
            logger.info("Dataset already on disk: %s", dataset_folder)

        # --- lora-OFF baseline (first entity only, one per task x method combo) ---
        # Priority: shared task-level baseline folder > entity folder (pre-refactor fallback).
        # Shared folder is generated by 0_generate_dataset_original.py (one run per task).
        # Entity folder fallback: pre-refactor mixed on_* + off_* folder.
        #
        # NOTE on filename coupling:
        # The baseline embedding output file is named
        #   embeddings_{task}_original_{method}_{epochs}.json
        # — i.e. it includes method and epochs even though the lora-OFF images are
        # method-agnostic.  This is intentional for two reasons:
        #
        # 1. Backward compatibility: existing HF files already follow this naming
        #    convention (e.g. embeddings_people_original_distil_400.json).  Changing
        #    the filename would break HF skip-logic and require renaming all uploaded
        #    files.
        #
        # 2. The embedding is NOT identical across methods/epochs when the shared
        #    baseline folder does not exist yet (entity-folder fallback): the entity
        #    folder used for the fallback differs per (method, epochs) tuple, so the
        #    images read can differ.  Once the shared baseline folder exists (post
        #    0_generate_dataset_original.py), all method runs read the same images
        #    and will produce identical embedding JSON content — but the separate
        #    files still exist for backward compatibility.
        #
        # The duplication is wasteful for new runs but harmless.  A future
        # refactor could emit a single embeddings_{task}_original.json and update
        # all downstream consumers.  That is tracked as a TODO; not done here.
        if index == index_start:
            baseline_output_path = get_embedding_output_path(
                task, "original", method, num_train_epochs, base_folder=BASE_FOLDER
            )
            baseline_hf_path = get_embedding_hf_path(task, "original", method, num_train_epochs)
            baseline_progress_key = f"{task}/{method}/original"

            shared_baseline_folder = get_shared_baseline_folder(task, base_folder=BASE_FOLDER)
            if os.path.exists(shared_baseline_folder):
                off_dataset_folder = shared_baseline_folder
                off_source_label = "shared baseline folder"
            else:
                off_dataset_folder = dataset_folder
                off_source_label = "entity folder (pre-refactor fallback)"
            logger.info("lora-OFF source: %s (%s)", off_dataset_folder, off_source_label)

            run_session(
                forgotten_entity="original",
                target_preprocessed="original",
                lora_state="off",
                output_path=baseline_output_path,
                hf_path=baseline_hf_path,
                progress_key=baseline_progress_key,
                dataset_folder=off_dataset_folder,
                progress=progress,
                task=task,
                method=method,
                num_train_epochs=num_train_epochs,
                generate_dataset_seeds=generate_dataset_seeds,
                prompts=prompts,
                metadata_filtered=metadata_for_embed,
                replace_if_exists=replace_if_exists,
                upload_to_hf=upload_to_hf,
                hf_token=HF_TOKEN,
                progress_path=PROGRESS_PATH,
                embed_image_fn=embed_image,
            )

        # --- lora-ON embedding ---
        output_path = get_embedding_output_path(
            task, target_hf_name, method, num_train_epochs, base_folder=BASE_FOLDER
        )
        hf_path_entity = get_embedding_hf_path(task, target_hf_name, method, num_train_epochs)

        run_session(
            forgotten_entity=target_hf_name,
            target_preprocessed=target_hf_name,
            lora_state=lora_state,
            output_path=output_path,
            hf_path=hf_path_entity,
            progress_key=progress_key,
            dataset_folder=dataset_folder,
            progress=progress,
            task=task,
            method=method,
            num_train_epochs=num_train_epochs,
            generate_dataset_seeds=generate_dataset_seeds,
            prompts=prompts,
            metadata_filtered=metadata_for_embed,
            replace_if_exists=replace_if_exists,
            upload_to_hf=upload_to_hf,
            hf_token=HF_TOKEN,
            progress_path=PROGRESS_PATH,
            embed_image_fn=embed_image,
        )

        # Delete local dataset folder to free disk space (only if --delete-dataset was passed)
        if args.delete_dataset and os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)
            logger.info("Deleted local dataset folder: %s", dataset_folder)

        # Write status file every 10 entities (and always on the last entity)
        entities_processed = index - index_start + 1
        is_last = entities_processed == max_identities
        if entities_processed % 10 == 0 or is_last:
            write_run_status(
                method=method,
                task=task,
                index_start=index_start,
                index=index,
                max_identities=max_identities,
                progress=progress,
                base_folder=BASE_FOLDER,
            )
