"""Acquire the forget/retain training splits for the breeds spike target.

This reproduces the breeds branch of ``pipeline_01_get_data.ipynb`` (cells 22-33) for a
single target, so the "every epoch" ablation's W3 spike has real ``distil`` training data:

1. ``download_dataset_taras_breeds`` git-clones the public Dog-Breeds-Dataset and converts
   every image to jpg under ``assets/datasets/taras_breeds`` (idempotent; skips if present).
2. ``split_dataset_taras_breeds`` builds ``taras_breeds_splits_filtered/<target>/train_forget``
   (the target's images, capped at ``smallest_entity``) and ``.../train_retain`` (``smallest_entity``
   images from each of the other 99 filtered breeds), reading the already-present
   ``metadata_breeds_2_enriched_filtered.json`` for the 100-entity ``restrict_labels`` and the cap.

Windows note: ``split_dataset_taras_breeds`` uses ``os.symlink``, which fails on this Windows
host (WinError 1314 - the process lacks the create-symlink privilege). For the duration of the
split call ONLY, ``os.symlink`` is replaced by a real file copy. This changes nothing about which
images land in forget vs retain (the canonical selection logic is untouched) - only the filesystem
operation used to materialise them - and copies are in fact more robust for the training dataloader
on Windows than symlinks would be. No library file is modified.

Paths are resolved from ``__file__``, so the script is correct regardless of the process CWD.
``vision_unlearning`` is imported lazily inside ``main`` (matching ``select_entities.py``); run with
``PYTHONPATH`` pointing at the vision-unlearning repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

# This file lives at <repo>/vision_unlearning/benchmarks/I_care/ablations/every_epoch/.
_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]           # .../benchmarks/I_care


def _copy_as_symlink_substitute(src: str, dst: str) -> None:
    """Drop-in for ``os.symlink`` on Windows: materialise ``dst`` as a real copy of ``src``."""
    shutil.copy2(src, dst)


def main() -> int:
    from vision_unlearning.datasets import (
        count_classes_dataset_taras_breeds,
        download_dataset_taras_breeds,
        split_dataset_taras_breeds,
    )
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("every_epoch_acquire")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Acquire breeds forget/retain splits for one target.")
    parser.add_argument(
        "--icare-assets",
        default=str(_ICARE_DIR / "assets"),
        help="Canonical I-CARE assets folder (default: resolved from __file__).",
    )
    parser.add_argument(
        "--target",
        default="bouvier des flandres dog",
        help="Breeds target folder/entity name to build the split for.",
    )
    args = parser.parse_args()

    assets = Path(args.icare_assets).resolve()
    dataset_base_path = assets / "datasets" / "taras_breeds"
    cache_folder = assets / "datasets" / "temp_taras_unconverted"
    filtered_base = assets / "datasets" / "taras_breeds_splits_filtered"
    metadata_path = assets / "metadata_breeds_2_enriched_filtered.json"
    target = args.target

    logger.info("icare-assets   = %s", assets)
    logger.info("dataset_base   = %s", dataset_base_path)
    logger.info("filtered_base  = %s", filtered_base)
    logger.info("target         = %s", target)

    # read the already-filtered 100-entity metadata (skips the manual-CSV enrichment step)
    if not metadata_path.exists():
        logger.error("Filtered metadata missing: %s", metadata_path)
        return 2
    metadata_filtered = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(metadata_filtered, list) and len(metadata_filtered) == 100
    restrict_labels = [e["name"] for e in metadata_filtered]
    smallest_entity = min(e["dataset_n_original"] for e in metadata_filtered)
    logger.info("restrict_labels=%d entities, smallest_entity=%d", len(restrict_labels), smallest_entity)
    if target not in restrict_labels:
        logger.error("Target %r is not among the 100 filtered breeds.", target)
        return 2

    # Step 1: download + convert the raw Taras Dog-Breeds dataset (idempotent)
    t0 = time.time()
    logger.info("Downloading/converting Taras breeds dataset (idempotent)...")
    download_dataset_taras_breeds(str(dataset_base_path), str(cache_folder))
    logger.info("Download/convert done in %.1fs", time.time() - t0)

    counts = dict(count_classes_dataset_taras_breeds(str(dataset_base_path)))
    missing = [lbl for lbl in restrict_labels if counts.get(lbl, 0) < smallest_entity]
    if missing:
        logger.error(
            "%d filtered breeds have <%d images in the download (cannot build a valid retain set): %s",
            len(missing), smallest_entity, missing[:10],
        )
        return 3
    logger.info("All 100 filtered breeds present with >=%d images.", smallest_entity)

    # Step 2: build the split for the single target, copying instead of symlinking
    forget_dir = filtered_base / target / "train_forget"
    retain_dir = filtered_base / target / "train_retain"
    if forget_dir.exists() or retain_dir.exists():
        logger.error("Split already exists for %r (%s). Remove it to rebuild.", target, filtered_base / target)
        return 4

    real_symlink = os.symlink
    os.symlink = _copy_as_symlink_substitute  # type: ignore[assignment]
    try:
        t1 = time.time()
        class_to_number = split_dataset_taras_breeds(
            str(dataset_base_path),
            str(forget_dir),
            str(retain_dir),
            target,
            forget_max_img=smallest_entity,
            retain_max_img_per_class=smallest_entity,
            restrict_labels=restrict_labels,
        )
        logger.info("Split built in %.1fs", time.time() - t1)
    finally:
        os.symlink = real_symlink  # type: ignore[assignment]

    # content validation (files, not just folders)
    forget_files = [p for p in forget_dir.iterdir() if p.suffix.lower() == ".jpg"]
    retain_files = [p for p in retain_dir.iterdir() if p.suffix.lower() == ".jpg"]
    n_classes_saved = sum(v > 0 for v in class_to_number.values())
    logger.info(
        "forget jpgs=%d (expect %d), retain jpgs=%d (expect %d), classes_saved=%d (expect %d)",
        len(forget_files), smallest_entity,
        len(retain_files), (len(restrict_labels) - 1) * smallest_entity,
        n_classes_saved, len(restrict_labels),
    )
    ok = (
        len(forget_files) == smallest_entity
        and len(retain_files) == (len(restrict_labels) - 1) * smallest_entity
        and n_classes_saved == len(restrict_labels)
        and (forget_dir / "metadata.jsonl").exists()
        and (retain_dir / "metadata.jsonl").exists()
    )
    if not ok:
        logger.error("Split validation FAILED - counts/metadata do not match the canonical contract.")
        return 5

    # sample-open a few images as valid RGB
    from PIL import Image
    for p in (forget_files[:2] + retain_files[:2]):
        with Image.open(p) as im:
            im.convert("RGB").load()
    logger.info("Sample images open as valid RGB. Split OK: %s", filtered_base / target)
    print("ACQUIRE_OK", str(filtered_base / target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
