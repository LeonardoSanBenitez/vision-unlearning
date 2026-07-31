"""Acquire the forget/retain training splits for the scenes (SUN) spike/W5 target.

Reproduces the scenes branch of ``pipeline_01_get_data.ipynb`` (cells 3-8) for a single target. The SUN
dataset has no download helper in-repo (``download_dataset_sun`` is a stub), so the notebook downloads two
tarballs by hand; this script does the same:

1. Download ``SUNAttributeDB.tar.gz`` (~0.5 MB, attributes/filenames) and ``SUNAttributeDB_Images.tar.gz``
   (~1.8 GB, the images) from cs.brown.edu and extract both under ``assets/datasets/SUN`` (idempotent).
   The historical ``~gmpatter`` URL 301-redirects to ``people/gmpatter``; we use the current URL directly.
2. ``split_dataset_sun`` builds ``SUN_splits_filtered/<target>/train_forget`` (the target's images capped at
   ``smallest_entity``) and ``.../train_retain`` (``smallest_entity`` per each of the other 99 filtered
   scenes), reading the already-present ``metadata_scenes_2_enriched_filtered.json`` for ``restrict_labels``
   (underscored SUN category names, e.g. ``football_field``) and the cap.

Windows note: ``split_dataset_sun`` uses ``os.symlink`` (WinError 1314 here); for the split call only it is
replaced by a real file copy - same selection logic, no library change (identical to the breeds acquire).
The split folder is keyed by the underscored metadata name (``football_field``), NOT the "a football field
scene" prompt form. Paths resolve from ``__file__``; run with PYTHONPATH at the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_SUN_DB_URL = "https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB.tar.gz"
_SUN_IMG_URL = "https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB_Images.tar.gz"


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:  # noqa: S310 (trusted URL)
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.rename(dest)


def _copy_as_symlink_substitute(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def main() -> int:
    from vision_unlearning.datasets import split_dataset_sun
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("every_epoch_acquire_scenes")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Acquire scenes (SUN) forget/retain split for one target.")
    parser.add_argument("--icare-assets", default=str(_ICARE_DIR / "assets"))
    parser.add_argument("--target", default="football_field", help="Underscored SUN metadata name.")
    args = parser.parse_args()

    assets = Path(args.icare_assets).resolve()
    sun_dir = assets / "datasets" / "SUN"
    filtered_base = assets / "datasets" / "SUN_splits_filtered"
    metadata_path = assets / "metadata_scenes_2_enriched_filtered.json"
    target = args.target

    if not metadata_path.exists():
        logger.error("Filtered metadata missing: %s", metadata_path)
        return 2
    metadata_filtered = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(metadata_filtered, list) and len(metadata_filtered) == 100
    restrict_labels = [e["name"] for e in metadata_filtered]
    smallest_entity = min(e["dataset_n_original"] for e in metadata_filtered)
    logger.info("restrict_labels=%d, smallest_entity=%d, target=%r", len(restrict_labels), smallest_entity, target)
    if target not in restrict_labels:
        logger.error("Target %r not among the 100 filtered scenes.", target)
        return 2

    # Step 1: download + extract SUN (idempotent)
    sun_dir.mkdir(parents=True, exist_ok=True)
    if not (sun_dir / "SUNAttributeDB").exists() or not (sun_dir / "images").exists():
        t0 = time.time()
        db_tar = sun_dir / "SUNAttributeDB.tar.gz"
        img_tar = sun_dir / "SUNAttributeDB_Images.tar.gz"
        logger.info("downloading SUN attribute DB (~0.5MB) ...")
        _download(_SUN_DB_URL, db_tar)
        logger.info("downloading SUN images (~1.8GB, this is the slow part) ...")
        _download(_SUN_IMG_URL, img_tar)
        logger.info("download done in %.0fs; extracting ...", time.time() - t0)
        for tar_path in (db_tar, img_tar):
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(sun_dir)  # noqa: S202 (trusted archive)
        logger.info("extracted.")
    else:
        logger.info("SUN already downloaded+extracted, skipping.")

    if not (sun_dir / "SUNAttributeDB" / "images.mat").exists():
        logger.error("SUNAttributeDB/images.mat missing after extract at %s", sun_dir)
        return 3
    if not (sun_dir / "images").is_dir():
        logger.error("images/ dir missing after extract at %s", sun_dir)
        return 3

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
        class_to_number = split_dataset_sun(
            str(sun_dir), str(forget_dir), str(retain_dir), target,
            forget_max_img=smallest_entity, retain_max_img_per_class=smallest_entity,
            restrict_labels=restrict_labels,
        )
        logger.info("split built in %.0fs", time.time() - t1)
    finally:
        os.symlink = real_symlink  # type: ignore[assignment]

    forget_files = [p for p in forget_dir.iterdir() if p.suffix.lower() == ".jpg"]
    retain_files = [p for p in retain_dir.iterdir() if p.suffix.lower() == ".jpg"]
    n_classes = sum(v > 0 for v in class_to_number.values())
    logger.info(
        "forget=%d (expect %d), retain=%d (expect %d), classes=%d (expect %d)",
        len(forget_files), smallest_entity,
        len(retain_files), (len(restrict_labels) - 1) * smallest_entity,
        n_classes, len(restrict_labels),
    )
    ok = (
        len(forget_files) == smallest_entity
        and len(retain_files) == (len(restrict_labels) - 1) * smallest_entity
        and n_classes == len(restrict_labels)
        and (forget_dir / "metadata.jsonl").exists()
        and (retain_dir / "metadata.jsonl").exists()
    )
    if not ok:
        logger.error("Split validation FAILED - counts/metadata do not match the canonical contract.")
        return 5
    from PIL import Image
    for p in (forget_files[:2] + retain_files[:2]):
        with Image.open(p) as im:
            im.convert("RGB").load()
    logger.info("Sample images open as valid RGB. Split OK: %s", filtered_base / target)
    print("ACQUIRE_OK", str(filtered_base / target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
