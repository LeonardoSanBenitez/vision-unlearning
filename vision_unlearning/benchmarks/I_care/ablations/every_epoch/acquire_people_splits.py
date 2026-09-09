"""Acquire the forget/retain training splits for the people spike/W4 target.

Reproduces the people branch of ``pipeline_01_get_data.ipynb`` (cell 41) for a single target:
``download_dataset_lfw`` pulls the ``bitmind/lfw`` dataset from HuggingFace and, in the same pass,
writes ``lfw_splits_filtered/<target>/train_forget`` (the target's images, capped at ``smallest_entity``)
and ``.../train_retain`` (``smallest_entity`` images from each of the other 99 filtered people), reading
the already-present ``metadata_people_2_enriched_filtered.json`` for ``restrict_labels`` and the cap.

Unlike the breeds/scenes splits, LFW writes real JPEGs (``img.save``), so there is no ``os.symlink``
step and no Windows copy-substitute is needed. Paths resolve from ``__file__``; run with ``PYTHONPATH``
at the repo root, ``HF_TOKEN`` set, ``HF_HUB_DISABLE_XET=1``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]


def main() -> int:
    from vision_unlearning.datasets import download_dataset_lfw
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("every_epoch_acquire_people")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Acquire people (LFW) forget/retain split for one target.")
    parser.add_argument("--icare-assets", default=str(_ICARE_DIR / "assets"))
    parser.add_argument("--target", default="Mark_Philippoussis")
    args = parser.parse_args()

    assets = Path(args.icare_assets).resolve()
    filtered_base = assets / "datasets" / "lfw_splits_filtered"
    metadata_path = assets / "metadata_people_2_enriched_filtered.json"
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
        logger.error("Target %r not among the 100 filtered people.", target)
        return 2

    forget_dir = filtered_base / target / "train_forget"
    retain_dir = filtered_base / target / "train_retain"
    if forget_dir.exists() or retain_dir.exists():
        logger.error("Split already exists for %r (%s). Remove it to rebuild.", target, filtered_base / target)
        return 4

    class_to_number = download_dataset_lfw(
        str(forget_dir), str(retain_dir), target,
        forget_max_img=smallest_entity,
        retain_max_img_per_class=smallest_entity,
        restrict_labels=restrict_labels,
    )

    forget_files = [p for p in forget_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    retain_files = [p for p in retain_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
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
