from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.metrics.image import MetricImageClassifier


def _repo_root() -> Path:
    print(">>> _repo_root")
    return Path(__file__).resolve().parents[3]


def _assets_root() -> Path:
    print(">>> _assets_root")
    return _repo_root() / "assets"


def ensure_stage1_layout() -> None:
    print(">>> ensure_stage1_layout")
    assets = _assets_root()
    for path in [
        assets / "datasets" / "reference",
        assets / "models",
        assets / "outputs" / "confusion_matrices",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_metadata(output_path: Optional[Path] = None) -> Path:
    print(">>> build_metadata")
    ensure_stage1_layout()
    if output_path is None:
        output_path = _assets_root() / "metadata_filtered.json"

    metadata = []
    for index, entity in enumerate(cfg.ENTITIES):
        metadata.append(
            {
                "name": entity,
                "index": index,
                "domain": cfg.entity_domain(entity),
                "unlearnable": cfg.is_unlearnable(entity),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote metadata to {output_path}")
    return output_path


def _match_theme_and_object(filename: str) -> Optional[Tuple[str, str]]:
    print(">>> _match_theme_and_object")
    stem = Path(filename).stem
    for theme in cfg.STYLE_ENTITIES:
        for object_class in cfg.OBJECT_ENTITIES:
            if re.fullmatch(rf"{re.escape(theme)}_{re.escape(object_class)}_\d+", stem):
                return theme, object_class
    return None


def _save_confusion_matrix(
    labels: List[str],
    true_labels: List[str],
    predicted_labels: List[str],
    output_path: Path,
) -> None:
    import sklearn.metrics as metrics
    print(">>> _save_confusion_matrix")
    matrix = metrics.confusion_matrix(true_labels, predicted_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(6, len(labels) * 0.8)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_classifier_sanity_check(
    reference_dir: Optional[Path] = None,
    style_checkpoint: Optional[Path] = None,
    object_checkpoint: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, Path]:
    print(">>> run_classifier_sanity_check")
    ensure_stage1_layout()

    if reference_dir is None:
        reference_dir = _assets_root() / "datasets" / "reference"
    if style_checkpoint is None:
        style_checkpoint = _assets_root() / "models" / "classifier_style.pth"
    if object_checkpoint is None:
        object_checkpoint = _assets_root() / "models" / "classifier_object.pth"
    if output_dir is None:
        output_dir = _assets_root() / "outputs" / "confusion_matrices"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference images directory not found: {reference_dir}")
    if not style_checkpoint.exists():
        raise FileNotFoundError(f"Style classifier checkpoint not found: {style_checkpoint}")
    if not object_checkpoint.exists():
        raise FileNotFoundError(f"Object classifier checkpoint not found: {object_checkpoint}")

    image_files = sorted(reference_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPG files found under {reference_dir}")

    style_metric = MetricImageClassifier(
        checkpoint_path=str(style_checkpoint),
        labels=cfg.STYLE_ENTITIES,
        device=device,
    )
    object_metric = MetricImageClassifier(
        checkpoint_path=str(object_checkpoint),
        labels=cfg.OBJECT_ENTITIES,
        device=device,
    )

    style_true: List[str] = []
    style_pred: List[str] = []
    object_true: List[str] = []
    object_pred: List[str] = []

    for image_path in image_files:
        parsed = _match_theme_and_object(image_path.name)
        if parsed is None:
            continue
        theme, object_class = parsed
        image = Image.open(image_path).convert("RGB")

        style_result = style_metric.score(image)
        object_result = object_metric.score(image)

        style_true.append(theme)
        style_pred.append(style_result["predicted_label"])
        object_true.append(object_class)
        object_pred.append(object_result["predicted_label"])

    style_output = output_dir / "style_confusion_matrix.png"
    object_output = output_dir / "object_confusion_matrix.png"

    _save_confusion_matrix(cfg.STYLE_ENTITIES, style_true, style_pred, style_output)
    _save_confusion_matrix(cfg.OBJECT_ENTITIES, object_true, object_pred, object_output)

    print(f"Style confusion matrix saved to {style_output}")
    print(f"Object confusion matrix saved to {object_output}")
    print("Classifier sanity check complete")
    return {"style": style_output, "object": object_output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 bootstrap for u-care")
    parser.add_argument("--write-metadata", action="store_true", help="Write assets/metadata_filtered.json")
    parser.add_argument("--run-classifier-check", action="store_true", help="Run classifier sanity-check over reference images")
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--style-checkpoint", type=Path, default=None)
    parser.add_argument("--object-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.write_metadata:
        build_metadata()
    if args.run_classifier_check:
        run_classifier_sanity_check(
            reference_dir=args.reference_dir,
            style_checkpoint=args.style_checkpoint,
            object_checkpoint=args.object_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
        )

    if not args.write_metadata and not args.run_classifier_check:
        print("No action requested. Use --write-metadata and/or --run-classifier-check.")
