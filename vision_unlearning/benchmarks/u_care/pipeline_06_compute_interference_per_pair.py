"""Classify a U-Care answer set into one metric record per receiver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.metadata import BaselineAccuracy
from vision_unlearning.benchmarks.u_care.upload_assets import upload_file_asset

if TYPE_CHECKING:
    from vision_unlearning.metrics.image import MetricImageClassifier


def receiver_prompt_pairs(receiver: str, seeds: Sequence[int]) -> List[Tuple[int, str]]:
    """Return the answer-set images whose true class is ``receiver``."""
    if receiver not in cfg.ENTITIES:
        raise ValueError(f"Unknown receiver: {receiver}")
    if cfg.entity_domain(receiver) == "style":
        return [
            (seed, cfg.answer_set_prompt(receiver, object_class))
            for seed in seeds
            for object_class in cfg.OBJECT_ENTITIES
        ]
    return [
        (seed, cfg.answer_set_prompt(theme, receiver))
        for seed in seeds
        for theme in cfg.STYLE_ENTITIES
    ]


def receiver_image_filenames(
    receiver: str, seeds: Sequence[int], prefix: str
) -> List[str]:
    """Return filenames for the receiver's slice of an answer set."""
    return [
        f"{prefix}_{seed:02d}_{prompt}.png"
        for seed, prompt in receiver_prompt_pairs(receiver, seeds)
    ]


def score_receiver(
    receiver: str,
    images: Sequence[Image.Image],
    classifier_style: MetricImageClassifier,
    classifier_object: MetricImageClassifier,
) -> Dict[str, float]:
    """Compute recognition accuracy and true-class probability for one receiver."""
    if not images:
        raise ValueError(f"No images supplied for receiver {receiver}")
    classifier = (
        classifier_style if cfg.entity_domain(receiver) == "style" else classifier_object
    )
    correct = 0
    probability_sum = 0.0
    for image in images:
        result = classifier.score(image)
        correct += int(result["predicted_label"] == receiver)
        probability_sum += float(result["probabilities"][receiver])
    count = len(images)
    return {
        "accuracy": correct / count,
        "target_probability": probability_sum / count,
    }


def compute_interference_per_pair(
    answer_set_folder: str,
    seeds: Sequence[int],
    classifier_style: MetricImageClassifier,
    classifier_object: MetricImageClassifier,
    prefix: str = "off",
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Classify all 71 receiver slices from an answer-set folder."""
    folder = Path(answer_set_folder)
    result: Dict[str, Dict[str, float]] = {}
    for receiver in cfg.ENTITIES:
        filenames = receiver_image_filenames(receiver, seeds, prefix)
        images = []
        for filename in filenames:
            image_path = folder / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Missing answer-set image: {image_path}")
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        scored = score_receiver(receiver, images, classifier_style, classifier_object)
        if baseline is not None:
            scored["accuracy_diff"] = scored["accuracy"] - baseline[receiver]["accuracy"]
            scored["target_probability_diff"] = (
                scored["target_probability"] - baseline[receiver]["target_probability"]
            )
        result[receiver] = scored
    return result


def write_json(data: object, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    from vision_unlearning.metrics.image import MetricImageClassifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-set-folder", required=True)
    parser.add_argument("--style-checkpoint", required=True)
    parser.add_argument("--object-checkpoint", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--seed", type=int, nargs="+", default=[188])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", choices=["off", "on"], default="off")
    parser.add_argument("--emitter")
    parser.add_argument("--method", choices=list(cfg.ALGORITHM_REGISTRY))
    parser.add_argument("--baseline-path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output JSON file.",
    )
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--hf-repo-id", default=cfg.U_CARE_REMOTE_REPOSITORY_NAME)
    args = parser.parse_args()

    if (args.emitter is None) != (args.method is None):
        parser.error("--emitter and --method must be supplied together")

    if args.output_path is not None:
        output_path = Path(args.output_path)
    elif args.emitter is not None:
        index = cfg.ENTITIES.index(args.emitter)
        output_path = Path(
            "assets/datasets"
            f"/interferences_caused_by_{index}_{args.method}"
            f"{cfg.model_segment('sd_style50')}.json"
        )
    else:
        output_path = Path("assets/datasets/accuracies_original.json")

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    style_classifier = MetricImageClassifier(
        checkpoint_path=args.style_checkpoint,
        labels=cfg.STYLE_ENTITIES,
        device=args.device,
    )
    object_classifier = MetricImageClassifier(
        checkpoint_path=args.object_checkpoint,
        labels=cfg.OBJECT_ENTITIES,
        device=args.device,
    )
    baseline = None
    if args.baseline_path:
        with open(args.baseline_path, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    result = compute_interference_per_pair(
        args.answer_set_folder,
        args.seed,
        style_classifier,
        object_classifier,
        prefix=args.prefix,
        baseline=baseline,
    )
    write_json(result, str(output_path))
    if args.upload_to_hf:
        if not args.hf_token:
            parser.error("--upload-to-hf requires --hf-token or HF_TOKEN")
        remote_path = (
            "datasets/accuracies_original.json"
            if args.emitter is None and args.method is None
            else output_path
        )
        upload_file_asset(
            output_path,
            remote_path,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
        )
    print(f"Wrote {len(result)} receiver records to {output_path}")


if __name__ == "__main__":
    main()
