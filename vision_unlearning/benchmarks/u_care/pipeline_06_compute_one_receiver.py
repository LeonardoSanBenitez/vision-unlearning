"""Classify one receiver group from a U-Care answer set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.pipeline_06_compute_interference_per_pair import (
    receiver_image_filenames,
    score_receiver,
)


def compute_one_receiver(
    answer_set_folder: str,
    receiver: str,
    seeds: Sequence[int],
    classifier_style: object,
    classifier_object: object,
    prefix: str = "off",
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Score one receiver and return a one-entry receiver-keyed result."""
    folder = Path(answer_set_folder)
    image_paths = [
        folder / filename
        for filename in receiver_image_filenames(receiver, seeds, prefix)
    ]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} image(s) for receiver {receiver}: {missing[0]}"
        )

    from PIL import Image

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    result = score_receiver(
        receiver,
        images,
        classifier_style,  # type: ignore[arg-type]
        classifier_object,  # type: ignore[arg-type]
    )
    if baseline is not None:
        result["accuracy_diff"] = result["accuracy"] - baseline[receiver]["accuracy"]
        result["target_probability_diff"] = (
            result["target_probability"] - baseline[receiver]["target_probability"]
        )
    return {receiver: result}


def main() -> None:
    from vision_unlearning.metrics.image import MetricImageClassifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", required=True, choices=cfg.ENTITIES)
    parser.add_argument("--answer-set-folder", required=True)
    parser.add_argument("--style-checkpoint", required=True)
    parser.add_argument("--object-checkpoint", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--seed", type=int, nargs="+", default=[188])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", choices=["off", "on"], default="off")
    parser.add_argument("--baseline-path")
    args = parser.parse_args()

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

    result = compute_one_receiver(
        answer_set_folder=args.answer_set_folder,
        receiver=args.receiver,
        seeds=args.seed,
        classifier_style=style_classifier,
        classifier_object=object_classifier,
        prefix=args.prefix,
        baseline=baseline,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = (
            f"assets/datasets/receiver_results/"
            f"{args.prefix}_{args.receiver}.json"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote receiver {args.receiver} result to {destination}")


if __name__ == "__main__":
    main()
