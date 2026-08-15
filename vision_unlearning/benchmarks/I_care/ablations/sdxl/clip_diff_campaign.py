'''Scores the whole Stable Diffusion XL campaign with CLIP, from the images already on disk.

This is the metric layer the validation report (plan stage S7.5) and the figures (S8) both read. It
loads no Stable Diffusion XL pipeline and generates nothing: it reads `assets/campaign_seed{seed}.json`,
opens each image, and scores it with the library's own `MetricImageTextSimilarity(metrics=['clip'])` --
the same primitive `pipeline_06` and the every-epoch ablation use, so the numbers sit on the same scale
as the existing Stable Diffusion 1.4 curves.

Two text conditions are scored for every image, not one:

* `clip_diff = clip_on - clip_off` against the entity's OWN prompt ("An image of Mark Philippoussis").
  This is the canonical interference metric; more negative means the image agrees less with the prompt.
* `clip_overwrite_diff` against the OVERWRITE concept the trainer distilled toward ("An image of a
  child"). SPARE does not merely degrade the target, it moves it toward a replacement concept, so this
  is the direction the training objective actually pushes in. It is a diagnostic, reported alongside
  `clip_diff` and never substituted for it.

Scoring both is what makes the metric question answerable: if an image visibly changes while
`clip_diff` stays flat, the second column says whether the signal went somewhere else or whether CLIP
saw nothing at all.

    python clip_diff_campaign.py --seeds 42,43
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_SELECTION = _HERE.parent / "every_epoch" / "assets" / "selection_people.json"
_TASK: Literal["people"] = "people"
_METHOD: Literal["distil"] = "distil"


def _epoch_key(epoch: Any) -> Any:
    """Sort key placing the off-baseline (`epoch: null`) before every trained checkpoint."""
    return (epoch is not None, epoch)


def main() -> None:
    from PIL import Image

    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    parser = argparse.ArgumentParser(description="CLIP scores for every campaign image, both text conditions.")
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    target_name: str = selection["target"]["name"]
    receiver_names: List[str] = [receiver["name"] for receiver in selection["receivers"]]
    entities: List[str] = [target_name] + receiver_names

    # Own prompt and overwrite prompt come from the same helper the generation used, never retyped.
    own_prompt: Dict[str, str] = {}
    overwrite_prompt: Dict[str, str] = {}
    for name in entities:
        display, overwrite = get_target_overwrite(_TASK, _METHOD, name)
        own_prompt[name] = f"An image of {display}"
        overwrite_prompt[name] = f"An image of {overwrite}"

    metric = MetricImageTextSimilarity(metrics=["clip"])
    result: Dict[str, Any] = {
        "task": _TASK,
        "method": _METHOD,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "seeds": seeds,
        "target": target_name,
        "receivers": receiver_names,
        "own_prompt": own_prompt,
        "overwrite_prompt": overwrite_prompt,
        "per_seed": {},
    }
    n_images_scored = 0

    for seed in seeds:
        manifest_path = _OUT / f"campaign_seed{seed}.json"
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_entity: Dict[str, Dict[Any, str]] = {}
        for row in rows:
            by_entity.setdefault(row["entity"], {})[row["epoch"]] = row["path"]

        missing = sorted(set(entities) - set(by_entity))
        assert not missing, f"seed {seed} manifest is missing entities: {missing}"

        epochs = sorted({row["epoch"] for row in rows}, key=_epoch_key)
        assert epochs[0] is None, f"seed {seed} manifest has no off-baseline row"
        trained_epochs = [epoch for epoch in epochs if epoch is not None]

        per_entity: Dict[str, Any] = {}
        for name in entities:
            ordered = [None] + list(trained_epochs)
            images: List[Any] = [Image.open(by_entity[name][epoch]).convert("RGB") for epoch in ordered]
            n_images_scored += len(images)
            own = [float(s["clip"]) for s in metric.score_batch_same_text(images, own_prompt[name])]
            over = [float(s["clip"]) for s in metric.score_batch_same_text(images, overwrite_prompt[name])]
            clip_off, clip_off_overwrite = own[0], over[0]
            per_entity[name] = {
                "clip_off": clip_off,
                "clip_off_overwrite": clip_off_overwrite,
                "trajectory": [
                    {
                        "epoch": epoch,
                        "path": by_entity[name][epoch],
                        "clip_on": own[index + 1],
                        "clip_diff": own[index + 1] - clip_off,
                        "clip_on_overwrite": over[index + 1],
                        "clip_overwrite_diff": over[index + 1] - clip_off_overwrite,
                    }
                    for index, epoch in enumerate(trained_epochs)
                ],
            }
        result["per_seed"][str(seed)] = {"epochs": trained_epochs, "per_entity": per_entity}

    output_path = _OUT / "clip_diff_campaign.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # The check prints its own denominators; a later reader gets the arithmetic, not a word.
    expected = len(seeds) * len(entities) * (1 + len(result["per_seed"][str(seeds[0])]["epochs"]))
    print(f"entities: {len(entities)} (1 target + {len(receiver_names)} receivers)")
    print(f"seeds: {seeds}; checkpoints per seed: {len(result['per_seed'][str(seeds[0])]['epochs'])}")
    print(f"images scored: {n_images_scored}; expected {len(seeds)} x {len(entities)} x (1 off + "
          f"{len(result['per_seed'][str(seeds[0])]['epochs'])} epochs) = {expected}")
    print(f"images scored equals expected: {n_images_scored == expected}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
