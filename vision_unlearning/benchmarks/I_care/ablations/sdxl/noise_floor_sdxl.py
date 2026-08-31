'''Plan stage S7 -- the Stable Diffusion XL noise floor of `clip_diff`.

WHAT A NOISE FLOOR IS, since the term is load-bearing in this task and is not general vocabulary.

`clip_diff` for an entity is `clip_on - clip_off`: the CLIP agreement between the image and its prompt
after unlearning, minus the same quantity before. A non-zero value is supposed to mean "unlearning
changed this entity". But two images of the SAME prompt from the SAME unmodified model already differ,
because they start from a different random noise draw -- a different face, a different pose, a different
background, all equally valid renderings of "An image of Mark Philippoussis". Those two images get two
different CLIP scores, and their difference was produced by nothing but the random draw.

That difference is the noise floor: the smallest `clip_diff` that can carry any meaning at all. A
measured `clip_diff` smaller than the floor is indistinguishable from having generated the same
unmodified model twice, and no acceptability gate should ever be set below it.

Two seeds give one difference per entity. That is an indication of scale, not an error bar (n = 2 per
entity), and it is labelled that way wherever it is used -- the same caveat `every_epoch/
metric_progression.py` carries for the Stable Diffusion 1.4 floor this one is read beside.

The method mirrors `every_epoch/metric_progression.py` (`noise_floor_summary`, and the same-prompt-two-
seeds measurement) rather than importing it: that script's paths and image names are Stable Diffusion
1.4-specific. Reads only the OFF-baseline images already on disk; generates nothing.

    python noise_floor_sdxl.py --seeds 42,43
'''
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Literal

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_SELECTION = _HERE.parent / "every_epoch" / "assets" / "selection_people.json"
_TASK: Literal["people"] = "people"
_METHOD: Literal["distil"] = "distil"


def noise_floor_summary(per_entity: Dict[str, float]) -> Dict[str, float]:
    """Median, maximum and minimum of the base-model score differences between the two seeds.

    Copied deliberately from `every_epoch/metric_progression.py` so both models' floors are computed by
    the same formula and can be placed in one table.
    """
    values = sorted(per_entity.values())
    return {"median": statistics.median(values), "maximum": max(values), "minimum": min(values),
            "number_of_entities": len(values)}


def main() -> None:
    from PIL import Image

    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    parser = argparse.ArgumentParser(description="Stable Diffusion XL noise floor of clip_diff, from off-baselines.")
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    target_name: str = selection["target"]["name"]
    entities: List[str] = [target_name] + [receiver["name"] for receiver in selection["receivers"]]
    prompt_of = {name: f"An image of {get_target_overwrite(_TASK, _METHOD, name)[0]}" for name in entities}

    off_path_of: Dict[int, Dict[str, str]] = {}
    for seed in seeds:
        rows = json.loads((_OUT / f"campaign_seed{seed}.json").read_text(encoding="utf-8"))
        off_path_of[seed] = {row["entity"]: row["path"] for row in rows if row["epoch"] is None}
        missing = sorted(set(entities) - set(off_path_of[seed]))
        assert not missing, f"seed {seed} has no off-baseline for: {missing}"

    metric = MetricImageTextSimilarity(metrics=["clip"])
    per_entity: Dict[str, float] = {}
    base_scores: Dict[str, Dict[str, Any]] = {}
    for name in entities:
        by_seed: Dict[str, float] = {}
        for seed in seeds:
            image = Image.open(off_path_of[seed][name]).convert("RGB")
            by_seed[str(seed)] = float(metric.score_batch_same_text([image], prompt_of[name])[0]["clip"])
        base_scores[name] = {
            "per_seed_clip": by_seed,
            "paths": {str(seed): off_path_of[seed][name] for seed in seeds},
        }
        per_entity[name] = abs(by_seed[str(seeds[0])] - by_seed[str(seeds[-1])])

    summary = noise_floor_summary(per_entity)
    result: Dict[str, Any] = {
        "task": _TASK,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "seeds": seeds,
        "prompt_of": prompt_of,
        "per_entity": per_entity,
        "summary": summary,
        "base_scores": base_scores,
    }
    output_path = _OUT / "noise_floor_people_sdxl.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    reference_path = _HERE.parent / "every_epoch" / "assets" / "noise_floor_people.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["summary"]
    print(f"entities compared: {summary['number_of_entities']}; seeds: {seeds}; one difference per entity")
    for name in sorted(per_entity, key=lambda k: -per_entity[k]):
        scores = base_scores[name]["per_seed_clip"]
        print(f"  {name:24s} clip seed{seeds[0]} {scores[str(seeds[0])]:7.3f} | "
              f"clip seed{seeds[-1]} {scores[str(seeds[-1])]:7.3f} | absolute difference {per_entity[name]:6.3f}")
    print(f"stable diffusion xl floor: median {summary['median']:.3f}, "
          f"maximum {summary['maximum']:.3f}, minimum {summary['minimum']:.3f}")
    print(f"stable diffusion 1.4 floor (every_epoch, same formula): median {reference['median']:.3f}, "
          f"maximum {reference['maximum']:.3f}, minimum {reference['minimum']:.3f}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
