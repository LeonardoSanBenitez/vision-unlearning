'''Scores the whole Stable Diffusion XL campaign with CLIP, from the images already on disk.

This is the metric layer the validation report (plan stage S9) and the figures (S10) both read. It
loads no Stable Diffusion XL pipeline and generates nothing: it reads `assets/campaign_seed{seed}.json`,
opens each image, and scores it. The scoring itself -- both text conditions, and why there are two --
is `clip_scoring.py`, which `random_ten_control.py` shares, so this file only assembles the campaign's
own envelope around it.

    python clip_diff_campaign.py --seeds 42,43
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import clip_scoring

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP scores for every campaign image, both text conditions.")
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    selection = json.loads(clip_scoring.SELECTION.read_text(encoding="utf-8"))
    target_name: str = selection["target"]["name"]
    receiver_names: List[str] = [receiver["name"] for receiver in selection["receivers"]]
    entities: List[str] = [target_name] + receiver_names

    own_prompt, overwrite_prompt = clip_scoring.prompts_for(entities)
    result: Dict[str, Any] = {
        "task": clip_scoring.TASK,
        "method": clip_scoring.METHOD,
        "model": clip_scoring.MODEL_ID,
        "seeds": seeds,
        "target": target_name,
        "receivers": receiver_names,
        "own_prompt": own_prompt,
        "overwrite_prompt": overwrite_prompt,
        "per_seed": {},
    }
    n_images_scored = 0

    for seed in seeds:
        rows = json.loads((_OUT / f"campaign_seed{seed}.json").read_text(encoding="utf-8"))
        per_entity, trained_epochs, scored = clip_scoring.score_entities(rows, entities)
        n_images_scored += scored
        result["per_seed"][str(seed)] = {"epochs": trained_epochs, "per_entity": per_entity}

    output_path = _OUT / "clip_diff_campaign.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # The check prints its own denominators; a later reader gets the arithmetic, not a word.
    n_epochs = len(result["per_seed"][str(seeds[0])]["epochs"])
    expected = len(seeds) * len(entities) * (1 + n_epochs)
    print(f"entities: {len(entities)} (1 target + {len(receiver_names)} receivers)")
    print(f"seeds: {seeds}; checkpoints per seed: {n_epochs}")
    print(f"images scored: {n_images_scored}; expected {len(seeds)} x {len(entities)} x (1 off + "
          f"{n_epochs} epochs) = {expected}")
    print(f"images scored equals expected: {n_images_scored == expected}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
