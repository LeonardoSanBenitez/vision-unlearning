"""Planning spike: is a CLIP-based "the base model generates this entity recognisably" rule usable?

A planning probe, not evidence. It reads the first corpus's baseline images, whose noise convention the
regenerated corpus replaces, so its numbers only say whether a rule of this shape leaves enough entities
per task; the rule itself is computed on the regenerated baseline.

For every baseline image of a task it scores the image against all 100 generation prompts of that task with
the scorer `pipeline_06` uses (CLIP ViT-B/16; score = 100 * max(cosine, 0), as torchmetrics' CLIPScore),
then reports per task:
  - own-prompt score, mean score against the other 99 prompts, and their difference ("specificity");
  - the rank of the own prompt among the 100, per seed;
  - how many entities pass "own prompt in the top K in at least M of the 4 seeds" for a few (K, M);
  - the floor 0.5 x mean specificity over random five-entity panels drawn from the passing entities.

Usage:
  python <this file> --images-root <folder holding generated_{task}_baseline/> --output <json>
The images are the files `off_{seed}_{prompt}.png`; the prompt is read from the file name.
"""
import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch16"
TASKS = ["people", "scenes", "breeds"]
SEEDS = [42, 43, 44, 45]


def load_task(folder: str) -> Dict[str, Dict[int, str]]:
    """Map prompt -> seed -> image path, from `off_{seed}_{prompt}.png` file names."""
    images: Dict[str, Dict[int, str]] = defaultdict(dict)
    for name in os.listdir(folder):
        if not (name.startswith("off_") and name.endswith(".png")):
            continue
        seed_text, prompt = name[len("off_"):-len(".png")].split("_", 1)
        images[prompt][int(seed_text)] = os.path.join(folder, name)
    return images


@torch.no_grad()
def score_task(model: CLIPModel, processor: CLIPProcessor, images: Dict[str, Dict[int, str]], device: str) -> dict:
    prompts = sorted(images)
    text = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_emb = model.get_text_features(**text)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    per_entity = {}
    for i, prompt in enumerate(prompts):
        rows = []
        for seed in SEEDS:
            image = Image.open(images[prompt][seed]).convert("RGB")
            pixel = processor(images=image, return_tensors="pt").to(device)
            image_emb = model.get_image_features(**pixel)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            scores = (100 * (image_emb @ text_emb.T).clamp(min=0)).squeeze(0).tolist()
            own = scores[i]
            others = [s for j, s in enumerate(scores) if j != i]
            rank = 1 + sum(1 for j, s in enumerate(scores) if j != i and s > own)
            rows.append({"seed": seed, "own": own, "others_mean": statistics.fmean(others),
                         "specificity": own - statistics.fmean(others), "rank": rank})
        per_entity[prompt] = rows
    return per_entity


def summarise(per_entity: dict) -> dict:
    spec = [statistics.fmean(r["specificity"] for r in rows) for rows in per_entity.values()]
    own = [statistics.fmean(r["own"] for r in rows) for rows in per_entity.values()]
    passing = {}
    for k in (1, 3, 5, 10):
        for m in (2, 3, 4):
            names = [p for p, rows in per_entity.items() if sum(r["rank"] <= k for r in rows) >= m]
            passing[f"top{k}_in_{m}of4"] = len(names)
    eligible = [p for p, rows in per_entity.items() if sum(r["rank"] <= 5 for r in rows) >= 3]
    rng = random.Random(0)
    floors = []
    if len(eligible) >= 5:
        for _ in range(1000):
            panel = rng.sample(eligible, 5)
            floors.append(0.5 * statistics.fmean(
                statistics.fmean(r["specificity"] for r in per_entity[p]) for p in panel))
    def quant(xs: List[float], q: float) -> float:
        return sorted(xs)[int(q * (len(xs) - 1))]
    return {
        "n_entities": len(per_entity),
        "own_score_mean": statistics.fmean(own),
        "specificity": {"min": min(spec), "median": statistics.median(spec), "max": max(spec),
                        "n_nonpositive": sum(s <= 0 for s in spec)},
        "entities_passing": passing,
        "floor_over_random_panels_of_eligible_top5_in_3of4": (
            {"p05": quant(floors, 0.05), "median": quant(floors, 0.5), "p95": quant(floors, 0.95)} if floors else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL)
    result = {}
    for task in TASKS:
        images = load_task(os.path.join(args.images_root, f"generated_{task}_baseline"))
        complete = {p: s for p, s in images.items() if all(seed in s for seed in SEEDS)}
        per_entity = score_task(model, processor, complete, device)
        result[task] = {"summary": summarise(per_entity), "per_entity": per_entity}
        print(task, json.dumps(result[task]["summary"]), flush=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    print("SPIKE_DONE", args.output)


if __name__ == "__main__":
    main()
