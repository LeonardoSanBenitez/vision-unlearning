'''Measures how much colour the adapters drain, and whether that drain is what `clip_diff` is reading.

Two records disagreed and that is why this exists. The S4 outcome recorded "colour drains from every
adapted image, target and receivers alike ... it is a confound to name: part of any `clip_diff`
movement may be this desaturation rather than identity loss" -- read off contact sheets. The S5
campaign's epoch-5 Megawati Sukarnoputri is plainly still in full colour. So this measures instead of
looking: mean HSV saturation per image, the change from each entity's own off-baseline, and the
correlation of that change with `clip_diff` over every adapted image in the run.

It runs against EITHER base model, which is the point of the `--model` argument: reading the Stable
Diffusion XL numbers alone cannot tell "Stable Diffusion XL desaturates" from "this unlearning method
desaturates", and the Stable Diffusion 1.4 campaign is 140 images already on disk that answer it for
nothing. Whichever is chosen, the analysis below is the same code on the same scale.

It reads only artifacts and loads no model.

    python saturation_audit.py --model sdxl --seed 42
    python saturation_audit.py --model sd14 --seed 42

The output is the comparison, never a verdict: per-image levels, deltas, both correlation coefficients
with their p-values, and how many points sit inside the CLIP noise floor while their colour moves.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import campaign_configuration as cfg
import sd14_campaign

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"


def mean_saturation(path: str) -> float:
    '''Mean of the HSV saturation channel, 0-255, over the whole image.'''
    return float(np.asarray(Image.open(path).convert("HSV"))[:, :, 1].mean())


def _sdxl_run(seed: int) -> Tuple[Dict[str, Dict[Any, str]], Dict[str, Dict[Any, float]], List[int]]:
    '''Images and `clip_diff` of the Stable Diffusion XL campaign, keyed entity -> epoch.'''
    rows = json.loads((_OUT / f"campaign_seed{seed}.json").read_text(encoding="utf-8"))
    scores = json.loads((_OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    seed_block = scores["per_seed"][str(seed)]
    paths: Dict[str, Dict[Any, str]] = {}
    for row in rows:
        paths.setdefault(row["entity"], {})[row["epoch"]] = row["path"]
    clip_diff: Dict[str, Dict[Any, float]] = {
        entity: {point["epoch"]: float(point["clip_diff"]) for point in payload["trajectory"]}
        for entity, payload in seed_block["per_entity"].items()
    }
    return paths, clip_diff, list(seed_block["epochs"])


def _sd14_run(seed: int) -> Tuple[Dict[str, Dict[Any, str]], Dict[str, Dict[Any, float]], List[int]]:
    '''The same, for the every-epoch Stable Diffusion 1.4 campaign this one is compared against.'''
    paths: Dict[str, Dict[Any, str]] = {}
    clip_diff: Dict[str, Dict[Any, float]] = {}
    epochs: List[int] = []
    for entity in sd14_campaign.generation_order(seed):
        entity_paths, entity_values, epochs = sd14_campaign.entity_cells(seed, entity)
        paths[entity] = {epoch: str(path) for epoch, path in entity_paths.items()}
        clip_diff[entity] = dict(entity_values)
    return paths, clip_diff, epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Colour drain per epoch, and its relation to clip_diff.")
    parser.add_argument("--model", choices=["sdxl", "sd14"], default="sdxl")
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43])
    args = parser.parse_args()

    paths, clip_diff, epochs = _sdxl_run(args.seed) if args.model == "sdxl" else _sd14_run(args.seed)
    model_id = cfg.MODEL_ID if args.model == "sdxl" else sd14_campaign.MODEL_ID
    resolution = cfg.GENERATION_RESOLUTION if args.model == "sdxl" else sd14_campaign.RESOLUTION
    # Each model is judged against ITS OWN floor, and the two are built differently -- the Stable
    # Diffusion XL one is a standard deviation of one entity over six seeds, the Stable Diffusion 1.4
    # one is the median absolute difference of ten entities over two seeds. Using one model's floor on
    # the other model's numbers would be the cross-model comparison this ablation keeps refusing to
    # make, so the construction is printed beside the number every time.
    if args.model == "sdxl":
        floor = cfg.noise_floor_standard_deviation()
        floor_description = "one standard deviation of one entity over six seeds"
    else:
        floor = sd14_campaign.noise_floor_summary()["median"]
        floor_description = "median absolute difference of ten entities over two seeds"

    saturation: Dict[str, Dict[Any, float]] = {
        entity: {epoch: mean_saturation(path) for epoch, path in by_epoch.items()}
        for entity, by_epoch in paths.items()
    }
    n_images = sum(len(by_epoch) for by_epoch in paths.values())

    print(f"model {model_id} at {resolution} pixels, seed {args.seed}: {n_images} images, "
          f"{len(saturation)} entities, epochs {epochs}")
    header = " ".join(f"{'e' + str(e):>8}" for e in epochs)
    deltas_header = " ".join(f"{'d' + str(e):>8}" for e in epochs)
    print(f"\n{'entity':<24} {'off':>8} {header}   {deltas_header}")
    for entity, values in saturation.items():
        off = values[None]
        levels = " ".join(f"{values[e]:8.1f}" for e in epochs)
        deltas = " ".join(f"{values[e] - off:+8.1f}" for e in epochs)
        print(f"{entity:<24} {off:8.1f} {levels}   {deltas}")

    d_saturation: List[float] = []
    d_clip: List[float] = []
    for entity, by_epoch in clip_diff.items():
        off = saturation[entity][None]
        for epoch in epochs:
            d_saturation.append(saturation[entity][epoch] - off)
            d_clip.append(by_epoch[epoch])

    r, p_r = pearsonr(d_saturation, d_clip)
    rho, p_rho = spearmanr(d_saturation, d_clip)
    inside = [(s, c) for s, c in zip(d_saturation, d_clip) if abs(c) <= floor]
    mean_delta = float(np.mean(d_saturation))
    print(f"\nadapted images: {len(d_saturation)} = {len(saturation)} entities x {len(epochs)} epochs")
    print(f"mean saturation change over all adapted images: {mean_delta:+.1f} (0-255 scale)")
    print(f"Pearson r (saturation change against clip_diff):  {r:+.3f}  p={p_r:.2e}")
    print(f"Spearman rho:                                     {rho:+.3f}  p={p_rho:.2e}")
    print(f"points with |clip_diff| inside this model's own {floor:.3f} floor ({floor_description}): "
          f"{len(inside)} of {len(d_clip)}; "
          f"their saturation change spans {min(s for s, _ in inside):+.1f} to {max(s for s, _ in inside):+.1f}")
    largest = min(zip(d_saturation, d_clip))
    print(f"largest colour loss in the set: {largest[0]:+.1f} saturation at clip_diff {largest[1]:+.2f}")

    record = {
        "model": model_id, "resolution": resolution, "seed": args.seed, "epochs": epochs,
        "n_images": n_images, "n_adapted_points": len(d_saturation),
        "noise_floor": floor, "noise_floor_description": floor_description,
        "mean_saturation_change": mean_delta,
        "pearson_r": r, "pearson_p": p_r, "spearman_rho": rho, "spearman_p": p_rho,
        "points_inside_floor": len(inside),
        "saturation": {entity: {str(epoch): value for epoch, value in values.items()}
                       for entity, values in saturation.items()},
    }
    record_path = _OUT / f"saturation_audit_{args.model}_seed{args.seed}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"written: {record_path}")
    print(f"SATURATION_AUDIT_DONE model={args.model} seed={args.seed} points={len(d_saturation)}")


if __name__ == "__main__":
    main()
