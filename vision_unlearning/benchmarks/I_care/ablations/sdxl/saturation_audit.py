'''Measures how much colour the adapters drain, and whether that drain is what `clip_diff` is reading.

Two records disagreed and that is why this exists. The S4 outcome recorded "colour drains from every
adapted image, target and receivers alike ... it is a confound to name: part of any `clip_diff`
movement may be this desaturation rather than identity loss" -- read off contact sheets. The S5
campaign's epoch-5 Megawati Sukarnoputri is plainly still in full colour. So this measures instead of
looking: mean HSV saturation per image, the change from each entity's own off-baseline, and the
correlation of that change with `clip_diff` over every adapted image in the manifest.

It reads only artifacts -- `assets/campaign_seed{seed}.json` for the images and
`assets/clip_diff_campaign.json` for the scores -- and loads no model.

    python saturation_audit.py --seed 42

The output is the comparison, never a verdict: per-image levels, deltas, both correlation coefficients
with their p-values, and how many points sit inside the CLIP noise floor while their colour moves.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_CLIP_NOISE_FLOOR = 2.258  # standard deviation across six seeds, assets/clip_band_analysis.json


def mean_saturation(path: str) -> float:
    '''Mean of the HSV saturation channel, 0-255, over the whole image.'''
    return float(np.asarray(Image.open(path).convert("HSV"))[:, :, 1].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43])
    args = parser.parse_args()

    rows = json.loads((_OUT / f"campaign_seed{args.seed}.json").read_text(encoding="utf-8"))
    saturation: Dict[str, Dict[Any, float]] = {}
    for row in rows:
        saturation.setdefault(row["entity"], {})[row["epoch"]] = mean_saturation(row["path"])

    scores = json.loads((_OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    per_entity = scores["per_seed"][str(args.seed)]["per_entity"]
    epochs: List[Any] = scores["per_seed"][str(args.seed)]["epochs"]

    print(f"seed {args.seed}: {len(rows)} images, {len(saturation)} entities, epochs {epochs}")
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
    for entity, entry in per_entity.items():
        off = saturation[entity][None]
        for point in entry["trajectory"]:
            d_saturation.append(saturation[entity][point["epoch"]] - off)
            d_clip.append(float(point["clip_diff"]))

    r, p_r = pearsonr(d_saturation, d_clip)
    rho, p_rho = spearmanr(d_saturation, d_clip)
    inside = [(s, c) for s, c in zip(d_saturation, d_clip) if abs(c) <= _CLIP_NOISE_FLOOR]
    print(f"\nadapted images: {len(d_saturation)} = {len(saturation)} entities x {len(epochs)} epochs")
    print(f"Pearson r (saturation change vs clip_diff):  {r:+.3f}  p={p_r:.2e}")
    print(f"Spearman rho:                                {rho:+.3f}  p={p_rho:.2e}")
    print(f"points with |clip_diff| inside the {_CLIP_NOISE_FLOOR} floor: {len(inside)} of {len(d_clip)}; "
          f"their saturation change spans {min(s for s, _ in inside):+.1f} to {max(s for s, _ in inside):+.1f}")
    largest = min(zip(d_saturation, d_clip))
    print(f"largest colour loss in the set: {largest[0]:+.1f} saturation at clip_diff {largest[1]:+.2f}")


if __name__ == "__main__":
    main()
