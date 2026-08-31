'''What reproduces at the second seed, and what was one random draw.

Every cell of this campaign is a single image, so any one `clip_diff` carries the noise of one draw.
Running the same adapters at a second seed is what separates a property of the unlearning from a
property of the initial noise: the strongly affected entities should reproduce, and entities sitting
near the floor should not be expected to keep their order, because that order is reading noise.

For each entity it reports four numbers -- the worst checkpoint and the last checkpoint, at each of
the two seeds -- because with an effect this transient the endpoint alone answers the wrong question.
It also reports the rank agreement between the seeds over the worst values, which is the summary
statistic for "the same entities are affected", and the counts of damaged entities per seed.

Reads `assets/clip_diff_campaign.json` (or any scores file in the same shape, so the random-ten
control can be passed through the same code) and writes a markdown table beside a JSON record.

    python compare_seeds_sdxl.py --scores assets/clip_diff_campaign.json \\
        --output assets/seed_comparison_sdxl.md
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scipy.stats import spearmanr

import campaign_configuration as cfg

_HERE = Path(__file__).resolve().parent


def _worst(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''The most negative point of a trajectory, with the epoch it happened at.'''
    point = min(trajectory, key=lambda item: item["clip_diff"])
    return {"epoch": point["epoch"], "clip_diff": point["clip_diff"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-seed reproducibility of one campaign.")
    parser.add_argument("--scores", default="assets/clip_diff_campaign.json")
    parser.add_argument("--output", default="assets/seed_comparison_sdxl.md")
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]

    scores = json.loads((_HERE / args.scores).read_text(encoding="utf-8"))
    floor = cfg.noise_floor_standard_deviation()
    target: Optional[str] = scores.get("target")
    missing = [seed for seed in seeds if str(seed) not in scores["per_seed"]]
    assert not missing, f"{args.scores} has no scores for seeds {missing}"

    entities = list(scores["per_seed"][str(seeds[0])]["per_entity"])
    others = sorted(name for name in entities if name != target)
    ordered = ([target] if target in entities else []) + others

    rows: List[Dict[str, Any]] = []
    for name in ordered:
        row: Dict[str, Any] = {"entity": name, "is_target": name == target}
        for seed in seeds:
            trajectory = scores["per_seed"][str(seed)]["per_entity"][name]["trajectory"]
            worst = _worst(trajectory)
            row[f"worst_{seed}"] = worst["clip_diff"]
            row[f"worst_epoch_{seed}"] = worst["epoch"]
            row[f"final_{seed}"] = trajectory[-1]["clip_diff"]
        rows.append(row)

    agreement = spearmanr([row[f"worst_{seeds[0]}"] for row in rows],
                          [row[f"worst_{seeds[1]}"] for row in rows])
    damaged = {
        seed: {
            "at_worst": [row["entity"] for row in rows if row[f"worst_{seed}"] < -floor],
            "at_final": [row["entity"] for row in rows if row[f"final_{seed}"] < -floor],
        }
        for seed in seeds
    }
    reproduced = sorted(set(damaged[seeds[0]]["at_worst"]) & set(damaged[seeds[1]]["at_worst"]))
    either = sorted(set(damaged[seeds[0]]["at_worst"]) | set(damaged[seeds[1]]["at_worst"]))

    header = (f"| entity | worst at seed {seeds[0]} | epoch | worst at seed {seeds[1]} | epoch | "
              f"final at seed {seeds[0]} | final at seed {seeds[1]} |")
    lines = [
        f"# What reproduces at the second seed (floor {floor:.3f})", "",
        "Damage is `clip_diff` below minus the floor. The worst checkpoint is reported beside the last "
        "one because the effect is transient: an entity can be destroyed mid-training and back inside "
        "the floor at the end.", "",
        header, "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        marker = " (target)" if row["is_target"] else ""
        lines.append(
            f"| {row['entity'].replace('_', ' ')}{marker} | {row[f'worst_{seeds[0]}']:+.2f} | "
            f"{row[f'worst_epoch_{seeds[0]}']} | {row[f'worst_{seeds[1]}']:+.2f} | "
            f"{row[f'worst_epoch_{seeds[1]}']} | {row[f'final_{seeds[0]}']:+.2f} | "
            f"{row[f'final_{seeds[1]}']:+.2f} |")
    lines += [
        "",
        f"- entities damaged at their worst checkpoint: seed {seeds[0]} "
        f"{len(damaged[seeds[0]]['at_worst'])} of {len(rows)}, seed {seeds[1]} "
        f"{len(damaged[seeds[1]]['at_worst'])} of {len(rows)}",
        f"- entities damaged at the last checkpoint: seed {seeds[0]} "
        f"{len(damaged[seeds[0]]['at_final'])} of {len(rows)}, seed {seeds[1]} "
        f"{len(damaged[seeds[1]]['at_final'])} of {len(rows)}",
        f"- damaged at the worst checkpoint in BOTH seeds: {len(reproduced)} of {len(either)} "
        f"entities damaged in either",
        f"- rank agreement of the worst values across the two seeds: Spearman "
        f"{agreement.statistic:+.3f}, p = {agreement.pvalue:.2e}, n = {len(rows)}",
        "",
    ]
    output_path = _HERE / args.output
    output_path.write_text("\n".join(lines), encoding="utf-8")
    (output_path.with_suffix(".json")).write_text(json.dumps({
        "scores": args.scores, "seeds": seeds, "floor": floor, "rows": rows,
        "damaged": damaged, "reproduced_at_worst": reproduced,
        "spearman_worst": {"statistic": agreement.statistic, "p_value": agreement.pvalue},
    }, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"written: {output_path}")
    print("COMPARE_SEEDS_SDXL_DONE")


if __name__ == "__main__":
    main()
