'''Does the damage stay? The one cross-model quantity that IS comparable between the two models.

`clip_diff` magnitudes cannot be compared across base models -- different model, autoencoder,
resolution and noise floor -- and every figure in this ablation says so. But *recovery* can be, because
it is computed inside one model and divided by that model's own worst point:

    recovery = 1 - |clip_diff at the last checkpoint| / max over the trajectory of |clip_diff|

A recovery near 1 means the entity ended where it started after having been moved; near 0 means it
ended as damaged as it ever was. Both numerator and denominator come from the same model, the same
entity and the same seed, so the ratio carries none of the scale that makes the raw values
incomparable. What it does carry is the shape of the trajectory, which is exactly the thing this
ablation set out to compare.

Two caveats stated in the figure itself rather than left to a reader: an entity whose peak is inside
the noise floor never moved at all, so its recovery is meaningless and it is drawn hollow; and the
peak is over the sampled checkpoints, not over training, so a spike between two checkpoints is
invisible to both models equally.

    python cross_model_recovery.py --seed 42 --output assets/cross_model_recovery_seed42.png
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import campaign_configuration as cfg
import sd14_campaign

_HERE = Path(__file__).resolve().parent


def recovery(trajectory: List[float]) -> Tuple[float, float, float]:
    '''(peak absolute value, final absolute value, recovery fraction) of one trajectory.'''
    peak = max(abs(value) for value in trajectory)
    final = abs(trajectory[-1])
    return peak, final, (1.0 - final / peak) if peak > 0 else 0.0


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Peak against final clip_diff, both base models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scores = json.loads((cfg.OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    block = scores["per_seed"][str(args.seed)]
    target = scores["target"]
    entities = [target] + [name for name in block["per_entity"] if name != target]

    sdxl_floor = cfg.noise_floor_standard_deviation()
    sd14_floor = sd14_campaign.noise_floor_summary()["median"]

    rows: List[Dict[str, Any]] = []
    for entity in entities:
        sdxl_trajectory = [point["clip_diff"] for point in block["per_entity"][entity]["trajectory"]]
        _, sd14_values, sd14_epochs = sd14_campaign.entity_cells(args.seed, entity)
        sd14_trajectory = [sd14_values[epoch] for epoch in sd14_epochs]
        sdxl_peak, sdxl_final, sdxl_recovery = recovery(sdxl_trajectory)
        sd14_peak, sd14_final, sd14_recovery = recovery(sd14_trajectory)
        rows.append({
            "entity": entity, "is_target": entity == target,
            "sdxl_peak": sdxl_peak, "sdxl_final": sdxl_final, "sdxl_recovery": sdxl_recovery,
            "sdxl_moved": sdxl_peak > sdxl_floor,
            "sd14_peak": sd14_peak, "sd14_final": sd14_final, "sd14_recovery": sd14_recovery,
            "sd14_moved": sd14_peak > sd14_floor,
        })

    rows.sort(key=lambda row: (not row["is_target"], -row["sdxl_peak"]))
    labels = [row["entity"].replace("_", " ") for row in rows]
    positions = list(range(len(rows)))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), squeeze=False)
    for panel, (model, floor, title) in enumerate([
        ("sd14", sd14_floor, f"stable diffusion 1.4, resolution {sd14_campaign.RESOLUTION}"),
        ("sdxl", sdxl_floor, f"stable diffusion xl, resolution {cfg.GENERATION_RESOLUTION}"),
    ]):
        axis = axes[0][panel]
        axis.axvspan(0, floor, color="0.88", zorder=0)
        for position, row in zip(positions, rows):
            peak, final, moved = row[f"{model}_peak"], row[f"{model}_final"], row[f"{model}_moved"]
            axis.plot([final, peak], [position, position], color="0.5", linewidth=1.2, zorder=1)
            axis.scatter([peak], [position], marker="o", s=42, color="tab:red", zorder=2)
            axis.scatter([final], [position], marker="o", s=42, zorder=3,
                         facecolors="tab:blue" if moved else "none", edgecolors="tab:blue")
        moved_count = sum(1 for row in rows if row[f"{model}_moved"])
        recovered = [row for row in rows if row[f"{model}_moved"] and row[f"{model}_recovery"] >= 0.5]
        axis.set_title(f"{title}\n"
                       f"peak outside the floor: {moved_count} of {len(rows)} entities\n"
                       f"of those, at least half recovered by the last checkpoint: "
                       f"{len(recovered)} of {moved_count}", fontsize=9)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("absolute clip_diff")
        axis.grid(alpha=0.25, axis="x")
    axes[0][0].scatter([], [], color="tab:red", label="worst checkpoint")
    axes[0][0].scatter([], [], color="tab:blue", label="last checkpoint")
    axes[0][0].scatter([], [], facecolors="none", edgecolors="tab:blue",
                       label="last checkpoint, entity never left the floor")
    axes[0][0].legend(fontsize=7, loc="lower right")

    figure.suptitle(
        f"task={cfg.TASK} | method=spare | seed={args.seed} | entities={len(rows)} | "
        f"checkpoints={len(block['epochs'])} | shaded band = that model's own noise floor\n"
        f"absolute clip_diff at the worst checkpoint and at the last one, per entity, per base model. "
        f"The two panels' x axes are NOT comparable; what is comparable is how far each point moves "
        f"back toward its own floor.",
        fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output_path = _HERE / args.output
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    header = ("| entity | 1.4 peak | 1.4 final | 1.4 recovery | XL peak | XL final | XL recovery |")
    table = [f"# peak against final absolute clip_diff, seed {args.seed}", "",
             f"Noise floors: stable diffusion 1.4 {sd14_floor:.2f} (median over ten entities, two "
             f"seeds), stable diffusion xl {sdxl_floor:.2f} (one standard deviation, one entity, six "
             f"seeds). Recovery = 1 - final / peak, computed inside one model.", "",
             header, "|---|---|---|---|---|---|---|"]
    for row in rows:
        marker = " (target)" if row["is_target"] else ""
        table.append(
            f"| {row['entity'].replace('_', ' ')}{marker} | {row['sd14_peak']:.2f} | "
            f"{row['sd14_final']:.2f} | {row['sd14_recovery']:+.0%} | {row['sdxl_peak']:.2f} | "
            f"{row['sdxl_final']:.2f} | {row['sdxl_recovery']:+.0%} |")
    table_path = output_path.with_suffix(".md")
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    (output_path.with_suffix(".json")).write_text(
        json.dumps({"seed": args.seed, "sd14_floor": sd14_floor, "sdxl_floor": sdxl_floor,
                    "rows": rows}, indent=2), encoding="utf-8")

    for model, floor in [("sd14", sd14_floor), ("sdxl", sdxl_floor)]:
        moved = [row for row in rows if row[f"{model}_moved"]]
        recovered = [row for row in moved if row[f"{model}_recovery"] >= 0.5]
        mean_recovery = sum(row[f"{model}_recovery"] for row in moved) / len(moved) if moved else 0.0
        print(f"{model}: {len(moved)} of {len(rows)} entities left the {floor:.2f} floor at some "
              f"checkpoint; {len(recovered)} of {len(moved)} recovered at least half; "
              f"mean recovery over those that moved {mean_recovery:+.0%}")
    print(f"written: {output_path}")
    print(f"written: {table_path}")
    print("CROSS_MODEL_RECOVERY_DONE")


if __name__ == "__main__":
    main()
