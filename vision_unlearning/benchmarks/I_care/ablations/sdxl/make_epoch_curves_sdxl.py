'''S10 of PLAN-TASK-2026-08-12-SDXL: the trajectories behind the epoch grid, one line per entity.

The grid shows what the images look like; this shows the same numbers as curves, which is what makes
the TIMING visible -- when the target leaves the noise floor, which receivers follow it, and whether
they come back. It reads the scores file and nothing else: no image is opened, no model is loaded.

Two panels, because this campaign has two questions and they are not the same question:

* `clip_diff` against the entity's own prompt -- the canonical interference metric, and the one the
  Stable Diffusion 1.4 curves are drawn in, so the two are directly comparable in shape;
* `clip_overwrite_diff` against the concept the trainer distilled toward ("An image of a child") --
  the direction the objective actually pushes in. An entity that merely degrades and an entity that
  is being replaced by the overwrite concept look the same in the left panel and different in the
  right one.

The shaded band is the measured noise floor (one standard deviation of one entity over six seeds),
read from the artifact that measured it. A line inside the band is a line that has not moved.

    python make_epoch_curves_sdxl.py --scores assets/clip_diff_campaign.json --seed 42 \\
        --output assets/epoch_curves_sdxl_seed42.png

Writes the figure and, beside it, a markdown table of every value drawn -- so a number quoted in a
report comes from a file rather than from reading a line off a chart.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import campaign_configuration as cfg

_HERE = Path(__file__).resolve().parent
_TARGET_STYLE: Dict[str, Any] = {"color": "black", "linewidth": 2.6, "linestyle": "-", "zorder": 5}


def _series(per_entity: Dict[str, Any], entity: str, field: str) -> List[float]:
    '''One entity's trajectory in `field`, epoch order, with the off-baseline's implicit zero first.'''
    return [0.0] + [float(point[field]) for point in per_entity[entity]["trajectory"]]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="clip_diff and clip_overwrite_diff against training epoch.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scores = json.loads((_HERE / args.scores).read_text(encoding="utf-8"))
    seed_block = scores["per_seed"][str(args.seed)]
    epochs: List[int] = seed_block["epochs"]
    per_entity: Dict[str, Any] = seed_block["per_entity"]
    target: Optional[str] = scores.get("target")
    floor = cfg.noise_floor_standard_deviation()

    # Epoch 0 is the off-baseline: the difference of an image with itself, zero by definition. It is
    # drawn so every line starts from the same point, and it is why the axis starts at 0 rather than 1.
    axis_epochs = [0] + list(epochs)
    ordered = ([target] if target else []) + sorted(
        (name for name in per_entity if name != target),
        key=lambda name: per_entity[name]["trajectory"][-1]["clip_diff"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    colours = plt.get_cmap("tab10")
    fields = [("clip_diff", "against each entity's own prompt"),
              ("clip_overwrite_diff", "against the overwrite concept the trainer distilled toward")]
    for panel, (field, subtitle) in enumerate(fields):
        axis = axes[0][panel]
        axis.axhspan(-floor, floor, color="0.85", zorder=0)
        axis.axhline(0.0, color="0.4", linewidth=0.8, zorder=1)
        for index, entity in enumerate(ordered):
            values = _series(per_entity, entity, field)
            style = dict(_TARGET_STYLE) if entity == target else {
                "color": colours(index % 10), "linewidth": 1.4,
            }
            axis.plot(axis_epochs, values, marker="o", markersize=3,
                      label=entity.replace("_", " "), **style)
        outside = sum(1 for entity in ordered
                      if abs(per_entity[entity]["trajectory"][-1][field]) > floor)
        axis.set_title(f"{field}, {subtitle}\n"
                       f"entities outside the noise floor at epoch {epochs[-1]}: "
                       f"{outside} of {len(ordered)}", fontsize=10)
        axis.set_xlabel("training epoch")
        axis.set_ylabel(f"{field}")
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xticks(axis_epochs)
        axis.set_xticklabels([str(epoch) for epoch in axis_epochs], fontsize=7)
        axis.grid(alpha=0.25)
    axes[0][1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    figure.suptitle(
        f"model=stable-diffusion-xl-base-1.0 | task={cfg.TASK} | method=spare | seed={args.seed} | "
        f"resolution={cfg.GENERATION_RESOLUTION} | entities={len(ordered)} | checkpoints={len(epochs)}\n"
        f"shaded band = measured noise floor, one standard deviation of one entity over six seeds "
        f"= {floor:.3f} | epoch 0 is the off-baseline and is zero by construction",
        fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output_path = _HERE / args.output
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    header = "| entity | " + " | ".join(f"epoch {epoch}" for epoch in epochs) + " |"
    separator = "|---" * (len(epochs) + 1) + "|"
    lines = [f"# clip_diff by epoch, seed {args.seed}, noise floor {floor:.3f}", "", header, separator]
    for entity in ordered:
        cells = [f"{point['clip_diff']:+.2f}" for point in per_entity[entity]["trajectory"]]
        marker = " (target)" if entity == target else ""
        lines.append(f"| {entity.replace('_', ' ')}{marker} | " + " | ".join(cells) + " |")
    lines += ["", f"# clip_overwrite_diff by epoch, seed {args.seed}", "", header, separator]
    for entity in ordered:
        cells = [f"{point['clip_overwrite_diff']:+.2f}" for point in per_entity[entity]["trajectory"]]
        marker = " (target)" if entity == target else ""
        lines.append(f"| {entity.replace('_', ' ')}{marker} | " + " | ".join(cells) + " |")
    table_path = output_path.with_suffix(".md")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"entities plotted: {len(ordered)}; checkpoints: {len(epochs)}")
    print(f"noise floor (one standard deviation): {floor:.3f}")
    for field, _ in fields:
        outside_entities = [entity for entity in ordered
                            if abs(per_entity[entity]["trajectory"][-1][field]) > floor]
        print(f"{field}: {len(outside_entities)} of {len(ordered)} outside the floor at epoch "
              f"{epochs[-1]}: {outside_entities}")
    print(f"written: {output_path}")
    print(f"written: {table_path}")
    print("EPOCH_CURVES_SDXL_DONE")


if __name__ == "__main__":
    main()
