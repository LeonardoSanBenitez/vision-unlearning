"""Line chart of clip_diff against training epoch, one line per entity, for one every-epoch run.

The companion of ``make_epoch_grid.py``: the grid shows what the images look like at each epoch, this shows
the same numbers as trajectories, which makes the timing visible - when each entity starts to move, whether
it recovers, and in what order the receivers are affected.

Reads the grid's result JSON (``epoch_grid{suffix}_seed{seed}.json``) and the canonical selection file, so it
runs on the CPU in seconds and never regenerates an image. Every line starts at (0, 0): epoch 0 is the
original model, whose clip_diff against itself is zero by definition.

Colour encodes the receiver's canonical selection group (the similarity-by-interference group it was chosen
for), reusing the palette of ``select_entities.py`` so the two figures agree; the two entities sharing a
group are separated by line style, and the forget target is drawn in black.

Writes epoch_curves{suffix}_seed{seed}.png. Run from this directory with PYTHONPATH at the repo root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_METHOD = "distil"
_LINE_STYLES = ["-", "--"]
_TARGET_STYLE: Dict[str, Any] = {"color": "black", "linewidth": 2.6, "linestyle": "-", "zorder": 5}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vision_unlearning.benchmarks.I_care.result_templates import (
        _display_unlearning_algorithm, _short_entity_display,
    )
    from select_entities import _GROUP_STYLE

    parser = argparse.ArgumentParser(description="clip_diff trajectories for one every-epoch run.")
    parser.add_argument("--task", choices=["breeds", "people", "scenes"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-suffix", type=str, default="",
                        help="the suffix used by make_epoch_grid.py for this run, e.g. _campaign_breeds")
    args = parser.parse_args()

    grid = json.loads((_OUT / f"epoch_grid{args.run_suffix}_seed{args.seed}.json").read_text(encoding="utf-8"))
    selection = json.loads((_OUT / f"selection_{args.task}.json").read_text(encoding="utf-8"))

    target_name = selection["target"]["name"]
    hf_name_of = {target_name: selection["target"]["hf_name"]}
    group_of = {}
    for receiver in selection["receivers"]:
        hf_name_of[receiver["name"]] = receiver["hf_name"]
        group_of[receiver["name"]] = receiver["group"]

    names: List[str] = [entry["name"] for entry in grid["entities_by_interference"]]
    epochs: List[int] = grid["epochs"]
    clip_diff: Dict[str, float] = grid["clip_diff"]
    last_row = len(epochs)

    def trajectory(index: int) -> List[float]:
        return [clip_diff[f"{row},{index}"] for row in range(last_row + 1)]

    # Same order as the grid's columns: the target first, then the receivers by their last-epoch value, so
    # the legend of one figure reads like the columns of the other.
    ordered: List[int] = grid["display_column_order"]

    style_index: Dict[str, int] = {}
    fig, ax = plt.subplots(figsize=(12, 8.5))
    for index in ordered:
        name = names[index]
        label = _short_entity_display(hf_name_of[name], max_chars=34)
        if name == target_name:
            style: Dict[str, Any] = dict(_TARGET_STYLE)
            label = f"{label} (target)"
        else:
            group = group_of[name]
            seen = style_index.get(group, 0)
            style_index[group] = seen + 1
            style = {"color": _GROUP_STYLE[group]["color"], "linewidth": 1.6,
                     "linestyle": _LINE_STYLES[seen % len(_LINE_STYLES)], "zorder": 3}
            label = f"{label} — {group}"
        ax.plot([0] + epochs, trajectory(index), marker="o", markersize=3, label=label, **style)

    ax.axhline(0.0, color="#999999", linewidth=0.9, zorder=1)
    ax.set_xlabel("training epoch (0 = the original model, before any unlearning)")
    ax.set_ylabel("clip_diff against the original model\n(more negative = further from the entity's own prompt)")
    ax.grid(alpha=0.25)
    # Below the axes, never inside: the target and the strongly interfered receivers occupy the lower part
    # of the plot, which is exactly where an inside legend lands.
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    target_hf, overwrite = selection["target"]["hf_name"], selection["target"]["overwrite_concept"]
    fig.suptitle(
        f"Method: {_display_unlearning_algorithm(_METHOD).upper()} | "
        f"Overwrite '{target_hf}' to '{overwrite}' | "
        f"seed={args.seed}, learning rate={grid['learning_rate']:g}\n"
        f"one line per entity; markers mark the epochs at which an adapter was saved; "
        f"colour is the canonical selection group the receiver was chosen for",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = _OUT / f"epoch_curves{args.run_suffix}_seed{args.seed}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    # The companion table for the write-up, generated from the same JSON as the figure so the two cannot
    # disagree and no number is retyped by hand.
    canonical_of = {entry["name"]: entry["canonical_clip_diff"] for entry in grid["entities_by_interference"]}
    lines = [
        "| Entity | Canonical interference | `clip_diff` at epoch "
        f"{epochs[-1]} | First epoch below −5 |",
        "|---|---:|---:|---|",
    ]
    for index in ordered:
        name = names[index]
        values = trajectory(index)
        below = [epoch for epoch, value in zip(epochs, values[1:]) if value < -5.0]
        label = hf_name_of[name] + (" (target)" if name == target_name else "")
        emphasis = "**" if name == target_name else ""
        lines.append(f"| {emphasis}{label}{emphasis} | {canonical_of[name]:.2f} | {values[-1]:.2f} | "
                     f"{'epoch ' + str(below[0]) if below else 'never'} |")
    out_table = _OUT / f"epoch_curves{args.run_suffix}_seed{args.seed}_table.md"
    out_table.write_text("\n".join(lines) + "\n", encoding="utf-8")

    extremes: List[Tuple[str, float]] = sorted(
        ((names[i], clip_diff[f"{last_row},{i}"]) for i in ordered), key=lambda pair: pair[1])
    print(f"CURVES_OK {out_png}")
    print(f"TABLE_OK {out_table}")
    print("last epoch, most to least interfered: "
          + ", ".join(f"{name} {value:.2f}" for name, value in extremes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
