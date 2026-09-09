'''S10 of PLAN-TASK-2026-08-12-SDXL: the epoch-by-entity image grid for a Stable Diffusion XL campaign.

Rows are the off-baseline followed by every saved checkpoint; columns are the entities, the forget
target pinned to column 0 and the rest ordered by their `clip_diff` at the last checkpoint, most
negative nearest the target. That is `make_epoch_grid.py`'s layout in the every-epoch ablation, kept
deliberately so a reader who knows the Stable Diffusion 1.4 figures reads this one without relearning
it. What it is NOT is a parametrization of that script: this one generates nothing and scores nothing.
It reads images a campaign already wrote and numbers a scorer already computed, which is the whole
reason it is safe to run on a laptop in seconds.

It works on any artifact pair written in this ablation's two shapes -- a manifest of
`{epoch, entity, path, seed}` rows and a scores file with a `per_seed` envelope -- so the campaign and
the random-ten control are drawn by the same code:

    python make_epoch_grid_sdxl.py --manifest assets/campaign_seed42.json \\
        --scores assets/clip_diff_campaign.json --seed 42 \\
        --output assets/epoch_grid_sdxl_seed42.png

    python make_epoch_grid_sdxl.py --manifest assets/random_ten_control_seed42.json \\
        --scores assets/clip_diff_random_ten_control.json --seed 42 \\
        --output assets/random_ten_control_sheet_seed42.png

Beside the figure it writes a JSON holding every cell's path and value, so a claim about the figure
can be checked without reading pixels.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import campaign_configuration as cfg

_HERE = Path(__file__).resolve().parent


def column_order(final_clip_diff: Dict[str, float], target: Optional[str]) -> List[str]:
    '''Entity display order: the target first if there is one, then ascending `clip_diff`.

    Ascending puts the most negative -- the most interfered-with -- next to the target, which is what
    makes a staircase of receivers legible as a staircase. The order is a presentation choice and has
    nothing to do with the generation order, which was fixed before any image existed.
    '''
    others = sorted((name for name in final_clip_diff if name != target),
                    key=lambda name: final_clip_diff[name])
    return ([target] if target is not None else []) + others


def _cells(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, str], str]:
    '''(epoch, entity) -> image path, from the manifest rows. `None` is the off-baseline epoch.'''
    return {(row["epoch"], row["entity"]): row["path"] for row in rows}


def render_grid(
    cell_path: Dict[Tuple[Any, str], str],
    clip_diff: Dict[Tuple[Any, str], float],
    epochs: List[int],
    display_order: List[str],
    target: Optional[str],
    title: str,
    output_path: Path,
) -> None:
    '''Draws the grid: figure row `r` is `[None] + epochs` at `r`, figure column `c` is
    `display_order[c]`.

    Both dictionaries are keyed by (epoch, entity) rather than by (row, column), which is the one
    thing that stops a re-ordered or transposed axis from producing a full, plausible, wrong figure:
    the label, the image and the number for a cell are all fetched with the same key.
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    labels: List[Any] = [None] + list(epochs)
    number_of_rows, number_of_columns = len(labels), len(display_order)
    figure, axes = plt.subplots(number_of_rows, number_of_columns, squeeze=False,
                                figsize=(1.7 * number_of_columns, 1.9 * number_of_rows))
    for row, epoch in enumerate(labels):
        for column, entity in enumerate(display_order):
            axis = axes[row][column]
            axis.imshow(np.asarray(Image.open(cell_path[(epoch, entity)]).convert("RGB")))
            axis.set_xticks([])
            axis.set_yticks([])
            if epoch is not None:
                axis.set_title(f"clip_diff={clip_diff[(epoch, entity)]:.2f}", fontsize=6)
            if column == 0:
                axis.set_ylabel("original" if epoch is None else f"epoch {epoch}",
                                fontsize=8, rotation=0, ha="right", va="center")
    for column, entity in enumerate(display_order):
        shown = entity.replace("_", " ")
        axes[0][column].set_title(f"{shown} (target)" if entity == target else shown, fontsize=6)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epoch-by-entity image grid for a Stable Diffusion XL campaign.")
    parser.add_argument("--manifest", required=True, help="generation manifest, relative to this directory")
    parser.add_argument("--scores", required=True, help="scores file with a per_seed envelope")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, help="output .png, relative to this directory")
    args = parser.parse_args()

    manifest_path = _HERE / args.manifest
    scores_path = _HERE / args.scores
    output_path = _HERE / args.output

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    seed_block = scores["per_seed"][str(args.seed)]
    epochs: List[int] = seed_block["epochs"]
    per_entity: Dict[str, Any] = seed_block["per_entity"]
    target: Optional[str] = scores.get("target")

    clip_diff: Dict[Tuple[Any, str], float] = {}
    for entity, payload in per_entity.items():
        for point in payload["trajectory"]:
            clip_diff[(point["epoch"], entity)] = point["clip_diff"]

    final_clip_diff = {entity: clip_diff[(epochs[-1], entity)] for entity in per_entity}
    display_order = column_order(final_clip_diff, target)

    cell_path = _cells([row for row in rows if row["seed"] == args.seed])
    missing = [(epoch, entity) for entity in display_order for epoch in [None] + epochs
               if (epoch, entity) not in cell_path]
    assert not missing, f"manifest has no image for: {missing}"
    absent = [path for path in cell_path.values() if not Path(path).is_file()]
    assert not absent, f"manifest points at files that do not exist: {absent[:5]}"

    floor = cfg.noise_floor_standard_deviation()
    title = (f"model=stable-diffusion-xl-base-1.0 | task={cfg.TASK} | method=spare | seed={args.seed} | "
             f"resolution={cfg.GENERATION_RESOLUTION} | guidance={cfg.GENERATION_KWARGS['guidance_scale']}\n"
             f"rows=off-baseline+{len(epochs)} checkpoints | columns={len(display_order)} entities | "
             f"cell=clip_diff against the same entity's off-baseline | noise_floor_1sd={floor:.3f}")
    render_grid(cell_path, clip_diff, epochs, display_order, target, title, output_path)

    record = {
        "figure": str(output_path),
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "seed": args.seed,
        "epochs": epochs,
        "display_order": display_order,
        "target": target,
        "noise_floor_standard_deviation": floor,
        "cells": [
            {"epoch": epoch, "entity": entity, "path": cell_path[(epoch, entity)],
             "clip_diff": None if epoch is None else clip_diff[(epoch, entity)]}
            for epoch in [None] + epochs for entity in display_order
        ],
    }
    record_path = output_path.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"rows: {len(epochs) + 1} = 1 off-baseline + {len(epochs)} checkpoints")
    print(f"columns: {len(display_order)} entities, target first: {display_order}")
    print(f"cells drawn: {len(record['cells'])}; expected {len(epochs) + 1} x {len(display_order)} = "
          f"{(len(epochs) + 1) * len(display_order)}")
    print(f"cells drawn equals expected: {len(record['cells']) == (len(epochs) + 1) * len(display_order)}")
    print(f"written: {output_path}")
    print(f"written: {record_path}")
    print("EPOCH_GRID_SDXL_DONE")


if __name__ == "__main__":
    main()
