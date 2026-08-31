'''S10 of PLAN-TASK-2026-08-12-SDXL: the same entity, the same epochs, the two base models side by side.

This is the figure the task exists for. The user's question was not "does the number go down on
Stable Diffusion XL" -- it was whether the I-CARE reading of *subtle, subjective* differences survives
a change of base model. So this puts one entity's Stable Diffusion 1.4 row directly above its Stable
Diffusion XL row, epoch by epoch, and lets a human read the two.

WHAT MAY AND MAY NOT BE COMPARED HERE, because the figure invites the invalid comparison by existing:

* comparable: the SHAPE of a trajectory (when the target leaves its baseline, whether a receiver is
  transiently affected and recovers, in what order receivers move) and the KIND of change (replaced by
  the overwrite concept, degraded, desaturated, untouched);
* NOT comparable: the `clip_diff` values themselves across the two models. They are different models
  at different resolutions with different autoencoders, and each has its own measured noise floor. The
  figure prints both floors and both resolutions in its own title so a reader cannot pick up one number
  and set it beside the other without seeing why they do not meet.

Stable Diffusion 1.4 images come from the every-epoch ablation's campaign folder, addressed through
its own manifest (an entity's file is named by its POSITION in that run's generation order, so the
manifest is the only safe way to find it). Stable Diffusion XL images come from this ablation's
manifest. Neither is regenerated.

    python subtle_difference_reading.py --entities Mark_Philippoussis,Tim_Henman,Juan_Carlos_Ferrero \\
        --seed 42 --output assets/subtle_difference_reading_seed42.png
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import campaign_configuration as cfg
import sd14_campaign

_HERE = Path(__file__).resolve().parent


def sdxl_cells(seed: int, entity: str) -> Tuple[Dict[Any, Path], Dict[Any, float], List[int]]:
    '''The same three things for the Stable Diffusion XL campaign, from this ablation's artifacts.'''
    rows = json.loads((cfg.OUT / f"campaign_seed{seed}.json").read_text(encoding="utf-8"))
    scores = json.loads((cfg.OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    seed_block = scores["per_seed"][str(seed)]
    epochs: List[int] = seed_block["epochs"]
    paths: Dict[Any, Path] = {row["epoch"]: Path(row["path"]) for row in rows
                              if row["entity"] == entity and row["seed"] == seed}
    values: Dict[Any, float] = {point["epoch"]: float(point["clip_diff"])
                                for point in seed_block["per_entity"][entity]["trajectory"]}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    assert not missing, f"Stable Diffusion XL images missing: {missing[:5]}"
    return paths, values, epochs


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description="One entity, two base models, epoch by epoch.")
    parser.add_argument("--entities", required=True, help="comma-separated entity names")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    entities = [name.strip() for name in args.entities.split(",") if name.strip()]

    per_entity: Dict[str, Any] = {}
    for entity in entities:
        sd14_paths, sd14_values, sd14_epochs = sd14_campaign.entity_cells(args.seed, entity)
        sdxl_paths, sdxl_values, sdxl_epochs = sdxl_cells(args.seed, entity)
        assert sd14_epochs == sdxl_epochs, (
            f"the two campaigns saved different checkpoints ({sd14_epochs} against {sdxl_epochs}); "
            "the rows of this figure would not line up"
        )
        per_entity[entity] = {
            "sd14": (sd14_paths, sd14_values), "sdxl": (sdxl_paths, sdxl_values), "epochs": sd14_epochs,
        }

    epochs: List[int] = per_entity[entities[0]]["epochs"]
    columns: List[Any] = [None] + epochs
    rows = 2 * len(entities)
    figure, axes = plt.subplots(rows, len(columns), squeeze=False,
                                figsize=(1.45 * len(columns), 1.75 * rows))
    for entity_index, entity in enumerate(entities):
        for model_index, model in enumerate(["sd14", "sdxl"]):
            paths, values = per_entity[entity][model]
            row = 2 * entity_index + model_index
            for column, epoch in enumerate(columns):
                axis = axes[row][column]
                axis.imshow(np.asarray(Image.open(paths[epoch]).convert("RGB")))
                axis.set_xticks([])
                axis.set_yticks([])
                if epoch is not None:
                    axis.set_title(f"{values[epoch]:+.1f}", fontsize=6)
                if row == 0:
                    label = "off-baseline" if epoch is None else f"epoch {epoch}"
                    axis.set_xlabel(label, fontsize=7)
                    axis.xaxis.set_label_position("top")
                if column == 0:
                    shown = "stable diffusion 1.4" if model == "sd14" else "stable diffusion xl"
                    axis.set_ylabel(f"{entity.replace('_', ' ')}\n{shown}", fontsize=7,
                                    rotation=0, ha="right", va="center")

    sd14_floor = sd14_campaign.noise_floor_summary()
    sdxl_floor = cfg.noise_floor_standard_deviation()
    figure.suptitle(
        f"task={cfg.TASK} | method=spare | seed={args.seed} | entities={len(entities)} | "
        f"checkpoints={len(epochs)} | cell number = that model's own clip_diff\n"
        f"stable diffusion 1.4: resolution={sd14_campaign.RESOLUTION}, noise floor over ten entities and two "
        f"seeds median={sd14_floor['median']:.2f} maximum={sd14_floor['maximum']:.2f} | "
        f"stable diffusion xl: resolution={cfg.GENERATION_RESOLUTION}, noise floor over one entity and "
        f"six seeds, one standard deviation={sdxl_floor:.2f}\n"
        f"the two clip_diff scales are NOT comparable: different model, autoencoder, resolution and "
        f"floor. What is comparable is the shape of the trajectory and the kind of change.",
        fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = _HERE / args.output
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    record = {
        "figure": str(output_path),
        "seed": args.seed,
        "entities": entities,
        "epochs": epochs,
        "stable_diffusion_1_4": {"model": sd14_campaign.MODEL_ID, "resolution": sd14_campaign.RESOLUTION,
                                 "noise_floor_summary": sd14_floor,
                                 "clip_diff": {entity: per_entity[entity]["sd14"][1] for entity in entities}},
        "stable_diffusion_xl": {"model": cfg.MODEL_ID, "resolution": cfg.GENERATION_RESOLUTION,
                                "noise_floor_standard_deviation": sdxl_floor,
                                "clip_diff": {entity: per_entity[entity]["sdxl"][1] for entity in entities}},
    }
    record_path = output_path.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print(f"entities: {len(entities)} {entities}")
    print(f"rows: {rows} = 2 models x {len(entities)} entities; columns: {len(columns)} = "
          f"1 off-baseline + {len(epochs)} checkpoints")
    print(f"cells drawn: {rows * len(columns)}")
    print(f"written: {output_path}")
    print(f"written: {record_path}")
    print("SUBTLE_DIFFERENCE_READING_DONE")


if __name__ == "__main__":
    main()
