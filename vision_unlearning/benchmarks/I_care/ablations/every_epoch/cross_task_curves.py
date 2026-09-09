"""The three forget targets on one axis, against optimizer steps rather than epochs.

An epoch is not comparable across the three tasks: the number of optimizer steps in an epoch is set by the
size of the forget set, and the three forget sets differ by a factor of seven (35 breed images, 5 person
images, 20 scene images). Plotting three tasks against the epoch number would compare a breeds epoch worth
nine steps with a people epoch worth two. This figure puts the shared unit - the optimizer step - on the
horizontal axis, so "when does the target break" can be asked across tasks.

Steps per epoch are derived from the forget-set size and the effective batch (batch size 1 with gradient
accumulation 4, the setting the campaign trained with) and are checked against the totals recorded by the
training runs, so a wrong constant cannot pass silently.

CPU only, reads the grid result JSONs. Writes cross_task_curves.png and .json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_EFFECTIVE_BATCH = 4  # per_device_train_batch_size=1 x gradient_accumulation_steps=4
_TASK_STYLE: Dict[str, str] = {"breeds": "#1f77b4", "people": "#d62728", "scenes": "#2ca02c"}
_SEED_STYLE = {0: "-", 1: "--"}
# Total optimizer steps each campaign run reported, used only to check the derivation below.
_RECORDED_TOTAL_STEPS: Dict[str, int] = {"breeds": 270, "people": 400, "scenes": 300}


def steps_per_epoch(forget_images: int, effective_batch: int = _EFFECTIVE_BATCH) -> int:
    """Optimizer steps in one epoch: one per full or partial batch of the forget set.

    SPARE's step count is driven by the forget set, so this is what makes an epoch mean different amounts
    of training in different tasks.
    """
    return int(math.ceil(forget_images / effective_batch))


def steps_axis(epochs: List[int], forget_images: int) -> List[int]:
    """The epoch list converted to cumulative optimizer steps."""
    per_epoch = steps_per_epoch(forget_images)
    return [epoch * per_epoch for epoch in epochs]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vision_unlearning.benchmarks.I_care.result_templates import _display_unlearning_algorithm

    parser = argparse.ArgumentParser(description="Target trajectories of the three tasks against optimizer steps.")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--forget-images", default="breeds=35,people=5,scenes=20",
                        help="forget-set size per task, which sets the number of optimizer steps per epoch")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    forget_images = {part.split("=")[0]: int(part.split("=")[1]) for part in args.forget_images.split(",")}

    fig, ax = plt.subplots(figsize=(11, 7))
    record: Dict[str, Any] = {"seeds": seeds, "effective_batch": _EFFECTIVE_BATCH, "tasks": {}}
    for task, colour in _TASK_STYLE.items():
        grids = {seed: json.loads(
            (_OUT / f"epoch_grid_campaign_{task}_seed{seed}.json").read_text(encoding="utf-8"))
            for seed in seeds}
        reference = grids[seeds[0]]
        epochs: List[int] = reference["epochs"]
        names = [entry["name"] for entry in reference["entities_by_interference"]]
        selection = json.loads((_OUT / f"selection_{task}.json").read_text(encoding="utf-8"))
        target_index = names.index(selection["target"]["name"])
        steps = steps_axis(epochs, forget_images[task])
        total = steps_per_epoch(forget_images[task]) * epochs[-1]
        assert total == _RECORDED_TOTAL_STEPS[task], (
            f"{task}: derived {total} optimizer steps from {forget_images[task]} forget images and "
            f"{epochs[-1]} epochs, but the training run recorded {_RECORDED_TOTAL_STEPS[task]}")

        trajectories: Dict[str, List[float]] = {}
        for position, seed in enumerate(seeds):
            values = [grids[seed]["clip_diff"][f"{row},{target_index}"] for row in range(1, len(epochs) + 1)]
            trajectories[str(seed)] = values
            ax.plot([0] + steps, [0.0] + values, color=colour, linestyle=_SEED_STYLE[position % 2],
                    marker="o", markersize=3, linewidth=1.8,
                    label=f"{task}: {selection['target']['hf_name']}, seed {seed}")
        record["tasks"][task] = {
            "target": selection["target"]["hf_name"], "epochs": epochs,
            "forget_images": forget_images[task], "steps_per_epoch": steps_per_epoch(forget_images[task]),
            "optimizer_steps": steps, "clip_diff_per_seed": trajectories,
        }

    ax.axhline(0.0, color="#999999", linewidth=0.9, zorder=1)
    ax.set_xlabel("optimizer steps (epochs x steps per epoch; the forget set sets the steps per epoch)")
    ax.set_ylabel("clip_diff of the forget target against the original model\n"
                  "(more negative = further from the target's own prompt)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle(
        f"Method: {_display_unlearning_algorithm('distil').upper()} | learning rate=0.0006 | "
        f"three tasks, one forget target each, two seeds\n"
        f"forget-set sizes: "
        + ", ".join(f"{task} {forget_images[task]} images = {steps_per_epoch(forget_images[task])} "
                    f"steps per epoch" for task in _TASK_STYLE),
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = _OUT / "cross_task_curves.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    (_OUT / "cross_task_curves.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"CROSS_TASK_OK {out_png}")
    for task, entry in record["tasks"].items():
        first: List[Tuple[str, Any]] = []
        for seed_label, values in entry["clip_diff_per_seed"].items():
            crossings = [step for step, value in zip(entry["optimizer_steps"], values) if value < -5.0]
            first.append((str(seed_label), crossings[0] if crossings else None))
        print(f"{task}: target below -5 first at optimizer step " +
              ", ".join(f"seed {seed_label}: {step}" for seed_label, step in first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
