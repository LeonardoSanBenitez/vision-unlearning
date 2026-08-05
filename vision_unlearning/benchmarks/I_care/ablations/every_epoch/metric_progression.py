"""A second and third metric alongside clip_diff, over the same epochs, and the noise floor of clip_diff.

``clip_diff`` measures agreement with a text prompt, so it moves when the concept in the image is replaced
and stays put while an image merely degrades. This script measures the same images with the library's own
``MetricImageImage(metrics=['rmse', 'ssim'])`` - the primitive ``pipeline_06`` uses for the benchmark's
image metrics - against each entity's own base-model image, so the three can be read on one epoch axis and
the reader can see which of them registers what.

It also measures the noise floor of ``clip_diff``: the same prompt drawn twice from the *base* model (once
per seed) gives two different images and therefore two different CLIP scores, and the size of that
difference is the smallest ``clip_diff`` that can mean anything. Two seeds give one difference per entity,
which is an indication of scale and not an error bar, and it is labelled that way wherever it is used.

CPU only; all images are already on disk. Writes metric_progression{suffix}.json and .png, and
noise_floor_{task}.json (read by make_epoch_curves.py to draw the band).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_METHOD = "distil"


def seed_average(per_seed: Dict[int, List[float]]) -> List[float]:
    """Mean over seeds, position by position. The lists are trajectories over the same epochs."""
    seeds = sorted(per_seed)
    lengths = {len(per_seed[seed]) for seed in seeds}
    assert len(lengths) == 1, f"trajectories of different lengths cannot be averaged: {lengths}"
    return [statistics.mean(per_seed[seed][index] for seed in seeds) for index in range(lengths.pop())]


def noise_floor_summary(per_entity: Dict[str, float]) -> Dict[str, float]:
    """Median and maximum of the base-model score differences between the two seeds.

    These are drawn as a band on the trajectory figures: a movement smaller than this is within what two
    draws of the same prompt from the *unmodified* model already differ by.
    """
    values = sorted(per_entity.values())
    return {"median": statistics.median(values), "maximum": max(values), "minimum": min(values),
            "number_of_entities": len(values)}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from vision_unlearning.metrics import MetricImageImage, MetricImageTextSimilarity
    from vision_unlearning.benchmarks.I_care.result_templates import (
        _display_unlearning_algorithm, _short_entity_display,
    )
    from select_entities import _GROUP_STYLE

    parser = argparse.ArgumentParser(description="rmse and ssim alongside clip_diff, over the epoch axis.")
    parser.add_argument("--task", choices=["breeds", "people", "scenes"], required=True)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--run-suffix", type=str, default="")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    suffix = args.run_suffix

    grids = {seed: json.loads((_OUT / f"epoch_grid{suffix}_seed{seed}.json").read_text(encoding="utf-8"))
             for seed in seeds}
    reference = grids[seeds[0]]
    epochs: List[int] = reference["epochs"]
    for seed, grid in grids.items():
        assert grid["epochs"] == epochs, f"seed {seed} was rendered from a different set of epochs"
    names: List[str] = [entry["name"] for entry in reference["entities_by_interference"]]

    selection = json.loads((_OUT / f"selection_{args.task}.json").read_text(encoding="utf-8"))
    target_name = selection["target"]["name"]
    hf_name_of = {target_name: selection["target"]["hf_name"]}
    group_of: Dict[str, str] = {}
    for receiver in selection["receivers"]:
        hf_name_of[receiver["name"]] = receiver["hf_name"]
        group_of[receiver["name"]] = receiver["group"]
    prompt_of = {name: f"An image of {hf_name_of[name]}" for name in names}

    image_dir = _OUT / f"epoch_grid{suffix}"
    image_metric = MetricImageImage(metrics=["rmse", "ssim"])  # type: ignore[call-arg]
    text_metric = MetricImageTextSimilarity(metrics=["clip"])

    # --- rmse and ssim against each entity's own baseline, per seed --------------------------------- #
    per_seed_rmse: Dict[int, Dict[int, List[float]]] = {seed: {} for seed in seeds}
    per_seed_ssim: Dict[int, Dict[int, List[float]]] = {seed: {} for seed in seeds}
    for seed in seeds:
        for entity in range(len(names)):
            baseline = image_dir / f"off_s{seed}_b{entity}.png"
            scores = [image_metric.score(str(baseline), str(image_dir / f"on_ep{epoch}_s{seed}_b{entity}.png"))
                      for epoch in epochs]
            per_seed_rmse[seed][entity] = [float(s["rmse"]) for s in scores]
            per_seed_ssim[seed][entity] = [float(s["ssim"]) for s in scores]

    # --- noise floor: the same prompt drawn once per seed from the BASE model ------------------------ #
    noise_floor_per_entity: Dict[str, float] = {}
    base_scores: Dict[str, Dict[int, float]] = {}
    for entity, name in enumerate(names):
        by_seed = {}
        for seed in seeds:
            image = Image.open(image_dir / f"off_s{seed}_b{entity}.png").convert("RGB")
            by_seed[seed] = float(text_metric.score_batch_same_text([image], prompt_of[name])[0]["clip"])
        base_scores[name] = by_seed
        noise_floor_per_entity[name] = abs(by_seed[seeds[0]] - by_seed[seeds[-1]])
    floor = noise_floor_summary(noise_floor_per_entity)

    # --- figure: clip_diff, rmse and ssim on a shared epoch axis, averaged over the seeds ------------ #
    def clip_trajectory(seed: int, entity: int) -> List[float]:
        return [grids[seed]["clip_diff"][f"{row},{entity}"] for row in range(1, len(epochs) + 1)]

    display_order: List[int] = reference["display_column_order"]
    line_styles = ["-", "--"]
    style_index: Dict[str, int] = {}
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for entity in display_order:
        name = names[entity]
        label = _short_entity_display(hf_name_of[name], max_chars=34)
        if name == target_name:
            style: Dict[str, Any] = {"color": "black", "linewidth": 2.6, "linestyle": "-", "zorder": 5}
            label = f"{label} (target)"
        else:
            # The two receivers of one selection group share a colour, so they are separated by line
            # style; without it the legend would carry two entries with the same visual encoding.
            group = group_of[name]
            seen = style_index.get(group, 0)
            style_index[group] = seen + 1
            style = {"color": _GROUP_STYLE[group]["color"], "linewidth": 1.5,
                     "linestyle": line_styles[seen % len(line_styles)], "zorder": 3}
            label = f"{label} — {group}"
        axes[0].plot(epochs, seed_average({s: clip_trajectory(s, entity) for s in seeds}),
                     marker="o", markersize=3, label=label, **style)
        axes[1].plot(epochs, seed_average({s: per_seed_rmse[s][entity] for s in seeds}),
                     marker="o", markersize=3, **style)
        axes[2].plot(epochs, seed_average({s: per_seed_ssim[s][entity] for s in seeds}),
                     marker="o", markersize=3, **style)

    axes[0].axhspan(-floor["median"], floor["median"], color="#cccccc", alpha=0.55, zorder=0)
    axes[0].set_ylabel("clip_diff against the original model\n(more negative = further from the prompt)")
    axes[1].set_ylabel("root mean squared error against\nthe entity's own base-model image")
    axes[2].set_ylabel("structural similarity against\nthe entity's own base-model image")
    axes[2].set_xlabel("training epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
    # Below the axes: the target and the strongly interfered receivers occupy the lower part of the first
    # panel, which is where an inside legend lands.
    axes[2].legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), frameon=False)
    target_hf, overwrite = selection["target"]["hf_name"], selection["target"]["overwrite_concept"]
    fig.suptitle(
        f"Method: {_display_unlearning_algorithm(_METHOD).upper()} | "
        f"Overwrite '{target_hf}' to '{overwrite}' | "
        f"seeds={','.join(str(s) for s in seeds)}, learning rate={reference['learning_rate']:g}\n"
        f"the same images under three metrics, averaged over the seeds; the grey band on the first panel "
        f"is the median difference between the two seeds' base-model scores",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = _OUT / f"metric_progression{suffix}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    result: Dict[str, Any] = {
        "task": args.task, "seeds": seeds, "epochs": epochs, "run_suffix": suffix,
        "entities_in_generation_order": names,
        "rmse": {str(seed): {names[e]: per_seed_rmse[seed][e] for e in range(len(names))} for seed in seeds},
        "ssim": {str(seed): {names[e]: per_seed_ssim[seed][e] for e in range(len(names))} for seed in seeds},
        "base_model_clip_score_per_seed": base_scores,
        "clip_noise_floor_per_entity": noise_floor_per_entity,
        "clip_noise_floor_summary": floor,
    }
    (_OUT / f"metric_progression{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (_OUT / f"noise_floor_{args.task}.json").write_text(
        json.dumps({"task": args.task, "seeds": seeds, "per_entity": noise_floor_per_entity,
                    "summary": floor}, indent=2), encoding="utf-8")

    print(f"METRICS_OK {out_png}")
    print(f"clip noise floor over {floor['number_of_entities']} entities: median {floor['median']:.2f}, "
          f"range {floor['minimum']:.2f} to {floor['maximum']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
