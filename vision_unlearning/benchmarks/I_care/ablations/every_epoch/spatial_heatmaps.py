"""Where in the image the unlearning shows up: per-epoch difference maps and how localised they are.

``make_epoch_grid.py`` answers what each entity looks like at each epoch and ``make_epoch_curves.py``
answers when it changes. Neither answers *where*: a target that turns into the overwrite concept and a
retained entity whose whole frame drifts can carry the same ``clip_diff``. This script takes the images
those runs already wrote and, for every (entity, epoch), computes the absolute difference against that
entity's own base-model image, averaged over the colour channels and over the seeds.

Two things come out of each difference map:

* the mean absolute change - how much moved, in pixel units;
* the fraction of the total change carried by the most-changed tenth of the pixels. A change spread
  uniformly over the frame gives 0.1, and a change confined to a small region approaches 1, so the number
  says how localised the change is on a scale that does not depend on its magnitude. It needs no
  segmentation, which matters because this repository has none: a claim about "the face" or "the
  background" would require a detector that does not exist here.

CPU only; every input image is already on disk. Writes spatial_heatmaps{suffix}.png and .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_TOP_FRACTION = 0.1  # the "most-changed tenth of the pixels" the concentration statistic is defined on


def difference_map(on_image: Any, off_image: Any) -> Any:
    """Absolute per-pixel difference between an epoch image and its own baseline, averaged over channels.

    Both arguments are height x width x channel arrays of the same shape. Keeping this a function of the
    two arrays (rather than of an entity index) is what makes the on/off pairing testable: swapping an
    entity for its neighbour is invisible in the rendered figure but changes this output.
    """
    import numpy as np
    on_array = np.asarray(on_image, dtype=np.float64)
    off_array = np.asarray(off_image, dtype=np.float64)
    assert on_array.shape == off_array.shape, f"shape mismatch: {on_array.shape} against {off_array.shape}"
    return np.abs(on_array - off_array).mean(axis=2)


def concentration(change_map: Any, top_fraction: float = _TOP_FRACTION) -> float:
    """Fraction of the total absolute change that falls in the most-changed ``top_fraction`` of pixels.

    Equal to ``top_fraction`` when the change is uniform and approaches 1 when it is confined to a small
    region, so it measures localisation independently of magnitude. A map that is exactly zero everywhere
    has no localisation to report and returns ``top_fraction``, the uniform value.
    """
    import numpy as np
    flat = np.sort(np.asarray(change_map, dtype=np.float64).ravel())[::-1]
    total = float(flat.sum())
    if total <= 0.0:
        return float(top_fraction)
    count = max(1, int(round(len(flat) * top_fraction)))
    return float(flat[:count].sum() / total)


def statistics_of(statistics: Dict[str, Dict[str, float]], epoch: int, entity: int) -> Dict[str, float]:
    """The record for one cell. The key is built in one place so a swapped epoch and entity cannot happen."""
    return statistics[f"{epoch},{entity}"]


def column_order_by_change(last_epoch_change: Dict[int, float], names: List[str], target_index: int) -> List[int]:
    """Figure columns: the target first, then the receivers by decreasing change in the last epoch.

    The figure is ordered by its own quantity rather than by ``clip_diff``, so that what the reader sees
    (the size of the difference maps) is the sort key. Ties break on the entity name.
    """
    receivers = sorted((index for index in last_epoch_change if index != target_index),
                       key=lambda index: (-last_epoch_change[index], names[index]))
    return [target_index] + receivers


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from vision_unlearning.benchmarks.I_care.result_templates import (
        _display_unlearning_algorithm, _short_entity_display,
    )

    parser = argparse.ArgumentParser(description="Per-epoch spatial difference maps for one every-epoch run.")
    parser.add_argument("--task", choices=["breeds", "people", "scenes"], required=True)
    parser.add_argument("--seeds", default="42,43", help="comma-separated seeds to average over")
    parser.add_argument("--run-suffix", type=str, default="",
                        help="the suffix make_epoch_grid.py used for this run, e.g. _campaign_breeds")
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
    for receiver in selection["receivers"]:
        hf_name_of[receiver["name"]] = receiver["hf_name"]
    target_index = names.index(target_name)

    image_dir = _OUT / f"epoch_grid{suffix}"

    def load(path: Path) -> Any:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)

    maps: Dict[Tuple[int, int], Any] = {}
    statistics: Dict[str, Dict[str, float]] = {}
    for entity in range(len(names)):
        baselines = {seed: load(image_dir / f"off_s{seed}_b{entity}.png") for seed in seeds}
        for row, epoch in enumerate(epochs):
            per_seed = [difference_map(load(image_dir / f"on_ep{epoch}_s{seed}_b{entity}.png"), baselines[seed])
                        for seed in seeds]
            averaged = np.mean(per_seed, axis=0)
            maps[(row, entity)] = averaged
            statistics[f"{epoch},{entity}"] = {
                "mean_absolute_change": float(averaged.mean()),
                "concentration_top_decile": concentration(averaged),
            }

    last_row = len(epochs) - 1
    display_order = column_order_by_change(
        last_epoch_change={entity: statistics[f"{epochs[last_row]},{entity}"]["mean_absolute_change"]
                           for entity in range(len(names))},
        names=names, target_index=target_index,
    )

    # One colour scale for the whole figure, so a cell that looks brighter really did change more. The
    # maximum is taken over all cells rather than per cell, which would make every row look alike.
    vmax = max(float(m.max()) for m in maps.values())

    number_of_rows, number_of_columns = len(epochs), len(display_order)
    fig, axes = plt.subplots(number_of_rows, number_of_columns, squeeze=False,
                             figsize=(1.7 * number_of_columns, 1.9 * number_of_rows))
    for row in range(number_of_rows):
        for column, entity in enumerate(display_order):
            ax = axes[row][column]
            ax.imshow(maps[(row, entity)], cmap="inferno", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            stats = statistics[f"{epochs[row]},{entity}"]
            ax.set_title(f"change={stats['mean_absolute_change']:.1f}, "
                         f"top tenth={stats['concentration_top_decile']:.2f}", fontsize=5)
            if column == 0:
                ax.set_ylabel(f"epoch {epochs[row]}", fontsize=8, rotation=0, ha="right", va="center")
    for column, entity in enumerate(display_order):
        label = _short_entity_display(hf_name_of[names[entity]])
        axes[0][column].set_title(
            (f"{label} (target)" if column == 0 else label)
            + f"\nchange={statistics[f'{epochs[0]},{entity}']['mean_absolute_change']:.1f}, "
              f"top tenth={statistics[f'{epochs[0]},{entity}']['concentration_top_decile']:.2f}",
            fontsize=5)
    target_hf, overwrite = selection["target"]["hf_name"], selection["target"]["overwrite_concept"]
    fig.suptitle(
        f"Method: {_display_unlearning_algorithm('distil').upper()} | "
        f"Overwrite '{target_hf}' to '{overwrite}' | "
        f"seeds={','.join(str(s) for s in seeds)}, learning rate={reference['learning_rate']:g}\n"
        f"absolute difference from each entity's own base-model image, averaged over the colour channels "
        f"and over the seeds; one colour scale for the whole figure (0 to {vmax:.0f})\n"
        f"columns = the target then the receivers by decreasing change in the last epoch; "
        f"'top tenth' = the share of the change carried by the most-changed tenth of the pixels "
        f"(0.1 = uniform)",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = _OUT / f"spatial_heatmaps{suffix}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # How localisation relates to magnitude, over the ten entities at the last epoch. A negative
    # correlation means the entities that changed most are the ones whose change is spread over the whole
    # frame, which is the opposite of what "more localised unlearning" would look like.
    last_change = [statistics_of(statistics, epochs[last_row], entity)["mean_absolute_change"]
                   for entity in range(len(names))]
    last_concentration = [statistics_of(statistics, epochs[last_row], entity)["concentration_top_decile"]
                          for entity in range(len(names))]
    correlation = float(np.corrcoef(last_change, last_concentration)[0, 1])

    result: Dict[str, Any] = {
        "task": args.task, "seeds": seeds, "epochs": epochs, "run_suffix": suffix,
        "correlation_change_against_concentration_last_epoch": correlation,
        "learning_rate": reference["learning_rate"],
        "colour_scale_maximum": vmax,
        "entities_in_generation_order": names,
        "display_column_order": display_order,
        "top_fraction": _TOP_FRACTION,
        "per_epoch_per_entity": statistics,
    }
    out_json = _OUT / f"spatial_heatmaps{suffix}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    target_last = statistics_of(statistics, epochs[last_row], target_index)
    controls = [entity for entity in range(len(names)) if entity != target_index]
    print(f"HEATMAPS_OK {out_png}")
    print(f"target last epoch: change={target_last['mean_absolute_change']:.1f}, "
          f"concentration={target_last['concentration_top_decile']:.3f}")
    print("receivers last epoch, concentration: " + ", ".join(
        f"{names[entity]} {statistics_of(statistics, epochs[last_row], entity)['concentration_top_decile']:.3f}"
        for entity in controls))
    print(f"correlation between change and concentration over the ten entities: {correlation:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
