"""Compare two learning rates on the same every-epoch grid: forgetting versus collateral change.

Reads the artifacts produced by ``make_epoch_grid.py`` for two runs of the same target, seed and entity
ordering (canonical learning rate 6e-4 and the smaller 6e-5), and measures, per entity, two quantities
between the base-model row and the epoch-10 row:

* ``clip_diff`` - the semantic change already stored in the grid JSON (more negative = concept forgotten).
* mean absolute pixel change - how much the rendered image moved, semantics aside. On the retained
  entities this is collateral change: the initial noise is identical in both rows, so any difference is
  the unlearned UNet following a drifted denoising trajectory.

Both grids must contain an epoch-10 row and the same entity order, which they do by construction
(``make_epoch_grid.py`` derives the order from ``selection_breeds.json``). CPU only, no generation.
Writes learning_rate_collateral.json and learning_rate_collateral.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_SEED = 42
_EPOCH = 10


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    parser = argparse.ArgumentParser(description="Forgetting vs collateral change at two learning rates.")
    parser.add_argument("--suffix-a", default="", help="run suffix of the first grid (canonical run)")
    parser.add_argument("--suffix-b", default="_lr6e-5", help="run suffix of the second grid")
    parser.add_argument("--learning-rate-a", type=float, default=6e-4,
                        help="Learning rate of the first grid. Only used when that grid's JSON predates the "
                             "learning_rate field; when the field is present the two must agree.")
    parser.add_argument("--learning-rate-b", type=float, default=6e-5,
                        help="Learning rate of the second grid, under the same rule as --learning-rate-a.")
    args = parser.parse_args()

    def load_run(suffix: str, declared_learning_rate: float) -> Dict[str, Any]:
        meta = json.loads((_OUT / f"epoch_grid{suffix}_seed{_SEED}.json").read_text(encoding="utf-8"))
        if "learning_rate" in meta:
            assert meta["learning_rate"] == declared_learning_rate, (
                f"grid '{suffix}' was produced at learning rate {meta['learning_rate']}, "
                f"but {declared_learning_rate} was declared on the command line")
        row = meta["rows"].index(f"epoch {_EPOCH}")
        img_dir = _OUT / f"epoch_grid{suffix}"
        # grids written before the script was generalized beyond breeds use the older key name
        ordered = meta.get("entities_by_interference") or meta["breeds_by_interference"]
        names = [b["name"] for b in ordered]
        pixel_change: List[float] = []
        for bi in range(len(names)):
            off = np.asarray(Image.open(img_dir / f"off_s{_SEED}_b{bi}.png").convert("RGB"), dtype=np.int64)
            on = np.asarray(Image.open(img_dir / f"on_ep{_EPOCH}_s{_SEED}_b{bi}.png").convert("RGB"), dtype=np.int64)
            pixel_change.append(float(np.mean(np.abs(on - off))))
        return {
            "learning_rate": declared_learning_rate,
            "names": names,
            "clip_diff": [meta["clip_diff"][f"{row},{bi}"] for bi in range(len(names))],
            "pixel_change": pixel_change,
        }

    run_a = load_run(args.suffix_a, args.learning_rate_a)
    run_b = load_run(args.suffix_b, args.learning_rate_b)
    assert run_a["names"] == run_b["names"], "the two grids use different entity orders"
    names = run_a["names"]
    # entities 0 (target) and 1 (its near-twin) are the ones being unlearned; the rest are retained
    retained = list(range(2, len(names)))

    result: Dict[str, Any] = {
        "seed": _SEED, "epoch": _EPOCH, "entities": names,
        "retained_entity_indices": retained,
        "runs": {f"lr_{r['learning_rate']:g}": r for r in (run_a, run_b)},
        "mean_pixel_change_over_retained": {
            f"lr_{r['learning_rate']:g}": round(float(sum(r["pixel_change"][i] for i in retained) / len(retained)), 2)
            for r in (run_a, run_b)
        },
        "mean_clip_diff_over_retained": {
            f"lr_{r['learning_rate']:g}": round(float(sum(r["clip_diff"][i] for i in retained) / len(retained)), 2)
            for r in (run_a, run_b)
        },
    }
    (_OUT / "learning_rate_collateral.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    labels = [n.replace(" dog", "") for n in names]
    x = np.arange(len(names))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, key, ylabel in (
        (axes[0], "clip_diff", "clip_diff at epoch 10\n(more negative = concept forgotten)"),
        (axes[1], "pixel_change", "mean absolute pixel change\nbase model to epoch 10 (0-255 scale)"),
    ):
        ax.bar(x - width / 2, run_a[key], width, label=f"learning rate {run_a['learning_rate']:g}", color="#c0392b")
        ax.bar(x + width / 2, run_b[key], width, label=f"learning rate {run_b['learning_rate']:g}", color="#2471a3")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axvline(1.5, color="grey", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    fig.suptitle(
        f"SPARE unlearning of 'a bouvier des flandres dog' at two learning rates, seed {_SEED}, epoch {_EPOCH}\n"
        f"left of the dashed line: the target and its near-twin; right: the eight retained breeds",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = _OUT / "learning_rate_collateral.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("COLLATERAL_OK", out_png)
    print(json.dumps({k: result[k] for k in ("mean_clip_diff_over_retained", "mean_pixel_change_over_retained")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
