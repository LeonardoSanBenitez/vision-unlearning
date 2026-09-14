"""Cross-seed check that each grid's reference row really was generated at its own seed.

A reference row produced with the wrong seed or the wrong prompt ordering is the failure this ablation
already hit once, and it cannot be caught from a single grid: within one run, the original-to-epoch-1 image
change is inflated both by a bad reference row and by the adapter appearing for the first time, and the two
ranges overlap (measured 1.07 to 1.88 on good runs against 2.09 on a deliberately mismatched one).

With two seeds on disk the ambiguity disappears, because a scale becomes available that a single run does
not have: the distance between the SAME entity's baseline images at two different seeds is what "two
unrelated draws of this prompt" looks like. A correct reference row is far closer to its own epoch 1 than
that; a reference row belonging to another seed sits at roughly that distance. In the calibration that
motivated this check the two values were 32.3 and 69.9.

Reports, per task and per seed, ``original -> epoch 1`` over the cross-seed baseline distance, both measured
over the control entities. CPU only, reads existing images. Writes reference_row_check.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_FORGOTTEN_CLIPDIFF = -5.0
# A correct reference row sits well below the two-unrelated-draws scale; a reference row from another seed
# sits at about 1.0. Halfway is a wide margin on both sides given the calibration (0.46 versus 1.00).
_SUSPECT_FRACTION = 0.7


def main() -> int:
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description="Cross-seed verification of every grid's reference row.")
    parser.add_argument("--tasks", nargs="+", default=["breeds", "people", "scenes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()

    def load(path: Path) -> Any:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)

    def mean_abs(first: Path, second: Path) -> float:
        return float(np.mean(np.abs(load(first) - load(second))))

    summary: Dict[str, Any] = {}
    all_passed = True
    for task in args.tasks:
        grid_dir = _OUT / f"epoch_grid_campaign_{task}"
        per_seed: Dict[int, Dict[str, Any]] = {}
        controls_by_seed: Dict[int, List[int]] = {}
        first_epoch: Dict[int, int] = {}
        for seed in args.seeds:
            result = json.loads((_OUT / f"epoch_grid_campaign_{task}_seed{seed}.json").read_text(encoding="utf-8"))
            last_row = len(result["epochs"])
            number_of_entities = len(result["entities_by_interference"])
            controls_by_seed[seed] = [entity for entity in range(number_of_entities)
                                      if result["clip_diff"][f"{last_row},{entity}"] > _FORGOTTEN_CLIPDIFF]
            first_epoch[seed] = result["epochs"][0]

        # Controls common to both seeds, so the same entities are compared on both sides of the ratio.
        controls = sorted(set.intersection(*(set(v) for v in controls_by_seed.values())))
        cross_seed = statistics.mean(
            mean_abs(grid_dir / f"off_s{args.seeds[0]}_b{entity}.png",
                     grid_dir / f"off_s{args.seeds[1]}_b{entity}.png")
            for entity in controls
        )
        for seed in args.seeds:
            reference_change = statistics.mean(
                mean_abs(grid_dir / f"off_s{seed}_b{entity}.png",
                         grid_dir / f"on_ep{first_epoch[seed]}_s{seed}_b{entity}.png")
                for entity in controls
            )
            ratio = reference_change / cross_seed if cross_seed > 0 else 0.0
            per_seed[seed] = {
                "original_to_first_epoch": round(reference_change, 2),
                "over_cross_seed_distance": round(ratio, 3),
                "reference_row_is_suspect": bool(ratio > _SUSPECT_FRACTION),
            }
            all_passed = all_passed and not per_seed[seed]["reference_row_is_suspect"]
            print(f"{task:7s} seed {seed}: original->epoch{first_epoch[seed]} {reference_change:5.1f} vs "
                  f"cross-seed baseline distance {cross_seed:5.1f} -> {ratio:.2f} "
                  f"({'SUSPECT' if per_seed[seed]['reference_row_is_suspect'] else 'ok'})")
        summary[task] = {
            "controls_common_to_both_seeds": controls,
            "cross_seed_baseline_distance": round(cross_seed, 2),
            "per_seed": {str(seed): values for seed, values in per_seed.items()},
        }

    out = _OUT / "reference_row_check.json"
    out.write_text(json.dumps({"suspect_fraction": _SUSPECT_FRACTION, "per_task": summary}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"REFERENCE_ROWS_OK all_passed={all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
