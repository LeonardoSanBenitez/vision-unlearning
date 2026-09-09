"""How much of the every-epoch result reproduces at a second seed.

Each grid is a single image per cell, so any one number carries sampling noise. Running the same adapters
at a second seed separates what is a property of the unlearning from what is a property of the initial
noise: the strongly affected entities should reproduce, and the entities sitting near zero should not be
expected to keep their relative order, because that order is reading noise.

Writes, per task, a markdown table of the last-epoch clip_diff at both seeds and the rank agreement over
the receivers, plus seed_comparison.json. CPU only, reads the grid result JSONs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_NEAR_ZERO = 5.0  # |clip_diff| below this at both seeds counts as "barely moved"


def main() -> int:
    import numpy as np

    from vision_unlearning.benchmarks.I_care.result_templates import _short_entity_display

    parser = argparse.ArgumentParser(description="Cross-seed reproducibility of the every-epoch grids.")
    parser.add_argument("--tasks", nargs="+", default=["breeds", "people", "scenes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()
    first, second = args.seeds

    summary: Dict[str, Any] = {}
    for task in args.tasks:
        selection = json.loads((_OUT / f"selection_{task}.json").read_text(encoding="utf-8"))
        hf_name_of = {selection["target"]["name"]: selection["target"]["hf_name"]}
        for receiver in selection["receivers"]:
            hf_name_of[receiver["name"]] = receiver["hf_name"]
        target_name = selection["target"]["name"]

        results = {}
        for seed in args.seeds:
            results[seed] = json.loads(
                (_OUT / f"epoch_grid_campaign_{task}_seed{seed}.json").read_text(encoding="utf-8"))
        names: List[str] = [entry["name"] for entry in results[first]["entities_by_interference"]]
        assert names == [entry["name"] for entry in results[second]["entities_by_interference"]], (
            f"{task}: the two seeds were generated in different entity orders, so their cells are not "
            "comparable cell by cell")

        def last(seed: int, entity: int) -> float:
            return float(results[seed]["clip_diff"][f"{len(results[seed]['epochs'])},{entity}"])

        receivers = [entity for entity, name in enumerate(names) if name != target_name]
        moved = [entity for entity in receivers
                 if abs(last(first, entity)) >= _NEAR_ZERO or abs(last(second, entity)) >= _NEAR_ZERO]
        still = [entity for entity in receivers if entity not in moved]

        def rank_agreement(entities: List[int]) -> float:
            """Spearman correlation of the last-epoch values across the two seeds, computed directly."""
            if len(entities) < 3:
                return float("nan")
            a = np.argsort(np.argsort([last(first, entity) for entity in entities]))
            b = np.argsort(np.argsort([last(second, entity) for entity in entities]))
            return float(np.corrcoef(a, b)[0, 1])

        lines = [f"| Entity | `clip_diff` at seed {first} | `clip_diff` at seed {second} |",
                 "|---|---:|---:|"]
        for entity in sorted(range(len(names)), key=lambda e: (names[e] != target_name, last(first, e))):
            label = hf_name_of[names[entity]] + (" (target)" if names[entity] == target_name else "")
            emphasis = "**" if names[entity] == target_name else ""
            lines.append(f"| {emphasis}{label}{emphasis} | {last(first, entity):.2f} | "
                         f"{last(second, entity):.2f} |")
        (_OUT / f"seed_comparison_{task}_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary[task] = {
            "target": hf_name_of[target_name],
            "target_clip_diff": {str(first): round(last(first, names.index(target_name)), 2),
                                 str(second): round(last(second, names.index(target_name)), 2)},
            "receivers_that_moved": [_short_entity_display(hf_name_of[names[e]]) for e in moved],
            "rank_agreement_over_all_receivers": round(rank_agreement(receivers), 3),
            "rank_agreement_over_receivers_that_moved": round(rank_agreement(moved), 3),
            "receivers_near_zero_at_both_seeds": len(still),
        }
        record = summary[task]
        print(f"{task:7s} target {record['target_clip_diff'][str(first)]:7.2f} / "
              f"{record['target_clip_diff'][str(second)]:7.2f} | "
              f"{len(moved)} receiver(s) moved, {len(still)} near zero | "
              f"rank agreement all {record['rank_agreement_over_all_receivers']:.2f}, "
              f"moved {record['rank_agreement_over_receivers_that_moved']:.2f}")

    out = _OUT / "seed_comparison.json"
    out.write_text(json.dumps({"seeds": args.seeds, "near_zero_threshold": _NEAR_ZERO,
                               "per_task": summary}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
