"""How often does SPARE damage some retained entity more than the forget target itself?

The every-epoch grids show this happening in two of three case studies, but three targets cannot support a
statement about the method. The canonical per-entity artifacts already answer it for every entity of every
task, at no compute cost: ``number_of_interfered_worse_than_target_clip_diff`` counts, for a given forget
target, how many of the 99 retained entities ended with a clip_diff more negative than the target's own
(``metrics.py::number_of_interfered_worse_than_target``, verified rather than assumed).

Reads ``interference_per_entity_{task}.json`` from the canonical I-CARE assets. CPU only, read-only.
Writes canonical_worse_than_target.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_ICARE_ASSETS = _THIS.parents[2] / "assets"
_OUT = _THIS.parent / "assets"
_CANONICAL_EPOCHS = {"people": 400, "scenes": 100, "breeds": 100}
_METRIC = "emitter_number_of_interfered_worse_than_target_clip_diff (↓)"


def main() -> int:
    import numpy as np

    summary: Dict[str, Any] = {}
    for task, epochs in _CANONICAL_EPOCHS.items():
        entities = json.loads((_ICARE_ASSETS / f"interference_per_entity_{task}.json").read_text(encoding="utf-8"))
        key = f"metric_distil_{epochs}_{_METRIC}"
        counts: List[int] = [entry[key] for entry in entities if key in entry]
        assert counts, f"{task}: no entity carries {key}"
        array = np.array(counts)
        summary[task] = {
            "canonical_epochs": epochs,
            "entities_measured": int(array.size),
            "entities_with_at_least_one_receiver_worse_than_the_target": int((array >= 1).sum()),
            "percentage_with_at_least_one": round(100.0 * float((array >= 1).mean()), 1),
            "median_receivers_worse_than_the_target": float(np.median(array)),
            "maximum_receivers_worse_than_the_target": int(array.max()),
        }
        # The same count for the one target this ablation actually studied, so the case study can be placed
        # within the population instead of being generalised from.
        selection = json.loads((_OUT / f"selection_{task}.json").read_text(encoding="utf-8"))
        chosen = selection["target"]["name"]
        matching = [entry[key] for entry in entities if entry["name"] == chosen and key in entry]
        assert len(matching) == 1, f"{task}: {chosen} matched {len(matching)} entities"
        summary[task]["studied_target"] = chosen
        summary[task]["receivers_worse_than_the_studied_target"] = int(matching[0])

        record = summary[task]
        print(f"{task:7s} epochs={epochs:3d}  {record['entities_with_at_least_one_receiver_worse_than_the_target']}"
              f"/{record['entities_measured']} targets ({record['percentage_with_at_least_one']}%) had at least "
              f"one retained entity damaged more than themselves; median {record['median_receivers_worse_than_the_target']}, "
              f"maximum {record['maximum_receivers_worse_than_the_target']}; the studied target "
              f"({record['studied_target']}) has {record['receivers_worse_than_the_studied_target']}")

    out = _OUT / "canonical_worse_than_target.json"
    out.write_text(json.dumps({"method": "distil", "metric": _METRIC, "per_task": summary}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
