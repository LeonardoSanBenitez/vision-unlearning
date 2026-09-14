'''Is the seed labelling real, or did two runs get their images crossed?

The claim under challenge is that seed 42 renders these people well at 512 pixels and seed 43 does
not. It rests on file names, and file names are exactly the kind of thing that can be wrong. This
check does not trust them: it compares, pixel by pixel, images produced by TWO DIFFERENT SCRIPTS in
TWO DIFFERENT PROCESSES on different days --

    overnight_evidence.py  ->  assets/overnight_evidence/baseline_alone_<entity>_seed<seed>.png
    rescue_grid.py         ->  assets/rescue_grid/campaign_defaults_<entity>_seed<seed>.png

-- for every pair of seeds. If the labelling is honest, the diagonal (same seed in both runs) is
near zero and every off-diagonal entry is large: two independent runs agreed about which seed
produces which image. If any script had crossed its seeds, the matrix would not be diagonal.

The verdict is computed from the printed matrix, not written beside it.

    PYTHONPATH=<repo root> python check_seed_labels.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_BLOCK1 = _OUT / "overnight_evidence"
_RESCUE = _OUT / "rescue_grid"
_RESULT = _OUT / "check_seed_labels.json"

_SEEDS = [42, 43, 45]
_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]


def _mean_abs_difference(left: Path, right: Path) -> float:
    import numpy as np
    from PIL import Image
    with Image.open(left) as a, Image.open(right) as b:
        first = np.asarray(a.convert("RGB"), dtype=np.int64)
        second = np.asarray(b.convert("RGB"), dtype=np.int64)
    return round(float(np.mean(np.abs(first - second))), 4)


def main() -> None:
    payload: Dict[str, Any] = {"entities": _ENTITIES, "seeds": _SEEDS, "matrices": {}}
    diagonals: List[float] = []
    off_diagonals: List[float] = []

    for entity in _ENTITIES:
        print(f"\n{entity}: mean absolute pixel difference, overnight run (rows) against "
              f"rescue grid (columns)")
        header = "        " + "".join(f"{seed:>12}" for seed in _SEEDS)
        print(header)
        matrix: Dict[str, Dict[str, float]] = {}
        for row_seed in _SEEDS:
            left = _BLOCK1 / f"baseline_alone_{entity}_seed{row_seed}.png"
            row: Dict[str, float] = {}
            cells = []
            for column_seed in _SEEDS:
                right = _RESCUE / f"campaign_defaults_{entity}_seed{column_seed}.png"
                value = _mean_abs_difference(left, right)
                row[str(column_seed)] = value
                cells.append(f"{value:>12.4f}")
                (diagonals if row_seed == column_seed else off_diagonals).append(value)
            matrix[str(row_seed)] = row
            print(f"seed {row_seed}" + "".join(cells))
        payload["matrices"][entity] = matrix

    payload["diagonal_maximum"] = max(diagonals)
    payload["off_diagonal_minimum"] = min(off_diagonals)
    payload["diagonal_count"] = len(diagonals)
    payload["off_diagonal_count"] = len(off_diagonals)
    payload["labelling_consistent_across_the_two_runs"] = max(diagonals) < min(off_diagonals)

    print(f"\ndiagonal ({len(diagonals)} cells, same seed in both runs): maximum {max(diagonals)}")
    print(f"off-diagonal ({len(off_diagonals)} cells, different seeds): minimum {min(off_diagonals)}")
    print(f"diagonal maximum < off-diagonal minimum: {max(diagonals) < min(off_diagonals)}")
    _RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"CHECK_SEED_LABELS_DONE written={_RESULT}")


if __name__ == "__main__":
    main()
