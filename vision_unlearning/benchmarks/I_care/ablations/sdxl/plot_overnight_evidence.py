'''Contact sheets for `overnight_evidence.py`, so 126 loose files become four reviewable figures.

Reads only what is already on disk. The two that matter are the position sweep and its reseeded control:
they are laid out identically on purpose, so the pair can be compared cell by cell.

    python plot_overnight_evidence.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"


def _grid(paths: List[List[str]], row_labels: List[str], column_labels: List[str],
          title: str, out_path: Path, cell: float = 1.7) -> Path:
    import matplotlib.pyplot as plt
    from PIL import Image

    rows, columns = len(paths), len(paths[0])
    fig, axes = plt.subplots(rows, columns, figsize=(cell * columns, (cell + 0.45) * rows))
    axes = axes.reshape(rows, columns)
    for row in range(rows):
        for column in range(columns):
            axes[row][column].imshow(Image.open(paths[row][column]).convert("RGB"))
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
            if row == 0:
                axes[row][column].set_title(column_labels[column], fontsize=7)
        axes[row][0].set_ylabel(row_labels[row], fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=95)
    plt.close(fig)
    return out_path


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    results: Dict[str, Any] = json.loads((_OUT / "overnight_evidence.json").read_text(encoding="utf-8"))
    blocks = results["blocks"]
    written: List[Path] = []

    block2 = blocks["2_position_sweep"]["per_entity_by_position"]
    names = list(block2)
    written.append(_grid(
        [block2[name] for name in names], [n.replace("_", " ") for n in names],
        [f"position {i}" for i in range(10)],
        "block 2 -- the SAME prompt ten times in one call, one generator advanced across them "
        "(what the campaign did)\nbase model, no adapter, seed 42, 512 pixels, 50 steps; only the position "
        "in the call differs between columns",
        _OUT / "overnight_block2_position_sweep.png"))

    block3 = blocks["3_reseeded_control"]["per_entity_by_call_index"]
    written.append(_grid(
        [block3[name] for name in names], [n.replace("_", " ") for n in names],
        [f"call {i}" for i in range(10)],
        "block 3 -- the SAME ten calls in the same process, generator RESEEDED before each\n"
        "identical to block 2 in every other respect; if these are clean, the defect is generator state",
        _OUT / "overnight_block3_reseeded_control.png"))

    block4 = blocks["4_parameter_grid"]["per_entity"]
    grid_names = list(block4)
    settings = list(block4[grid_names[0]])
    written.append(_grid(
        [[block4[name][setting] for setting in settings] for name in grid_names],
        [n.replace("_", " ") for n in grid_names], settings,
        "block 4 -- guidance and micro-conditioning, each entity generated ALONE (position 0), seed 42",
        _OUT / "overnight_block4_parameter_grid.png", cell=2.1))

    block1 = blocks["1_baselines_alone"]["per_seed"]
    seeds = list(block1)
    entities = list(block1[seeds[0]])
    written.append(_grid(
        [[block1[seed][name] for name in entities] for seed in seeds],
        [f"seed {seed}" for seed in seeds], [n.replace("_", " ") for n in entities],
        "block 1 -- corrected off-baselines: every entity generated ALONE, five seeds\n"
        "this is the multi-seed baseline for clip_diff and the input to re-measuring the noise floor",
        _OUT / "overnight_block1_baselines.png", cell=1.9))

    for path in written:
        print(f"written: {path}")


if __name__ == "__main__":
    main()
