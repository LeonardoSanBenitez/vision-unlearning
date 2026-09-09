'''Figure and distance table for `diagnose_render_quality.py`.

Puts the three conditions side by side for the entities whose campaign renders contain no human
subject, and prints the pixel distance between conditions so the control's residual can be judged
against the size of the effect rather than argued about.

    python plot_render_quality_diagnostic.py --seed 42
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
# The three entities the validation report read as containing no human subject at seed 42, plus the
# target for reference. Named explicitly because they are the cases under discussion.
_FOCUS = ["Juan_Carlos_Ferrero", "Andy_Roddick", "Ian_Thorpe", "Mark_Philippoussis"]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    parser = argparse.ArgumentParser(description="Compare the render-quality diagnostic's conditions.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    results = json.loads((_OUT / f"diagnose_render_quality_seed{seed}.json").read_text(encoding="utf-8"))
    conditions: List[str] = list(results["conditions"])
    metric = MetricImageTextSimilarity(metrics=["clip"])

    def array(path: str) -> Any:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)

    # Distance between conditions, on the same scale as the control's own residual.
    distances: Dict[str, Dict[str, float]] = {}
    for name in results["conditions"][conditions[0]]["per_entity"]:
        base = array(results["conditions"][conditions[0]]["per_entity"][name]["path"])
        distances[name] = {}
        for condition in conditions[1:]:
            other = array(results["conditions"][condition]["per_entity"][name]["path"])
            distances[name][condition] = float(np.abs(base - other).max())

    fig, axes = plt.subplots(len(_FOCUS), len(conditions), figsize=(3.3 * len(conditions), 3.7 * len(_FOCUS)))
    clip_scores: Dict[str, Dict[str, float]] = {}
    for row, name in enumerate(_FOCUS):
        prompt = f"An image of {get_target_overwrite('people', 'distil', name)[0]}"
        clip_scores[name] = {}
        for column, condition in enumerate(conditions):
            path = results["conditions"][condition]["per_entity"][name]["path"]
            image = Image.open(path).convert("RGB")
            score = float(metric.score_batch_same_text([image], prompt)[0]["clip"])
            clip_scores[name][condition] = score
            axes[row][column].imshow(image)
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
            axes[row][column].set_title(f"{condition}\nclip score {score:.2f}", fontsize=8)
        axes[row][0].set_ylabel(name.replace("_", " "), fontsize=9)
    fig.suptitle(f"base model only, no adapter, seed {seed}, all conditions rendered at 512 pixels, 50 steps\n"
                 "campaign = what the campaign did (original_size defaults to the render size); "
                 "the others override the micro-conditioning", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = _OUT / f"diagnose_render_quality_seed{seed}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)

    summary = {"seed": seed, "clip_scores": clip_scores, "condition_distances": distances,
               "control_residual": results["control"]}
    (_OUT / f"diagnose_render_quality_seed{seed}_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"control residual (campaign replay vs campaign images on disk): maximum absolute pixel "
          f"difference {results['control']['max_absolute_difference_over_all_entities']}, "
          f"identical {results['control']['n_identical']} of {results['control']['n_compared']}")
    for condition in conditions[1:]:
        values = [distances[name][condition] for name in distances]
        print(f"{condition} vs campaign: maximum absolute pixel difference over 10 entities -> "
              f"min {min(values)}, median {sorted(values)[len(values) // 2]}, max {max(values)}")
    print("clip score by condition, for the entities under discussion:")
    for name in _FOCUS:
        scores = " | ".join(f"{condition} {clip_scores[name][condition]:6.2f}" for condition in conditions)
        print(f"  {name:22s} {scores}")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
