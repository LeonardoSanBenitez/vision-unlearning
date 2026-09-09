'''Does similarity to the forget target predict interference, on this base model and on the previous one?

This is the relationship the benchmark exists to measure, and the question the paper answers with
`MetricSimilarityAlignment`: whether a `SimilarityBetweenEntities` can stand in for an expensive
`MetricInterferencePerEntityPair`. The paper's answer over its whole feasibility demonstration is that
no similarity predicts any interference metric well. This script asks the same question of one emitter
on two base models, so the two answers can be put beside each other.

Three sources of interference, deliberately not merged, because they are measured at different epochs
over different entity sets:

* **Stable Diffusion 1.4, canonical endpoint.** The benchmark's own per-pair artifact for this target
  at 400 epochs, covering 99 receivers. This is the paper's data.
* **Stable Diffusion 1.4, every-epoch ablation.** The same nine receivers this task measures, at their
  worst and last checkpoint of the thirteen.
* **Stable Diffusion XL, this task.** Nineteen entities: the nine receivers plus the ten drawn at
  random for the control, at their worst and last checkpoint, per seed.

The nineteen matter. Nine points cannot support a correlation, and the nine were chosen by a design that
spans the similarity axis on purpose (two entities in each cell of similar/dissimilar by high/low
interference, plus a median one), so a correlation over them alone would be reading a sampling design.
The ten random entities are what make the coefficient mean anything on our side.

DIRECTION, from `CONTRIBUTING_ICARE.md` section 5: `clip_diff = clip_on - clip_off`, so more negative is
more interference, and the hypothesis "more similar entities suffer more interference" predicts a
NEGATIVE correlation. The same convention holds for the canonical artifact.

    PYTHONPATH=<repo root> python similarity_alignment_sdxl.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import pearsonr, spearmanr

import campaign_configuration as cfg
import sd14_campaign

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_ICARE_ASSETS = cfg.ICARE_DIR / "assets"
_SIMILARITY_METRICS = ["clip", "dino", "act", "jacc"]
_CANONICAL_EPOCHS = 400
_TARGET_INDEX_KEY = "index"


def similarity_to_target(metric: str, target: str) -> Dict[str, float]:
    '''Similarity of every entity to the target, from the benchmark's own matrix for that metric.

    The file is a list of rows, one per emitter, each a dictionary of receiver name to value plus an
    `emitter` field. The target's own row is used, and the target is dropped from it: an entity's
    similarity to itself is not a data point about interference.
    '''
    rows = json.loads((_ICARE_ASSETS / f"similarity_{metric}_people.json").read_text(encoding="utf-8"))
    matching = [row for row in rows if row["emitter"] == target]
    assert len(matching) == 1, f"expected one row for {target} in similarity_{metric}_people.json"
    row = matching[0]
    return {name: float(value) for name, value in row.items()
            if name != "emitter" and name != target}


def canonical_interference(target: str) -> Dict[str, float]:
    '''`clip_diff` caused by unlearning the target, at the canonical endpoint, for all 99 receivers.'''
    from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_pair

    selection = json.loads(cfg.SELECTION.read_text(encoding="utf-8"))
    index = int(selection["target"][_TARGET_INDEX_KEY])
    per_pair = get_interference_per_pair("people", index, "distil", _CANONICAL_EPOCHS,
                                         base_folder=str(_ICARE_ASSETS))
    return {name: float(values["clip_diff"]) for name, values in per_pair.items() if name != target}


def _worst_and_final(trajectory: List[Dict[str, Any]]) -> Tuple[float, float]:
    worst = min(point["clip_diff"] for point in trajectory)
    return float(worst), float(trajectory[-1]["clip_diff"])


def sdxl_interference(seed: int) -> Dict[str, Tuple[float, float]]:
    '''(worst, final) `clip_diff` for the nine receivers and the ten control entities at one seed.'''
    campaign = json.loads((_OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    control = json.loads((_OUT / "clip_diff_random_ten_control.json").read_text(encoding="utf-8"))
    target = campaign["target"]
    values: Dict[str, Tuple[float, float]] = {}
    for name, payload in campaign["per_seed"][str(seed)]["per_entity"].items():
        if name == target:
            continue
        values[name] = _worst_and_final(payload["trajectory"])
    for name, payload in control["per_seed"][str(seed)]["per_entity"].items():
        values[name] = _worst_and_final(payload["trajectory"])
    return values


def sd14_every_epoch_interference(seed: int, names: List[str]) -> Dict[str, Tuple[float, float]]:
    '''The same two numbers from the every-epoch ablation, for whichever of `names` it measured.'''
    available = set(sd14_campaign.generation_order(seed))
    values: Dict[str, Tuple[float, float]] = {}
    for name in names:
        if name not in available:
            continue
        _, by_epoch, epochs = sd14_campaign.entity_cells(seed, name)
        trajectory = [{"clip_diff": by_epoch[epoch]} for epoch in epochs]
        values[name] = _worst_and_final(trajectory)
    return values


def correlate(similarity: Dict[str, float], interference: Dict[str, float]) -> Dict[str, Any]:
    '''Pearson and Spearman over the entities present in both, with the count that produced them.'''
    shared = sorted(set(similarity) & set(interference))
    x = [similarity[name] for name in shared]
    y = [interference[name] for name in shared]
    if len(shared) < 3:
        return {"n": len(shared), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan"), "entities": shared}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {"n": len(shared), "pearson_r": float(pearson[0]), "pearson_p": float(pearson[1]),
            "spearman_rho": float(spearman[0]), "spearman_p": float(spearman[1]), "entities": shared}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = json.loads(cfg.SELECTION.read_text(encoding="utf-8"))
    target = selection["target"]["name"]
    receivers = [receiver["name"] for receiver in selection["receivers"]]
    group_of = {receiver["name"]: receiver["group"] for receiver in selection["receivers"]}
    control_entities = json.loads(
        (_OUT / "random_ten_control_entities.json").read_text(encoding="utf-8"))["entities"]

    similarity = {metric: similarity_to_target(metric, target) for metric in _SIMILARITY_METRICS}
    canonical = canonical_interference(target)

    sources: List[Tuple[str, Dict[str, float], str]] = [
        ("stable diffusion 1.4, canonical endpoint, 99 receivers",
         canonical, "clip_diff at epoch 400"),
    ]
    for seed in [42, 43]:
        sdxl = sdxl_interference(seed)
        sources.append((f"stable diffusion xl, seed {seed}, 9 receivers + 10 random",
                        {name: worst for name, (worst, _) in sdxl.items()},
                        "worst clip_diff over 13 checkpoints"))
        sources.append((f"stable diffusion xl, seed {seed}, 9 receivers + 10 random",
                        {name: final for name, (_, final) in sdxl.items()},
                        "clip_diff at epoch 200"))
        sd14 = sd14_every_epoch_interference(seed, receivers)
        sources.append((f"stable diffusion 1.4 every-epoch, seed {seed}, 9 receivers",
                        {name: worst for name, (worst, _) in sd14.items()},
                        "worst clip_diff over 13 checkpoints"))

    results: List[Dict[str, Any]] = []
    for label, interference, definition in sources:
        for metric in _SIMILARITY_METRICS:
            entry = correlate(similarity[metric], interference)
            entry.update({"source": label, "interference": definition, "similarity": metric})
            results.append(entry)

    lines = ["# Does similarity to the forget target predict interference?", "",
             "`clip_diff = clip_on - clip_off`, so more negative is more interference and the hypothesis",
             "\"more similar suffers more\" predicts a NEGATIVE correlation.", "",
             "| source | interference | similarity | n | Pearson r | p | Spearman rho | p |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for entry in results:
        lines.append(
            f"| {entry['source']} | {entry['interference']} | {entry['similarity']} | {entry['n']} | "
            f"{entry['pearson_r']:+.3f} | {entry['pearson_p']:.3f} | {entry['spearman_rho']:+.3f} | "
            f"{entry['spearman_p']:.3f} |")

    lines += ["", "## The nine receivers, their selection group, and what happened to them", "",
              "| receiver | selection group | clip similarity to target | "
              "sd1.4 canonical clip_diff | sdxl worst seed 42 | sdxl worst seed 43 |",
              "|---|---|---:|---:|---:|---:|"]
    worst42 = {name: worst for name, (worst, _) in sdxl_interference(42).items()}
    worst43 = {name: worst for name, (worst, _) in sdxl_interference(43).items()}
    for name in receivers:
        lines.append(
            f"| {name.replace('_', ' ')} | {group_of[name]} | {similarity['clip'][name]:.1f} | "
            f"{canonical.get(name, float('nan')):+.2f} | {worst42[name]:+.2f} | {worst43[name]:+.2f} |")
    lines += ["", "| control entity | drawn at random | clip similarity to target | "
              "sd1.4 canonical clip_diff | sdxl worst seed 42 | sdxl worst seed 43 |",
              "|---|---|---:|---:|---:|---:|"]
    for name in control_entities:
        lines.append(
            f"| {name.replace('_', ' ')} | yes | {similarity['clip'][name]:.1f} | "
            f"{canonical.get(name, float('nan')):+.2f} | {worst42[name]:+.2f} | {worst43[name]:+.2f} |")

    table_path = _OUT / "similarity_alignment_sdxl.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (_OUT / "similarity_alignment_sdxl.json").write_text(
        json.dumps({"target": target, "similarity_metrics": _SIMILARITY_METRICS,
                    "results": results}, indent=2), encoding="utf-8")

    # One row per source that is drawn, one column per similarity metric, so the two models are read
    # against each other rather than in separate figures.
    drawn = [("stable diffusion 1.4, canonical endpoint, 99 receivers", canonical,
              "clip_diff at epoch 400", None),
             ("stable diffusion xl, seed 42, 9 receivers + 10 random",
              {name: worst for name, worst in worst42.items()},
              "worst clip_diff over 13 checkpoints", set(receivers)),
             ("stable diffusion xl, seed 43, 9 receivers + 10 random",
              {name: worst for name, worst in worst43.items()},
              "worst clip_diff over 13 checkpoints", set(receivers))]
    figure, axes = plt.subplots(len(drawn), len(_SIMILARITY_METRICS),
                                figsize=(4.0 * len(_SIMILARITY_METRICS), 3.4 * len(drawn)),
                                squeeze=False)
    for row, (label, interference, definition, selected) in enumerate(drawn):
        for column, metric in enumerate(_SIMILARITY_METRICS):
            axis = axes[row][column]
            shared = sorted(set(similarity[metric]) & set(interference))
            for name in shared:
                is_selected = selected is not None and name in selected
                axis.scatter(similarity[metric][name], interference[name], s=26,
                             facecolors="tab:orange" if is_selected else "tab:blue",
                             edgecolors="none", alpha=0.8)
            entry = correlate(similarity[metric], interference)
            axis.axhline(-cfg.noise_floor_standard_deviation(), color="0.6", linewidth=0.8,
                         linestyle="--")
            axis.set_title(f"similarity={metric}, n={entry['n']}\n"
                           f"Pearson {entry['pearson_r']:+.3f}, p={entry['pearson_p']:.3f}; "
                           f"Spearman {entry['spearman_rho']:+.3f}, p={entry['spearman_p']:.3f}",
                           fontsize=8)
            axis.set_xlabel(f"{metric} similarity to the forget target", fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{label}\n{definition}", fontsize=7)
            axis.grid(alpha=0.25)
    axes[1][0].scatter([], [], color="tab:orange", label="one of the nine selected receivers")
    axes[1][0].scatter([], [], color="tab:blue", label="drawn at random")
    axes[1][0].legend(fontsize=7, loc="lower left")
    figure.suptitle(
        f"task={cfg.TASK} | method=spare | forget target={target.replace('_', ' ')} | "
        f"dashed line = minus the stable diffusion xl noise floor\n"
        f"clip_diff = clip_on - clip_off, so lower means more interference and the hypothesis "
        f"'more similar suffers more' predicts a downward slope",
        fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = _OUT / "similarity_alignment_sdxl.png"
    figure.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    for entry in results:
        print(f"{entry['source'][:46]:<46} {entry['interference'][:34]:<34} {entry['similarity']:<5} "
              f"n={entry['n']:>3} Pearson {entry['pearson_r']:+.3f} (p={entry['pearson_p']:.3f}) "
              f"Spearman {entry['spearman_rho']:+.3f} (p={entry['spearman_p']:.3f})")
    print(f"written: {table_path}")
    print(f"written: {figure_path}")
    print("SIMILARITY_ALIGNMENT_DONE")


if __name__ == "__main__":
    main()
