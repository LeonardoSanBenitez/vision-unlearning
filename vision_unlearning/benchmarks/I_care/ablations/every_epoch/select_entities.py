"""Automated entity selection for the "changes at every epoch" SPARE study.

For each task (people, scenes, breeds) this selects, with no GPU and from the
already-computed canonical artifacts only, one forget TARGET entity and nine
RECEIVER entities to observe across training epochs:

  * TARGET: among entities genuinely forgotten at the canonical endpoint
    (strongly negative self ``clip_diff``) that did not collapse to noise, the
    one causing the strongest interference (football-field logic).
  * RECEIVERS (interference-first extremes): 2 high-interference + high-similarity,
    2 high-interference + low-similarity, 2 low-interference + high-similarity,
    2 low-interference + low-similarity, and 1 median entity.

Interference and similarity are correlated, so the top-interference pool skews
similar and the bottom-interference pool skews dissimilar; the two off-diagonal
groups are therefore the ones most likely to be less-extreme-than-ideal. This is
inherent to the correlation and is accepted, not corrected here.

The group labels are canonical-endpoint labels: they are defined by the
endpoint measurement and say nothing about how the entities behave at any
intermediate epoch. The study observes how these canonically-labelled entities
evolve; it does not claim the groupings hold at any other epoch.

Inputs (read-only): filtered metadata, the CLIP similarity matrix, and the
per-pair ``clip_diff`` interference files. Outputs, per task: ``selection_{task}.json``
and a similarity-vs-interference scatter ``selection_{task}.png``.

The pure selection functions take plain dictionaries and import nothing heavy;
the data-loading layer imports ``vision_unlearning`` lazily so the logic (and its
unit tests) run without torch.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


# --------------------------------------------------------------------------- #
# Constants (the selection basis; see module docstring)                       #
# --------------------------------------------------------------------------- #
METHOD = "distil"
SIMILARITY_METRIC = "clip"
MODEL = "sd1.4"
SEEDS = [42, 43]

CANONICAL_EPOCHS: Dict[str, int] = {"people": 400, "scenes": 100, "breeds": 100}

# "Genuinely forgotten" floor on self clip_diff (an explicit heuristic on the
# CLIP scale, reported alongside the actual distribution so it can be recalibrated).
FLOOR: Dict[str, float] = {"scenes": -5.0, "people": -5.0, "breeds": -3.0}

# "Not collapsed to noise": at least one retained entity is essentially untouched.
NOT_COLLAPSED_THRESHOLD = -2.0

# Fraction defining the interference extremes and the target scoring pool.
TOP_FRACTION = 0.10

# Receiver group labels (full words; no abbreviations, per the plotting conventions).
GROUP_SIMILAR_HIGH = "similar + high interference"
GROUP_LITTLE_HIGH = "little-similar + high interference"
GROUP_SIMILAR_LOW = "similar + low interference"
GROUP_LITTLE_LOW = "little-similar + low interference"
GROUP_MEDIAN = "median"

TASKS = ("people", "scenes", "breeds")


# --------------------------------------------------------------------------- #
# Pure selection logic (no vision_unlearning import; unit-tested directly)     #
# --------------------------------------------------------------------------- #
def sorted_by_interference(clip_diff_by_receiver: Dict[str, float]) -> List[str]:
    """Receiver names sorted by ``clip_diff`` ascending (most interfered first).

    More negative ``clip_diff`` means more interference. Ties are broken by name
    ascending, so the order is fully deterministic.
    """
    return sorted(clip_diff_by_receiver, key=lambda r: (clip_diff_by_receiver[r], r))


def pool_size(n_receivers: int, fraction: float = TOP_FRACTION) -> int:
    """Number of receivers in each extreme pool: ``round(fraction * n)`` but at
    least 4, because each pool must yield two most-similar and two least-similar
    entities without overlap."""
    return max(4, round(fraction * n_receivers))


def interference_pools(
    clip_diff_by_receiver: Dict[str, float],
    fraction: float = TOP_FRACTION,
) -> Tuple[List[str], List[str]]:
    """Return (top_pool, bottom_pool): the most-interfered and least-interfered
    receivers. The pools are guaranteed disjoint (raises otherwise)."""
    n = len(clip_diff_by_receiver)
    k = pool_size(n, fraction)
    ordered = sorted_by_interference(clip_diff_by_receiver)
    top_pool = ordered[:k]        # most negative clip_diff = most interfered
    bottom_pool = ordered[-k:]    # least negative / positive = least interfered
    overlap = set(top_pool) & set(bottom_pool)
    if overlap:
        raise ValueError(
            f"interference pools overlap ({sorted(overlap)}); n={n}, k={k} is too "
            "large relative to the number of receivers"
        )
    return top_pool, bottom_pool


def two_most_two_least_similar(
    pool: List[str],
    similarity_by_receiver: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    """From a pool, the 2 most-similar and 2 least-similar receivers (to the
    target). Sorted by similarity descending, ties broken by name ascending."""
    if len(pool) < 4:
        raise ValueError(f"pool of size {len(pool)} cannot yield 2 most + 2 least similar")
    ordered = sorted(pool, key=lambda r: (-similarity_by_receiver[r], r))
    most_similar = ordered[:2]
    least_similar = ordered[-2:]
    return most_similar, least_similar


def median_receiver(
    clip_diff_by_receiver: Dict[str, float],
    similarity_by_receiver: Dict[str, float],
    already_picked: Set[str],
    central_fraction: float = 0.20,
) -> str:
    """One roughly-central receiver: among those in the central interference band
    (the middle ``central_fraction`` by interference rank), the one whose CLIP
    similarity is nearest the median similarity, skipping any already picked. If
    the band is exhausted, widen to all receivers."""
    ordered = sorted_by_interference(clip_diff_by_receiver)
    n = len(ordered)
    lo = int((0.5 - central_fraction / 2) * n)
    hi = int((0.5 + central_fraction / 2) * n)
    band = ordered[lo:hi] if hi > lo else ordered
    median_similarity = float(np.median(list(similarity_by_receiver.values())))

    def nearest(candidates: List[str]) -> Optional[str]:
        for receiver in sorted(
            candidates,
            key=lambda r: (abs(similarity_by_receiver[r] - median_similarity), r),
        ):
            if receiver not in already_picked:
                return receiver
        return None

    chosen = nearest(band)
    if chosen is None:
        chosen = nearest(ordered)
    if chosen is None:
        raise ValueError("no median receiver available (all receivers already picked)")
    return chosen


def select_receiver_groups(
    clip_diff_by_receiver: Dict[str, float],
    similarity_by_receiver: Dict[str, float],
    fraction: float = TOP_FRACTION,
) -> Dict[str, str]:
    """Select the 9 receivers and return ``{receiver_name: group_label}``.

    Exactly two receivers in each of the four interference x similarity groups,
    plus one median receiver; all nine distinct (raises otherwise).
    """
    top_pool, bottom_pool = interference_pools(clip_diff_by_receiver, fraction)
    high_most, high_least = two_most_two_least_similar(top_pool, similarity_by_receiver)
    low_most, low_least = two_most_two_least_similar(bottom_pool, similarity_by_receiver)

    labels: Dict[str, str] = {}
    for receiver in high_most:
        labels[receiver] = GROUP_SIMILAR_HIGH
    for receiver in high_least:
        labels[receiver] = GROUP_LITTLE_HIGH
    for receiver in low_most:
        labels[receiver] = GROUP_SIMILAR_LOW
    for receiver in low_least:
        labels[receiver] = GROUP_LITTLE_LOW
    if len(labels) != 8:
        raise ValueError(
            f"expected 8 distinct group receivers, got {len(labels)} "
            "(most/least-similar picks collided within a pool)"
        )

    median = median_receiver(clip_diff_by_receiver, similarity_by_receiver, set(labels))
    labels[median] = GROUP_MEDIAN
    if len(labels) != 9:
        raise ValueError("median receiver collided with a group receiver")
    return labels


def target_interference_score(
    clip_diff_by_receiver: Dict[str, float],
    fraction: float = TOP_FRACTION,
) -> float:
    """Mean ``clip_diff`` over the target's top-``fraction`` most-interfered
    receivers. More negative means the target damages something more strongly."""
    n = len(clip_diff_by_receiver)
    k = max(1, round(fraction * n))
    ordered = sorted_by_interference(clip_diff_by_receiver)
    return float(np.mean([clip_diff_by_receiver[r] for r in ordered[:k]]))


def target_gates(
    self_clip_diff: float,
    receiver_clip_diffs: Dict[str, float],
    floor: float,
    not_collapsed_threshold: float = NOT_COLLAPSED_THRESHOLD,
) -> Tuple[bool, bool]:
    """Return (is_forgotten, is_not_collapsed) for a candidate target.

    * forgotten: the target's own ``clip_diff`` is below the task floor.
    * not collapsed: at least one receiver is essentially untouched (the maximum
      receiver ``clip_diff`` is above the not-collapsed threshold), excluding the
      degenerate "everything got destroyed" case.
    """
    is_forgotten = self_clip_diff < floor
    is_not_collapsed = max(receiver_clip_diffs.values()) > not_collapsed_threshold
    return is_forgotten, is_not_collapsed


# --------------------------------------------------------------------------- #
# Data loading (lazy vision_unlearning imports)                               #
# --------------------------------------------------------------------------- #
class TaskInputs:
    """Canonical selection inputs for one task, loaded from local/HuggingFace."""

    def __init__(
        self,
        task: str,
        names: List[str],
        similarity_by_emitter: Dict[str, Dict[str, float]],
        interference_loader: Callable[[int], Optional[Dict[str, Dict[str, float]]]],
    ) -> None:
        self.task = task
        self.names = names
        self.similarity_by_emitter = similarity_by_emitter
        self.interference_loader = interference_loader


def load_task_inputs(task: str, icare_assets: str) -> TaskInputs:
    """Load metadata, the CLIP similarity matrix, and a per-target interference
    loader for ``task``. Imports ``vision_unlearning`` lazily."""
    from vision_unlearning.datasets.testbed import get_metadata_filtered
    from vision_unlearning.benchmarks.I_care.similarity import Similarity
    from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_pair
    from vision_unlearning.benchmarks.I_care.configuration import (
        type_model, type_s, type_task, type_unlearning_algorithm,
    )
    from vision_unlearning.artifact import ArtifactNotAvailableError

    task_literal = cast(type_task, task)
    method_literal = cast(type_unlearning_algorithm, METHOD)

    metadata = get_metadata_filtered(task_literal, base_folder=icare_assets)
    names = [entry["name"] for entry in metadata]

    similarity_rows = Similarity(
        model=cast(type_model, MODEL), task=task_literal,
        similarity_metric=cast(type_s, SIMILARITY_METRIC), base_folder=icare_assets,
    ).compute()
    similarity_by_emitter = {row["emitter"]: row for row in similarity_rows}

    epochs = CANONICAL_EPOCHS[task]

    def interference_loader(index: int) -> Optional[Dict[str, Dict[str, float]]]:
        try:
            return get_interference_per_pair(task_literal, index, method_literal, epochs, base_folder=icare_assets)
        except ArtifactNotAvailableError:
            return None

    return TaskInputs(task, names, similarity_by_emitter, interference_loader)


def target_overwrite(task: str, name: str) -> Tuple[str, str]:
    """(HuggingFace-form name, overwrite concept) for an entity. Lazy import."""
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.benchmarks.I_care.configuration import type_task, type_unlearning_algorithm
    hf_name, overwrite = get_target_overwrite(
        cast(type_task, task), cast(type_unlearning_algorithm, METHOD), name,
    )
    return hf_name, overwrite


# --------------------------------------------------------------------------- #
# Selection over a whole task                                                 #
# --------------------------------------------------------------------------- #
def evaluate_candidate_targets(inputs: TaskInputs) -> List[Dict[str, Any]]:
    """For every entity, compute its self ``clip_diff``, receiver ``clip_diff``s,
    gate flags, and interference score. Missing interference files are skipped
    with a recorded note."""
    floor = FLOOR[inputs.task]
    candidates: List[Dict[str, Any]] = []
    for index, name in enumerate(inputs.names):
        interference = inputs.interference_loader(index)
        if interference is None:
            print(f"WARNING: no per-pair interference for {inputs.task}[{index}] {name}; skipped")
            continue
        self_clip_diff = float(interference[name]["clip_diff"])
        receiver_clip_diffs = {
            other: float(interference[other]["clip_diff"])
            for other in inputs.names if other != name
        }
        is_forgotten, is_not_collapsed = target_gates(self_clip_diff, receiver_clip_diffs, floor)
        candidates.append({
            "name": name,
            "index": index,
            "self_clip_diff": self_clip_diff,
            "receiver_clip_diffs": receiver_clip_diffs,
            "is_forgotten": is_forgotten,
            "is_not_collapsed": is_not_collapsed,
            "is_gated": is_forgotten and is_not_collapsed,
            "score": target_interference_score(receiver_clip_diffs),
        })
    return candidates


def choose_target(candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Among gated candidates, the one with the strongest (most negative) score.
    Returns (chosen, top5_gated). Raises if no candidate is gated."""
    gated = [c for c in candidates if c["is_gated"]]
    if not gated:
        raise ValueError("no candidate target passed the forgotten + not-collapsed gates")
    gated_sorted = sorted(gated, key=lambda c: (c["score"], c["name"]))
    return gated_sorted[0], gated_sorted[:5]


def self_clip_diff_distribution(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Percentile summary of self ``clip_diff`` across candidates (for floor
    recalibration)."""
    values = np.array([c["self_clip_diff"] for c in candidates], dtype=float)
    return {
        "minimum": float(values.min()),
        "percentile_25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "percentile_75": float(np.percentile(values, 75)),
        "maximum": float(values.max()),
        "n": int(values.size),
    }


def build_selection(inputs: TaskInputs) -> Dict[str, Any]:
    """Full selection for one task: chosen target + 9 grouped receivers, with all
    values and diagnostics needed for the JSON record and the scatter."""
    candidates = evaluate_candidate_targets(inputs)
    chosen, top5 = choose_target(candidates)

    target_name = chosen["name"]
    receiver_clip_diffs = chosen["receiver_clip_diffs"]
    similarity_row = inputs.similarity_by_emitter[target_name]
    similarity_by_receiver = {r: float(similarity_row[r]) for r in receiver_clip_diffs}

    group_by_receiver = select_receiver_groups(receiver_clip_diffs, similarity_by_receiver)

    target_hf_name, target_overwrite_concept = target_overwrite(inputs.task, target_name)

    receivers: List[Dict[str, Any]] = []
    for receiver, group in group_by_receiver.items():
        hf_name, _ = target_overwrite(inputs.task, receiver)
        receivers.append({
            "name": receiver,
            "hf_name": hf_name,
            "clip_similarity": similarity_by_receiver[receiver],
            "clip_diff": receiver_clip_diffs[receiver],
            "group": group,
        })
    # Deterministic record order: by group then by name.
    group_order = [
        GROUP_SIMILAR_HIGH, GROUP_LITTLE_HIGH, GROUP_SIMILAR_LOW, GROUP_LITTLE_LOW, GROUP_MEDIAN,
    ]
    receivers.sort(key=lambda r: (group_order.index(r["group"]), r["name"]))

    group_counts: Dict[str, int] = {g: 0 for g in group_order}
    for record in receivers:
        group_counts[record["group"]] += 1
    assert group_counts[GROUP_SIMILAR_HIGH] == 2
    assert group_counts[GROUP_LITTLE_HIGH] == 2
    assert group_counts[GROUP_SIMILAR_LOW] == 2
    assert group_counts[GROUP_LITTLE_LOW] == 2
    assert group_counts[GROUP_MEDIAN] == 1
    assert len({r["name"] for r in receivers}) == 9
    assert target_name not in {r["name"] for r in receivers}

    return {
        "task": inputs.task,
        "method": METHOD,
        "model": MODEL,
        "similarity_metric": SIMILARITY_METRIC,
        "canonical_epochs": CANONICAL_EPOCHS[inputs.task],
        "seeds": SEEDS,
        "selection_parameters": {
            "floor": FLOOR[inputs.task],
            "not_collapsed_threshold": NOT_COLLAPSED_THRESHOLD,
            "top_fraction": TOP_FRACTION,
        },
        "target": {
            "name": target_name,
            "hf_name": target_hf_name,
            "overwrite_concept": target_overwrite_concept,
            "index": chosen["index"],
            "self_clip_diff": chosen["self_clip_diff"],
            "interference_score": chosen["score"],
        },
        "target_selection_diagnostics": {
            "n_candidates": len(candidates),
            "n_gated": sum(1 for c in candidates if c["is_gated"]),
            "self_clip_diff_distribution": self_clip_diff_distribution(candidates),
            "top5_gated_by_score": [
                {"name": c["name"], "index": c["index"], "self_clip_diff": c["self_clip_diff"],
                 "interference_score": c["score"]}
                for c in top5
            ],
        },
        "receivers": receivers,
        # Everything needed by the scatter, without re-loading.
        "_scatter": {
            "target_self_similarity": float(similarity_row.get(target_name, float("nan"))),
            "all_receivers": [
                {"name": r, "clip_similarity": float(similarity_row[r]),
                 "clip_diff": receiver_clip_diffs[r]}
                for r in receiver_clip_diffs
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
_GROUP_STYLE: Dict[str, Dict[str, Any]] = {
    GROUP_SIMILAR_HIGH: {"color": "#d62728", "marker": "o"},
    GROUP_LITTLE_HIGH: {"color": "#ff7f0e", "marker": "s"},
    GROUP_SIMILAR_LOW: {"color": "#1f77b4", "marker": "^"},
    GROUP_LITTLE_LOW: {"color": "#2ca02c", "marker": "D"},
    GROUP_MEDIAN: {"color": "#9467bd", "marker": "P"},
}


def _display_method(method: str) -> str:
    return "spare" if method == "distil" else method


def render_scatter(selection: Dict[str, Any], out_path: str) -> None:
    """Similarity-vs-interference scatter: all receivers in grey, the nine
    selected highlighted by group, the target marked separately."""
    task = selection["task"]
    scatter = selection["_scatter"]
    group_by_receiver = {r["name"]: r["group"] for r in selection["receivers"]}

    fig, ax = plt.subplots(figsize=(9, 6))

    background_x = [p["clip_similarity"] for p in scatter["all_receivers"] if p["name"] not in group_by_receiver]
    background_y = [p["clip_diff"] for p in scatter["all_receivers"] if p["name"] not in group_by_receiver]
    ax.scatter(background_x, background_y, s=18, color="#bbbbbb", alpha=0.6,
               label="other receivers", zorder=1)

    for group, style in _GROUP_STYLE.items():
        xs = [r["clip_similarity"] for r in selection["receivers"] if r["group"] == group]
        ys = [r["clip_diff"] for r in selection["receivers"] if r["group"] == group]
        ax.scatter(xs, ys, s=90, color=style["color"], marker=style["marker"],
                   edgecolors="black", linewidths=0.5, label=group, zorder=3)

    target = selection["target"]
    ax.scatter([scatter["target_self_similarity"]], [target["self_clip_diff"]],
               s=200, color="black", marker="*", label="target (self)", zorder=4)
    ax.annotate(
        f"{target['hf_name']}\nself clip_diff={target['self_clip_diff']:.2f}",
        (scatter["target_self_similarity"], target["self_clip_diff"]),
        textcoords="offset points", xytext=(8, 8), fontsize=8,
    )

    ax.axhline(0.0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_xlabel("CLIP similarity (target to receiver)")
    ax.set_ylabel("clip_diff (lower = more interference)")
    diagnostics = selection["target_selection_diagnostics"]
    ax.set_title(
        f"select_entities | task={task} method={_display_method(METHOD)} | "
        f"similarity={SIMILARITY_METRIC} vs clip_diff\n"
        f"target={target['hf_name']} self_clip_diff={target['self_clip_diff']:.2f} | "
        f"gated_targets={diagnostics['n_gated']}/{diagnostics['n_candidates']}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Target-check image (best-effort; needs the target's canonical ON images)     #
# --------------------------------------------------------------------------- #
def _find_image(folder: str, seed: int, hf_name: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    wanted = f"{prefix}_{seed}_An image of {hf_name}.png"
    candidate = os.path.join(folder, wanted)
    if os.path.isfile(candidate):
        return candidate
    return None


def render_target_check(selection: Dict[str, Any], icare_assets: str, out_path: str) -> Optional[str]:
    """Render the chosen target's canonical ON vs OFF image, to confirm it
    degraded into the overwrite concept rather than into noise. Returns the
    output path if rendered, else ``None`` (ON images not available locally)."""
    task = selection["task"]
    target = selection["target"]
    epochs = selection["canonical_epochs"]
    hf_name = target["hf_name"]
    on_folder = os.path.join(icare_assets, "datasets", f"generated_{task}_{hf_name}_{METHOD}_{epochs}")
    off_folder = os.path.join(icare_assets, "datasets", f"generated_{task}_baseline")

    on_path = _find_image(on_folder, 42, hf_name, "on")
    off_path = _find_image(off_folder, 42, hf_name, "off")
    if on_path is None or off_path is None:
        print(
            f"NOTE: target-check image for {task}/{hf_name} skipped -- canonical ON images not "
            f"local (looked in {on_folder}). Fetch that folder to render it."
        )
        return None

    from matplotlib import image as mpimg
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for ax, path, title in ((axes[0], off_path, "off (baseline)"), (axes[1], on_path, f"on ({_display_method(METHOD)})")):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"target-check | task={task} {hf_name}\n"
        f"epochs={epochs} seed=42 | overwrite={target['overwrite_concept']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _default_icare_assets() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "assets"))


def _default_out() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets")


def run(tasks: List[str], icare_assets: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for task in tasks:
        print(f"\n=== selecting entities for task={task} ===")
        inputs = load_task_inputs(task, icare_assets)
        selection = build_selection(inputs)

        json_path = os.path.join(out, f"selection_{task}.json")
        scatter_path = os.path.join(out, f"selection_{task}.png")
        target_check_path = os.path.join(out, f"selection_{task}_target_check.png")

        rendered = render_target_check(selection, icare_assets, target_check_path)
        selection["target_check_image"] = os.path.basename(rendered) if rendered else None

        # Strip the scatter helper before persisting (it is figure-only scratch).
        scatter_data = selection.pop("_scatter")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2)
        selection["_scatter"] = scatter_data
        render_scatter(selection, scatter_path)

        target = selection["target"]
        print(f"target: {target['hf_name']} (index {target['index']}, "
              f"self clip_diff {target['self_clip_diff']:.2f}, score {target['interference_score']:.2f})")
        for receiver in selection["receivers"]:
            print(f"  receiver: {receiver['hf_name']:<32} group={receiver['group']:<32} "
                  f"similarity={receiver['clip_similarity']:.2f} clip_diff={receiver['clip_diff']:.2f}")
        print(f"wrote {json_path}")
        print(f"wrote {scatter_path}")
        if rendered:
            print(f"wrote {target_check_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icare-assets", default=_default_icare_assets(),
                        help="canonical I-CARE assets folder (default: resolved from this file)")
    parser.add_argument("--out", default=_default_out(),
                        help="output folder for selection JSON/figures (default: sibling assets/)")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
                        help="tasks to select for (default: all three)")
    args = parser.parse_args()
    run(args.tasks, args.icare_assets, args.out)


if __name__ == "__main__":
    main()
