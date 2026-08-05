"""Every-epoch x every-entity grid (the W6 deliverable format), for one task's selection.

Rows = [original base model, then each saved epoch]; columns = the forget target followed by the nine
receivers, ordered by their `clip_diff` in the LAST rendered epoch of this run (most negative, i.e. most
interfered, nearest the target). Each cell is the entity's own concept prompt generated from that epoch's
adapter, at a fixed seed, batch_size=1 (dedicated-VRAM path). Runs for one or more seeds (--seeds); each
seed's figure is ordered by its own last-epoch values, so a figure is self-contained: every number shown
on it comes from that figure's own run and seed, and the bottom row is the sort key.

Generation order is separate from column order, and is NOT the sort key: it is fixed once (by the
canonical selection file) because it determines each entity's position in the random-number sequence.
Column order is a presentation choice applied afterwards.

Seed-matching invariant (critical): for a given seed, the baseline AND every epoch are generated within
THIS script from the SAME prompt list in the SAME order, so an entity sits at the same RNG position in every
call and starts from IDENTICAL initial noise in every row (generate_dataset reseeds to `seed` at the start
of each call and advances the generator once per prompt). Images are therefore NOT reused across scripts or
across different orderings (doing so puts an entity at a different RNG position and silently breaks the seed
match) - the reason the grid generates all its own cells with one ordering. A manifest written next to the
images records that ordering, so --reuse-existing-images cannot silently pair one run's images with another
run's entity list.

Self-audit: over the CONTROL entities (those not strongly forgotten, which should evolve smoothly) the
script measures the mean-abs pixel change between consecutive rows, and flags any EPOCH-to-epoch transition
that exceeds the median of the epoch-to-epoch transitions by more than a factor. That baseline deliberately
excludes the original->epoch-1 change, so a badly generated reference row cannot inflate the bar. The
original->epoch-1 change itself is reported as a ratio to that baseline but is NOT a pass/fail criterion:
it is raised both by a mismatched reference row and by the adapter simply appearing for the first time, and
measurements on real runs show the two ranges overlap (see the constant block below).

Writes, per seed, epoch_grid{run-suffix}_seed{seed}.png and .json (clip_diff per cell + audit). The
adapters to render, the learning rate shown in the title, and the run suffix are CLI arguments, so a
hyperparameter variant (e.g. a smaller learning rate) is rendered by the same code into its own files.
Run with PYTHONPATH at the repo root, HF_TOKEN set, HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_METHOD: Literal["distil"] = "distil"
_FORGOTTEN_CLIPDIFF = -5.0   # an entity with last-epoch clip_diff below this is "being forgotten" (not a control)
_OUTLIER_FACTOR = 2.0        # an epoch-to-epoch transition > this x the baseline (over controls) is flagged


# --------------------------------------------------------------------------- #
# The parts a wrong implementation would still make look plausible: the on/off #
# pairing, the sign of clip_diff, the column order, and the row/column mapping #
# of the figure. They are module-level and dependency-light so that            #
# test_make_epoch_grid.py can exercise them without a GPU or a real model.     #
# --------------------------------------------------------------------------- #
def score_cells(
    number_of_entities: int,
    epochs: List[int],
    score_off: Callable[[int], float],
    score_on: Callable[[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """clip_diff for every (row, entity), as `on - off` against that SAME entity's original image.

    Row 0 is the original model and is zero by definition; row ``r`` (1-based) is ``epochs[r - 1]``.
    The two scoring callables receive an entity index (and, for the epochs, the epoch number), which is
    what keeps an entity paired with its own baseline instead of a neighbour's.
    """
    clip_diff: Dict[Tuple[int, int], float] = {}
    for entity in range(number_of_entities):
        off = score_off(entity)
        clip_diff[(0, entity)] = 0.0
        for row, epoch in enumerate(epochs, start=1):
            clip_diff[(row, entity)] = score_on(entity, epoch) - off
    return clip_diff


def column_order(last_row: Dict[int, float], names: List[str], target_index: int) -> List[int]:
    """Figure columns: the target first, then the receivers by ascending clip_diff in the last epoch.

    More negative means more interfered, so the most damaged receiver sits next to the target. Ties break
    on the entity name, so the order is deterministic. The target is pinned rather than sorted: it is the
    subject of the figure, and its number measures its own forgetting rather than interference.
    """
    receivers = sorted((index for index in last_row if index != target_index),
                       key=lambda index: (last_row[index], names[index]))
    return [target_index] + receivers


def audit_transitions(transitions: List[float], outlier_factor: float = _OUTLIER_FACTOR) -> Dict[str, Any]:
    """Judge the row-to-row image changes over the control entities.

    ``transitions[0]`` is the original->epoch-1 change and the rest are epoch-to-epoch. The baseline is the
    median of the epoch-to-epoch ones ONLY: a reference row generated with the wrong seed or ordering
    inflates the first transition, and including it in the baseline lets it partly hide itself (measured:
    a real mismatched reference scored 1.69 against an all-transitions median, below any usable threshold,
    but 2.09 against this one).

    ``reference_row_over_baseline`` is reported and deliberately not part of ``passed``: it is also raised
    by the adapter appearing for the first time, which dominates when later epochs move little (measured
    1.07 and 1.25 on known-good runs, but 1.71 and 1.88 on slow ones, against 2.09 for the mismatched row -
    overlapping, so it cannot carry a verdict alone).
    """
    baseline = statistics.median(transitions[1:])
    outliers = [row for row, value in enumerate(transitions[1:], start=2)
                if baseline > 0 and value > outlier_factor * baseline]
    return {
        "row_to_row_change_over_controls": [round(value, 2) for value in transitions],
        "epoch_to_epoch_baseline": round(baseline, 2),
        "reference_row_over_baseline": round(transitions[0] / baseline, 3) if baseline > 0 else 0.0,
        "outlier_epoch_transitions": outliers,
        "passed": not outliers,
    }


def render_grid(
    cell: Dict[Tuple[int, int], Path],
    clip_diff: Dict[Tuple[int, int], float],
    row_labels: List[str],
    column_labels: List[str],
    display_order: List[int],
    title: str,
    out_png: Path,
) -> None:
    """Draw the grid. Row ``r`` of the figure holds row ``r`` of the data, and figure column ``c`` holds
    the entity ``display_order[c]`` - the one place a transposed or re-sorted axis would go unnoticed,
    since any wrong mapping still produces a full, plausible-looking figure.

    ``column_labels`` and ``clip_diff`` are indexed by ENTITY, not by column, so the labels cannot drift
    away from the images when the column order changes.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    number_of_rows, number_of_columns = len(row_labels), len(display_order)
    # squeeze=False keeps `axes` two-dimensional even for a single row or column, which subplots would
    # otherwise collapse into a flat array and break the [row][column] indexing below.
    fig, axes = plt.subplots(number_of_rows, number_of_columns, squeeze=False,
                             figsize=(1.7 * number_of_columns, 1.9 * number_of_rows))
    for row in range(number_of_rows):
        for column, entity in enumerate(display_order):
            ax = axes[row][column]
            ax.imshow(np.asarray(Image.open(cell[(row, entity)]).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            if row > 0:  # the original row is zero by definition and is left blank
                ax.set_title(f"clip_diff={clip_diff[(row, entity)]:.2f}", fontsize=6)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, rotation=0, ha="right", va="center")
    for column, entity in enumerate(display_order):
        label = column_labels[entity]
        axes[0][column].set_title(f"{label} (target)" if column == 0 else label, fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def manifest_differences(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Fields in which an existing image folder disagrees with the run about to reuse it.

    The images are named by position in the generation order, so a different entity list would be silently
    relabelled rather than rejected; every field of the manifest must therefore match exactly.
    """
    return sorted(key for key in current if stored.get(key) != current[key])


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.metrics import MetricImageTextSimilarity
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers
    from vision_unlearning.benchmarks.I_care.result_templates import (
        _display_unlearning_algorithm, _short_entity_display,
    )

    logger = get_logger("epoch_grid")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Every-epoch x every-entity grid (one grid per seed).")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43")
    parser.add_argument("--task", choices=["breeds", "people", "scenes"], default="breeds",
                        help="Which task's selection to render; entities and their canonical "
                             "interference come from assets/selection_{task}.json.")
    parser.add_argument("--model-dir", default=str(_OUT / "models" / "breeds_demo_distil_30"),
                        help="Directory holding the epoch-{n} LoRA adapters to render.")
    parser.add_argument("--learning-rate", type=float, default=6e-4,
                        help="Learning rate the adapters were trained with; used in the figure title.")
    parser.add_argument("--reuse-existing-images", action="store_true",
                        help="Re-render the figure from the images a previous run already wrote, instead of "
                             "generating them again, when every expected file is present and readable. Only "
                             "the presentation changes; the images themselves are identical.")
    parser.add_argument("--run-suffix", type=str, default="",
                        help="Appended to the generated-image directory and to every output filename, so "
                             "several hyperparameter variants of this grid can coexist on disk.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    suffix = args.run_suffix

    model_dir = Path(args.model_dir)
    epochs: List[int] = sorted(
        int(p.name.split("-")[1]) for p in model_dir.glob("epoch-*")
        if (p / "pytorch_lora_weights.safetensors").exists()
    )
    logger.info("epochs available: %s | seeds: %s", epochs, seeds)

    task: Any = args.task
    sel = json.loads((_OUT / f"selection_{task}.json").read_text(encoding="utf-8"))
    entities: List[Tuple[str, float]] = [(sel["target"]["name"], sel["target"]["self_clip_diff"])]
    entities += [(r["name"], r["clip_diff"]) for r in sel["receivers"]]
    # Generation order fixes each entity's position in the random-number sequence, so it must stay
    # exactly as it is across runs and across seeds; it is derived from the selection file only, never
    # from anything measured in this run. Column order is decided separately, per seed, after scoring.
    entities.sort(key=lambda x: x[1])
    hf_name_of = {n: get_target_overwrite(task, _METHOD, n)[0] for n, _ in entities}
    prompt_of = {n: f"An image of {hf_name_of[n]}" for n, _ in entities}
    target_index = next(i for i, (n, _) in enumerate(entities) if n == sel["target"]["name"])

    target_pre, target_over = get_target_overwrite(task, _METHOD, sel["target"]["name"])
    grid_dir = _OUT / f"epoch_grid{suffix}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    m = MetricImageTextSimilarity(metrics=["clip"])

    def load_arr(p: Path) -> "np.ndarray":
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.int64)

    def mean_abs(p: Path, q: Path) -> float:
        return float(np.mean(np.abs(load_arr(p) - load_arr(q))))

    def build_for_seed(seed: int) -> Dict[str, Any]:
        def off_path(bi: int) -> Path:
            return grid_dir / f"off_s{seed}_b{bi}.png"

        def on_path(bi: int, n: int) -> Path:
            return grid_dir / f"on_ep{n}_s{seed}_b{bi}.png"

        # one generate_dataset call per row over ALL entities in order -> entity bi always at RNG position bi
        def gen(lora: Any, paths: List[Path]) -> None:
            prompts = [prompt_of[entities[bi][0]] for bi in range(len(entities))]
            generate_dataset(
                model_base_name=_MODEL_ID, lora_name=lora, prompts=prompts, output_path=str(grid_dir),
                seeds=[seed], filenames=[p.name for p in paths], batch_size=1, lora_requires_inversion=False,
            )

        # The images are named by their POSITION in the generation order (b{index}), which is what makes
        # the seed match hold; that same naming means a different entity list would silently reuse another
        # run's images under new labels. The manifest records what the files on disk actually depict, so
        # reuse is only allowed when the current run agrees with it in every field.
        manifest_path = grid_dir / f"manifest_s{seed}.json"
        manifest = {
            "seed": seed, "task": task, "model_dir": str(model_dir), "epochs": epochs,
            "model_base_name": _MODEL_ID, "batch_size": 1, "method": _METHOD,
            "generation_order": [name for name, _ in entities],
            "prompts_in_generation_order": [prompt_of[name] for name, _ in entities],
        }

        wanted = ([off_path(bi) for bi in range(len(entities))]
                  + [on_path(bi, n) for n in epochs for bi in range(len(entities))])
        if args.reuse_existing_images and all(p.exists() for p in wanted):
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{manifest_path} is missing: these images were written before the manifest existed, so "
                    "what they depict cannot be confirmed. Re-render without --reuse-existing-images."
                )
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            differing = manifest_differences(stored, manifest)
            if differing:
                raise ValueError(
                    f"{manifest_path} describes a different run; fields that differ: {differing}. Reusing "
                    "these images would label one run's images with another run's entities."
                )
            for p in wanted:
                Image.open(p).convert("RGB").close()  # every reused file must be a readable image
            logger.info("[seed %d] reusing %d existing images; nothing is generated", seed, len(wanted))
        else:
            logger.info("[seed %d] generating baseline (off) row ...", seed)
            gen(None, [off_path(bi) for bi in range(len(entities))])
            for n in epochs:
                logger.info("[seed %d] generating epoch-%d (on) row ...", seed, n)
                gen(str(model_dir / f"epoch-{n}"), [on_path(bi, n) for bi in range(len(entities))])
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        rows = ["original"] + [f"epoch {n}" for n in epochs]
        cell: Dict[Tuple[int, int], Path] = {(0, bi): off_path(bi) for bi in range(len(entities))}
        for bi in range(len(entities)):
            for ri, n in enumerate(epochs, start=1):
                cell[(ri, bi)] = on_path(bi, n)

        def score(path: Path, entity: int) -> float:
            image = Image.open(path).convert("RGB")
            return float(m.score_batch_same_text([image], prompt_of[entities[entity][0]])[0]["clip"])

        clip_diff = score_cells(
            number_of_entities=len(entities), epochs=epochs,
            score_off=lambda entity: score(off_path(entity), entity),
            score_on=lambda entity, epoch: score(on_path(entity, epoch), entity),
        )

        nrows, ncols = len(rows), len(entities)

        # --- self-audit: row-to-row change over CONTROL entities (not strongly forgotten) ---
        controls = [bi for bi in range(ncols) if clip_diff[(nrows - 1, bi)] > _FORGOTTEN_CLIPDIFF]
        transitions = [
            statistics.mean(mean_abs(cell[(ri - 1, bi)], cell[(ri, bi)]) for bi in controls)
            for ri in range(1, nrows)
        ]
        audit = audit_transitions(transitions)
        logger.info("[seed %d] row-to-row change over %d controls: %s "
                    "(epoch-to-epoch baseline %.1f, original->epoch1 is %.2f x baseline)",
                    seed, len(controls), [round(t, 1) for t in transitions],
                    audit["epoch_to_epoch_baseline"], audit["reference_row_over_baseline"])
        if audit["outlier_epoch_transitions"]:
            logger.warning("[seed %d] AUDIT: epoch transition(s) %s exceed %.1f x the baseline",
                           seed, audit["outlier_epoch_transitions"], _OUTLIER_FACTOR)
        else:
            logger.info("[seed %d] AUDIT OK: no epoch transition is an outlier", seed)

        display_order = column_order(
            last_row={bi: clip_diff[(nrows - 1, bi)] for bi in range(ncols)},
            names=[name for name, _ in entities], target_index=target_index,
        )

        out_png = _OUT / f"epoch_grid{suffix}_seed{seed}.png"
        render_grid(
            cell=cell, clip_diff=clip_diff, row_labels=rows,
            column_labels=[_short_entity_display(hf_name_of[name]) for name, _ in entities],
            display_order=display_order, out_png=out_png,
            title=(f"Method: {_display_unlearning_algorithm(_METHOD).upper()} | "
                   f"Overwrite '{target_pre}' to '{target_over}' | "
                   f"seed={seed}, learning rate={args.learning_rate:g}\n"
                   f"rows = the original model then each saved epoch; columns = the target then the "
                   f"receivers sorted by clip_diff in the last epoch; cell = clip_diff against the "
                   f"original row, which is zero there and therefore not shown"),
        )

        result = {
            "seed": seed, "task": task, "epochs": epochs, "rows": rows,
            "learning_rate": args.learning_rate,
            "model_dir": str(model_dir),
            # entities and every clip_diff key are in GENERATION order, which is also the order the image
            # filenames are indexed by; display_column_order maps figure columns onto those indices
            "entities_by_interference": [{"name": n, "canonical_clip_diff": cd} for n, cd in entities],
            "display_column_order": display_order,
            "clip_diff": {f"{ri},{bi}": clip_diff[(ri, bi)] for ri in range(nrows) for bi in range(ncols)},
            "audit": dict(audit, control_entity_indices=controls),
        }
        (_OUT / f"epoch_grid{suffix}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("[seed %d] grid -> %s", seed, out_png)
        return result

    passed_all = True
    for seed in seeds:
        res = build_for_seed(seed)
        passed_all = passed_all and res["audit"]["passed"]
    print("GRID_OK seeds=%s all_audits_passed=%s" % (seeds, passed_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
