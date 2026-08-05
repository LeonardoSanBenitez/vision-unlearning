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
from typing import Any, Dict, List, Literal, Tuple

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_METHOD: Literal["distil"] = "distil"
_FORGOTTEN_CLIPDIFF = -5.0   # an entity with last-epoch clip_diff below this is "being forgotten" (not a control)
_OUTLIER_FACTOR = 2.0        # an epoch-to-epoch transition > this x the baseline (over controls) is flagged


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
            differing = sorted(k for k in manifest if stored.get(k) != manifest[k])
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
        cell: Dict[Tuple[int, int], Path] = {}
        clip_diff: Dict[Tuple[int, int], float] = {}
        for bi, (name, _) in enumerate(entities):
            off = Image.open(off_path(bi)).convert("RGB")
            clip_off = m.score_batch_same_text([off], prompt_of[name])[0]["clip"]
            cell[(0, bi)] = off_path(bi)
            clip_diff[(0, bi)] = 0.0
            for ri, n in enumerate(epochs, start=1):
                cell[(ri, bi)] = on_path(bi, n)
                on = Image.open(on_path(bi, n)).convert("RGB")
                clip_diff[(ri, bi)] = m.score_batch_same_text([on], prompt_of[name])[0]["clip"] - clip_off

        nrows, ncols = len(rows), len(entities)

        # --- self-audit: row-to-row change over CONTROL entities (not strongly forgotten) ---
        # The baseline is the median of the EPOCH-to-epoch transitions only. It therefore never involves
        # the reference row, so a badly generated reference row cannot raise the bar it is judged against.
        controls = [bi for bi in range(ncols) if clip_diff[(nrows - 1, bi)] > _FORGOTTEN_CLIPDIFF]
        transitions = []
        for ri in range(1, nrows):
            transitions.append(statistics.mean(mean_abs(cell[(ri - 1, bi)], cell[(ri, bi)]) for bi in controls))
        baseline = statistics.median(transitions[1:])
        outliers = [ri for ri, t in enumerate(transitions[1:], start=2) if baseline > 0
                    and t > _OUTLIER_FACTOR * baseline]
        # Reported, deliberately NOT a pass/fail criterion. The original->epoch-1 change is inflated by two
        # different causes that a single run cannot tell apart: a mismatched reference row, and the fact
        # that this is the only transition where the adapter appears at all (which dominates when later
        # epochs move little). Measured: 1.07 and 1.25 on runs whose reference row is known good, but also
        # 1.71 and 1.88 on runs with small epoch-to-epoch movement, against 2.09 for a deliberately
        # mismatched reference row - overlapping ranges, so no threshold on this ratio is trustworthy.
        reference_ratio = (transitions[0] / baseline) if baseline > 0 else 0.0
        logger.info("[seed %d] row-to-row change over %d controls: %s "
                    "(epoch-to-epoch baseline %.1f, original->epoch1 is %.2f x baseline)",
                    seed, len(controls), [round(t, 1) for t in transitions], baseline, reference_ratio)
        if outliers:
            logger.warning("[seed %d] AUDIT: epoch transition(s) %s exceed %.1f x the baseline",
                           seed, outliers, _OUTLIER_FACTOR)
        else:
            logger.info("[seed %d] AUDIT OK: no epoch transition is an outlier", seed)
        passed = not outliers

        # --- column order: the target, then the receivers by their clip_diff in the last rendered epoch ---
        # (most negative = most interfered = nearest the target). Computed per seed from this run's own
        # numbers, so the bottom row of the figure is visibly the sort key. This is a presentation order
        # only; the images were generated in the fixed order above.
        receivers_by_last_epoch = sorted(
            (bi for bi in range(ncols) if bi != target_index),
            key=lambda bi: (clip_diff[(nrows - 1, bi)], entities[bi][0]),
        )
        display_order: List[int] = [target_index] + receivers_by_last_epoch

        # --- figure ---
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.9 * nrows))
        for ri in range(nrows):
            for ci, bi in enumerate(display_order):
                ax = axes[ri][ci]
                ax.imshow(np.asarray(Image.open(cell[(ri, bi)]).convert("RGB")))
                ax.set_xticks([])
                ax.set_yticks([])
                if ri > 0:
                    ax.set_title(f"clip_diff={clip_diff[(ri, bi)]:.2f}", fontsize=6)
                if ci == 0:
                    ax.set_ylabel(rows[ri], fontsize=8, rotation=0, ha="right", va="center")
        for ci, bi in enumerate(display_order):
            label = _short_entity_display(hf_name_of[entities[bi][0]])
            axes[0][ci].set_title(f"{label} (target)" if ci == 0 else label, fontsize=6)
        fig.suptitle(
            f"Method: {_display_unlearning_algorithm(_METHOD).upper()} | "
            f"Overwrite '{target_pre}' to '{target_over}' | "
            f"seed={seed}, learning rate={args.learning_rate:g}\n"
            f"rows = the original model then each saved epoch; columns = the target then the receivers "
            f"sorted by clip_diff in the last epoch; cell = clip_diff against the original row, "
            f"which is zero there and therefore not shown",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = _OUT / f"epoch_grid{suffix}_seed{seed}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)

        result = {
            "seed": seed, "task": task, "epochs": epochs, "rows": rows,
            "learning_rate": args.learning_rate,
            "model_dir": str(model_dir),
            # entities and every clip_diff key are in GENERATION order, which is also the order the image
            # filenames are indexed by; display_column_order maps figure columns onto those indices
            "entities_by_interference": [{"name": n, "canonical_clip_diff": cd} for n, cd in entities],
            "display_column_order": display_order,
            "clip_diff": {f"{ri},{bi}": clip_diff[(ri, bi)] for ri in range(nrows) for bi in range(ncols)},
            "audit": {
                "control_entity_indices": controls,
                "row_to_row_change_over_controls": [round(t, 2) for t in transitions],
                "epoch_to_epoch_baseline": round(baseline, 2),
                "reference_row_over_baseline": round(reference_ratio, 3),
                "outlier_epoch_transitions": outliers,
                "passed": passed,
            },
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
