"""Every-epoch x every-entity grid (the W6 deliverable format), for one task's selection.

Rows = [original base model, then each saved epoch]; columns = the 10 selected entities ordered by canonical
interference (target at column 0). Each cell is the entity's own concept prompt generated from that epoch's
adapter, at a fixed seed, batch_size=1 (dedicated-VRAM path). Runs for one or more seeds (--seeds).

Seed-matching invariant (critical): for a given seed, the baseline AND every epoch are generated within
THIS script from the SAME prompt list in the SAME order, so an entity sits at the same RNG position in every
call and starts from IDENTICAL initial noise in every row (generate_dataset reseeds to `seed` at the start
of each call and advances the generator once per prompt). Images are therefore NOT reused across scripts or
across different orderings (doing so puts an entity at a different RNG position and silently breaks the seed
match) - the reason the grid generates all its own cells with one ordering.

Self-audit (so a seed/reference mismatch can't pass silently): for each seed the script computes the
consecutive-row mean-abs pixel change over the CONTROL entities (those not strongly forgotten, which should
evolve smoothly) and flags any transition that is an outlier vs the median - in particular the
original->epoch1 transition, whose being an outlier is the fingerprint of a badly-generated reference row.

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
_OUTLIER_FACTOR = 2.0        # a consecutive transition > this x median (over controls) is flagged


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
    entities.sort(key=lambda x: x[1])  # most interfered (target) first
    prompt_of = {n: f"An image of {get_target_overwrite(task, _METHOD, n)[0]}" for n, _ in entities}

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

        logger.info("[seed %d] generating baseline (off) row ...", seed)
        gen(None, [off_path(bi) for bi in range(len(entities))])
        for n in epochs:
            logger.info("[seed %d] generating epoch-%d (on) row ...", seed, n)
            gen(str(model_dir / f"epoch-{n}"), [on_path(bi, n) for bi in range(len(entities))])

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

        # --- self-audit: consecutive-row change over CONTROL entities (not strongly forgotten) ---
        controls = [bi for bi in range(ncols) if clip_diff[(nrows - 1, bi)] > _FORGOTTEN_CLIPDIFF]
        transitions = []
        for ri in range(1, nrows):
            transitions.append(statistics.mean(mean_abs(cell[(ri - 1, bi)], cell[(ri, bi)]) for bi in controls))
        med = statistics.median(transitions)
        outliers = [ri for ri, t in enumerate(transitions, start=1) if med > 0 and t > _OUTLIER_FACTOR * med]
        ref_first_is_outlier = 1 in outliers
        logger.info("[seed %d] consecutive-row change over %d controls: %s (median %.1f)",
                    seed, len(controls), [round(t, 1) for t in transitions], med)
        if outliers:
            logger.warning("[seed %d] AUDIT: transition(s) %s are outliers (>%.1fx median); "
                           "original->epoch1 outlier=%s -> suspect a mismatched reference/seed/ordering row",
                           seed, outliers, _OUTLIER_FACTOR, ref_first_is_outlier)
        else:
            logger.info("[seed %d] AUDIT OK: no outlier transitions; reference row is consistent with the rest", seed)

        # --- figure ---
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.9 * nrows))
        for ri in range(nrows):
            for bi in range(ncols):
                ax = axes[ri][bi]
                ax.imshow(np.asarray(Image.open(cell[(ri, bi)]).convert("RGB")))
                ax.set_xticks([])
                ax.set_yticks([])
                if ri > 0:
                    ax.set_title(f"{clip_diff[(ri, bi)]:.0f}", fontsize=6)
                if bi == 0:
                    ax.set_ylabel(rows[ri], fontsize=8, rotation=0, ha="right", va="center")
        for bi, (name, cd) in enumerate(entities):
            axes[0][bi].set_title(f"{name.replace(' dog', '')}\ninterf {cd:.1f}", fontsize=6)
        audit_txt = "AUDIT OK" if not outliers else f"AUDIT WARNING: outlier transitions {outliers}"
        fig.suptitle(
            f"SPARE unlearning of '{target_pre}' -> '{target_over}', per epoch "
            f"(task {task}, seed {seed}, distil, learning rate {args.learning_rate:g})\n"
            f"rows = original + saved epochs; columns = 10 entities by canonical interference (target col 0); "
            f"cell = clip_diff vs original  |  {audit_txt}",
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
            "entities_by_interference": [{"name": n, "canonical_clip_diff": cd} for n, cd in entities],
            "clip_diff": {f"{ri},{bi}": clip_diff[(ri, bi)] for ri in range(nrows) for bi in range(ncols)},
            "audit": {
                "control_breed_indices": controls,
                "consecutive_row_change_over_controls": [round(t, 2) for t in transitions],
                "median_transition": round(med, 2),
                "outlier_transitions": outliers,
                "reference_to_first_is_outlier": ref_first_is_outlier,
                "passed": not outliers,
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
