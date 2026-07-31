"""Seed / selectivity validation for the every-epoch SPARE generation (report artifact).

Answers three questions the reviewer raised about whether the seed is properly set and the unlearning is
selective, for all 10 selected breeds (target + 9 receivers), across several prompt templates:

  (A) Determinism: does the SAME (prompt, seed) regenerate the SAME image across two separate
      generate_dataset calls? (max abs pixel diff should be ~0 if the seed truly fixes the noise.)
  (B) Seed sensitivity: does seed 42 vs seed 43 produce DIFFERENT images? (proves the seed is used.)
  (C) Selectivity: with one fixed seed, base model (off) vs the unlearned model (on, an epoch adapter),
      how much does each breed's image change? Low/no-interference breeds should barely change (same pose
      and background); the target and high-interference breeds should change a lot. If low-interference
      breeds change a lot, something is wrong (seed not fixing noise, or non-selective corruption).

Generation-only (fp16), so fast and low-memory. Writes seed_validation.json and per-template base-vs-on
grids + an interference-vs-image-change scatter. Paths resolve from __file__; run with PYTHONPATH at the
repo root, HF_TOKEN set, HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_ICARE_ASSETS = _ICARE_DIR / "assets"
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TASK: Literal["breeds"] = "breeds"
_METHOD: Literal["distil"] = "distil"

# Prompt templates: vary context (pose / background / framing), not just the breed name.
_TEMPLATES = [
    "An image of {b}",
    "A photograph of {b} running on grass, high definition",
    "{b} sitting on a wooden floor indoors",
]


class ResourceMonitor:
    def __init__(self, log_path: Path, interval_s: float = 30.0) -> None:
        import psutil
        import torch
        self._psutil, self._torch = psutil, torch
        self._log_path, self._interval = log_path, interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            vm = self._psutil.virtual_memory()
            free_v, tot_v = self._torch.cuda.mem_get_info(0)
            ram_free = vm.available / 1024 ** 3
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[monitor] CPU {self._psutil.cpu_percent(interval=None):.0f}% | RAM {vm.percent:.0f}% ({ram_free:.2f}GB free) | VRAM {(tot_v - free_v) / 1024 ** 3:.2f}/{tot_v / 1024 ** 3:.2f}GB\n")
            if ram_free < 0.6:
                os._exit(137)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.metrics import MetricImageTextSimilarity
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("seed_validation")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Seed / selectivity validation (breeds, generation-only).")
    parser.add_argument("--adapter", default=str(_OUT / "models" / "breeds_bouvier_demo_distil_30" / "epoch-10"),
                        help="Unlearned LoRA adapter dir to use as 'after' (default: demo epoch-10).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alt-seed", type=int, default=43)
    args = parser.parse_args()
    adapter = args.adapter

    sel = json.loads((_OUT / "selection_breeds.json").read_text(encoding="utf-8"))
    breeds: List[Tuple[str, float]] = [(sel["target"]["name"], sel["target"]["self_clip_diff"])]
    breeds += [(r["name"], r["clip_diff"]) for r in sel["receivers"]]
    # order by interference (most negative first) for the grids
    breeds.sort(key=lambda x: x[1])
    prompt_of = {name: f"An image of {get_target_overwrite(_TASK, _METHOD, name)[0]}" for name, _ in breeds}
    breed_pre = {name: get_target_overwrite(_TASK, _METHOD, name)[0] for name, _ in breeds}
    logger.info("breeds (by interference): %s", [f"{n}:{cd:.1f}" for n, cd in breeds])

    gen_dir = _OUT / "seed_validation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(_OUT / "seed_validation_monitor.log")
    monitor.start()

    def gen(prompts: List[str], filenames: List[str], seed: int, lora: Any) -> None:
        # batch_size=1: on this ROCm GPU, generate_dataset at batch_size>1 with deterministic
        # algorithms falls into a slow shared-memory kernel path (~10x). Batch 1 keeps generation on
        # dedicated VRAM (~18s/image). Same images, one at a time.
        generate_dataset(
            model_base_name=_MODEL_ID, lora_name=lora, prompts=prompts,
            output_path=str(gen_dir), seeds=[seed], filenames=filenames,
            batch_size=1, lora_requires_inversion=False,
        )

    def load(fname: str) -> "np.ndarray":
        return np.asarray(Image.open(gen_dir / fname).convert("RGB"), dtype=np.float64)

    # build the full (breed x template) prompt list
    keys = [(bi, ti) for bi in range(len(breeds)) for ti in range(len(_TEMPLATES))]
    all_prompts = [_TEMPLATES[ti].format(b=breed_pre[breeds[bi][0]]) for bi, ti in keys]
    base_names = [f"base_s{args.seed}_b{bi}_t{ti}.png" for bi, ti in keys]
    on_names = [f"on_s{args.seed}_b{bi}_t{ti}.png" for bi, ti in keys]

    logger.info("generating base (off) seed %d ...", args.seed)
    gen(all_prompts, base_names, args.seed, None)
    logger.info("generating unlearned (on) seed %d from %s ...", args.seed, adapter)
    gen(all_prompts, on_names, args.seed, adapter)

    # (A) determinism + (B) seed sensitivity, on template 0 for all breeds
    t0_prompts = [_TEMPLATES[0].format(b=breed_pre[n]) for n, _ in breeds]
    rep_names = [f"base_s{args.seed}_rep_b{bi}.png" for bi in range(len(breeds))]
    alt_names = [f"base_s{args.alt_seed}_b{bi}.png" for bi in range(len(breeds))]
    logger.info("determinism: regenerating base seed %d (separate call) ...", args.seed)
    gen(t0_prompts, rep_names, args.seed, None)
    logger.info("seed sensitivity: base seed %d ...", args.alt_seed)
    gen(t0_prompts, alt_names, args.alt_seed, None)
    monitor.stop()

    m = MetricImageTextSimilarity(metrics=["clip"])

    # (A)/(B) metrics
    determinism: List[Dict[str, Any]] = []
    seed_sensitivity: List[Dict[str, Any]] = []
    for bi, (name, cd) in enumerate(breeds):
        a = load(f"base_s{args.seed}_b{bi}_t0.png")
        rep = load(rep_names[bi])
        alt = load(alt_names[bi])
        determinism.append({"breed": name, "max_abs_pixel_diff": float(np.max(np.abs(a - rep))), "mean_abs": float(np.mean(np.abs(a - rep)))})
        seed_sensitivity.append({"breed": name, "mean_abs_diff_seed42_vs_43": float(np.mean(np.abs(a - alt)))})

    # (C) selectivity metrics per (breed, template)
    selectivity: List[Dict[str, Any]] = []
    for bi, (name, cd) in enumerate(breeds):
        for ti in range(len(_TEMPLATES)):
            off = load(f"base_s{args.seed}_b{bi}_t{ti}.png")
            on = load(f"on_s{args.seed}_b{bi}_t{ti}.png")
            prompt = _TEMPLATES[ti].format(b=breed_pre[name])
            clip_on = m.score_batch_same_text([Image.fromarray(on.astype("uint8"))], prompt)[0]["clip"]
            clip_off = m.score_batch_same_text([Image.fromarray(off.astype("uint8"))], prompt)[0]["clip"]
            selectivity.append({
                "breed": name, "template_idx": ti, "canonical_clip_diff": cd,
                "clip_diff_on_minus_off": clip_on - clip_off,
                "base_vs_on_mean_abs_pixel": float(np.mean(np.abs(off - on))),
                "base_vs_on_ssim": float(ssim(off, on, channel_axis=2, data_range=255.0)),
            })

    result = {
        "adapter": adapter, "seed": args.seed, "alt_seed": args.alt_seed, "templates": _TEMPLATES,
        "breeds_by_interference": [{"name": n, "canonical_clip_diff": cd} for n, cd in breeds],
        "A_determinism": determinism,
        "A_determinism_max_over_breeds": max(float(d["max_abs_pixel_diff"]) for d in determinism),
        "B_seed_sensitivity": seed_sensitivity,
        "C_selectivity": selectivity,
    }
    (_OUT / "seed_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("determinism max abs pixel diff over all breeds = %.3f", result["A_determinism_max_over_breeds"])

    # figures: one base|on grid per template (rows = breeds by interference)
    for ti in range(len(_TEMPLATES)):
        fig, axes = plt.subplots(len(breeds), 2, figsize=(6, 2.4 * len(breeds)))
        for bi, (name, cd) in enumerate(breeds):
            off_img = Image.open(gen_dir / f"base_s{args.seed}_b{bi}_t{ti}.png").convert("RGB")
            on_img = Image.open(gen_dir / f"on_s{args.seed}_b{bi}_t{ti}.png").convert("RGB")
            rec = next(r for r in selectivity if r["breed"] == name and r["template_idx"] == ti)
            axes[bi][0].imshow(np.asarray(off_img))
            axes[bi][1].imshow(np.asarray(on_img))
            axes[bi][1].axis("off")
            axes[bi][0].set_xticks([])
            axes[bi][0].set_yticks([])
            axes[bi][0].set_ylabel(f"{name}\ninterf {cd:.1f}", fontsize=7, rotation=0, ha="right", va="center")
            axes[bi][1].set_title(f"SSIM {rec['base_vs_on_ssim']:.2f}", fontsize=7)
        axes[0][0].set_title("base (off)", fontsize=9)
        fig.suptitle(f"base vs unlearned (seed {args.seed}) | template: {_TEMPLATES[ti]}\nrows by canonical interference; low-interference breeds should barely change", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(_OUT / f"seed_validation_grid_t{ti}.png", dpi=100)
        plt.close(fig)

    # scatter: canonical interference vs measured base-vs-on change (mean over templates)
    by_breed: Dict[str, List[float]] = {}
    for r in selectivity:
        by_breed.setdefault(r["breed"], []).append(r["base_vs_on_mean_abs_pixel"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, cd in breeds:
        y = float(np.mean(by_breed[name]))
        ax.scatter(cd, y, s=40)
        ax.annotate(name.replace(" dog", ""), (cd, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("canonical interference clip_diff (more negative = more interfered)")
    ax.set_ylabel("base vs unlearned mean abs pixel change")
    ax.set_title("Selectivity: does image change track interference?\n(low-interference breeds bottom-right should have low change)")
    fig.tight_layout()
    fig.savefig(_OUT / "seed_validation_scatter.png", dpi=110)
    plt.close(fig)

    print("SEEDVAL_OK", "determinism_max_abs=%.3f" % result["A_determinism_max_over_breeds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
