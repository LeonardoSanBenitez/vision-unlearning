"""Every-epoch degradation demonstration for the breeds target (report artifact).

Trains the canonical ``distil`` (SPARE) unlearning for 'a bouvier des flandres dog' with the
``save_lora_at_epochs`` hook, saving several intermediate LoRA adapters, then generates the TARGET
concept prompt at each saved epoch (plus the base-model baseline), computes ``clip_diff`` per epoch,
and assembles a single strip figure showing the concept degrade across epochs. This is the end-to-end
proof that the whole per-epoch machinery works: hook -> intermediate adapters -> generation -> metric.

Uses the three local adaptations validated by the spike (num_workers=0, batch-1 x accum-4, small
inference batch) and the same 3-axis CPU/RAM/VRAM monitor with a hard RAM-abort floor. Paths resolve
from __file__; run with PYTHONPATH at the repo root, HF_TOKEN set, HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_ICARE_ASSETS = _ICARE_DIR / "assets"
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TARGET = "bouvier des flandres dog"
_TASK: Literal["breeds"] = "breeds"
_METHOD: Literal["distil"] = "distil"
_RAM_ABORT_GB = 0.6


class ResourceMonitor:
    def __init__(self, log_path: Path, interval_s: float = 30.0) -> None:
        import psutil
        import torch
        self._psutil = psutil
        self._torch = torch
        self._log_path = log_path
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_vram_used_gb = 0.0
        self.min_ram_free_gb = float("inf")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            vm = self._psutil.virtual_memory()
            cpu = self._psutil.cpu_percent(interval=None)
            free_v, tot_v = self._torch.cuda.mem_get_info(0)
            used_v = (tot_v - free_v) / 1024 ** 3
            ram_free = vm.available / 1024 ** 3
            self.peak_vram_used_gb = max(self.peak_vram_used_gb, used_v)
            self.min_ram_free_gb = min(self.min_ram_free_gb, ram_free)
            line = f"[monitor] CPU {cpu:.0f}% | RAM {vm.percent:.0f}% ({ram_free:.2f}GB free) | VRAM {used_v:.2f}/{tot_v / 1024 ** 3:.2f}GB"
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if ram_free < _RAM_ABORT_GB:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[monitor] FREE RAM {ram_free:.2f}GB < {_RAM_ABORT_GB}GB -> HARD ABORT\n")
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

    from vision_unlearning.unlearner import UnlearnerLoraDistillation
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple
    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.metrics import MetricImageTextSimilarity
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("every_epoch_demo")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Every-epoch degradation demo (breeds target).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoints", type=str, default="1,2,3,5,10,20,30")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n_epochs = args.epochs
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",")})
    assert max(checkpoints) <= n_epochs

    _OUT.mkdir(parents=True, exist_ok=True)
    monitor_log = _OUT / "demo_monitor.log"
    monitor_log.write_text("", encoding="utf-8")

    target_pre, target_over = get_target_overwrite(_TASK, _METHOD, _TARGET)
    metadata = json.loads((_ICARE_ASSETS / f"metadata_{_TASK}_2_enriched_filtered.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in metadata]
    idx = names.index(_TARGET)
    target_pre_next, _ = get_target_overwrite(_TASK, _METHOD, names[(idx + 1) % 100])
    prompt = f"An image of {target_pre}"
    logger.info("target=%r prompt=%r overwrite=%r epochs=%d checkpoints=%s", target_pre, prompt, target_over, n_epochs, checkpoints)

    split_base = _ICARE_ASSETS / "datasets" / "taras_breeds_splits_filtered" / _TARGET
    model_dir = _OUT / "models" / f"{_TASK}_bouvier_demo_{_METHOD}_{n_epochs}"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(model_dir), "hub_model_id": None,
        "final_eval_prompts_forget": [prompt], "final_eval_prompts_retain": [f"An image of {target_pre_next}"],
        "model_name_or_path": _MODEL_ID,
        "dataset_forget_name": str(split_base / "train_forget"),
        "dataset_retain_name": str(split_base / "train_retain"),
        "validation_prompt": prompt,
        "dataloader_num_workers": 0, "resolution": 512, "num_validation_images": 1,
        "mixed_precision": "no", "learning_rate": 6e-4, "max_grad_norm": 5.0,
        "num_train_epochs": n_epochs, "validation_epochs": n_epochs + 1,
        "checkpointing_steps": 100000, "lr_scheduler_type": "constant", "lr_warmup_steps": 0,
        "save_strategy": "epoch", "save_total_limit": 2, "random_flip": True,
        "lora_r": 16, "target_modules": ["to_k", "to_q", "to_v", "to_out.0"], "lora_alpha": 4,
        "lora_dropout": 0.2, "seed": 42,
        "per_device_train_batch_size": 1, "train_batch_size": 1, "gradient_accumulation_steps": 4,
        "overwritting_concept": target_over,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        "save_lora_at_epochs": checkpoints,
    }

    monitor = ResourceMonitor(monitor_log)
    monitor.start()
    try:
        unlearner = UnlearnerLoraDistillation(**hyperparameters)
        t0 = time.time()
        unlearner.train()
        train_s = time.time() - t0
        del unlearner
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("training done in %.0fs (%.1f min)", train_s, train_s / 60)

        # generate the target concept at each checkpoint + baseline (base model)
        gen_dir = _OUT / "demo_generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        def gen(lora: Any, fname: str) -> Path:
            generate_dataset(
                model_base_name=_MODEL_ID, lora_name=lora, prompts=[prompt],
                output_path=str(gen_dir), seeds=[args.seed], filenames=[fname],
                batch_size=2, lora_requires_inversion=False,
            )
            return gen_dir / fname

        off_path = gen(None, f"off_seed{args.seed}.png")
        on_paths: Dict[int, Path] = {}
        for n in checkpoints:
            on_paths[n] = gen(str(model_dir / f"epoch-{n}"), f"on_epoch{n}_seed{args.seed}.png")
    finally:
        monitor.stop()

    # clip_diff per epoch (clip_on - clip_off), the same metric primitive pipeline_06 uses
    m = MetricImageTextSimilarity(metrics=["clip"])
    off_img = Image.open(off_path).convert("RGB")
    clip_off = m.score_batch_same_text([off_img], prompt)[0]["clip"]
    traj: List[Dict[str, Any]] = []
    for n in checkpoints:
        on_img = Image.open(on_paths[n]).convert("RGB")
        clip_on = m.score_batch_same_text([on_img], prompt)[0]["clip"]
        traj.append({"epoch": n, "clip_on": clip_on, "clip_off": clip_off, "clip_diff": clip_on - clip_off})
        logger.info("epoch %d: clip_on=%.3f clip_off=%.3f clip_diff=%.3f", n, clip_on, clip_off, clip_on - clip_off)

    result = {
        "task": _TASK, "method": _METHOD, "target": target_pre, "overwrite": target_over,
        "prompt": prompt, "seed": args.seed, "epochs": n_epochs, "checkpoints": checkpoints,
        "train_seconds": round(train_s, 1), "trajectory": traj,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 2),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 2),
        "off_image": str(off_path), "on_images": {n: str(p) for n, p in on_paths.items()},
    }
    (_OUT / "demo_trajectory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # strip figure: baseline off | each epoch, titled with clip_diff
    cols = 1 + len(checkpoints)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.6))
    axes[0].imshow(np.asarray(Image.open(off_path).convert("RGB")))
    axes[0].set_title("base model (off)\n(no unlearning)", fontsize=9)
    axes[0].axis("off")
    for i, n in enumerate(checkpoints, start=1):
        cd = next(t["clip_diff"] for t in traj if t["epoch"] == n)
        axes[i].imshow(np.asarray(Image.open(on_paths[n]).convert("RGB")))
        axes[i].set_title(f"epoch {n}\nclip_diff {cd:.2f}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(
        f"SPARE unlearning of '{target_pre}' -> '{target_over}', per epoch (seed {args.seed})\n"
        f"prompt: {prompt}  |  clip_diff = clip_on - clip_off (more negative = more forgotten)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    strip = _OUT / "demo_trajectory_strip.png"
    fig.savefig(strip, dpi=110)
    plt.close(fig)
    logger.info("strip figure -> %s", strip)
    print("DEMO_OK", str(strip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
