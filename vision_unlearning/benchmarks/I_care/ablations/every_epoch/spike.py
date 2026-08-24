"""W3 ROCm feasibility spike for the every-epoch SPARE study (go/no-go for local training).

Runs the canonical ``distil`` (SPARE) unlearning for the chosen breeds target for a SMALL number of
epochs (default 2) with the new ``save_lora_at_epochs`` hook, then:

  * records the instantiated LoRA config's effective dropout (F7 - not changed, just observed);
  * validates each ``epoch-{n}`` adapter by safetensors CONTENT (non-empty LoRA keys), not existence;
  * checks final-equivalence: the last requested epoch's adapter equals the root final adapter tensorwise;
  * generates images with the epoch-{max} adapter and MEASURES fixed model-load time separately from
    marginal per-image time (F6), via two generate calls of different sizes;
  * runs the section-5 seed/baseline characterization (regenerate one stored baseline off-image and
    compare SSIM / max abs pixel diff) - report only, does not gate;
  * monitors CPU / RAM / VRAM throughout and HARD-EXITS before a RAM crash (a low-RAM abort is itself the
    plan's "cannot run locally" go/no-go signal).

Writes ``assets/spike_result.json``. Prints ``SPIKE_OK`` / ``SPIKE_GO_NO_GO_FAIL``. Paths resolve from
``__file__``; run with ``PYTHONPATH`` at the repo root, ``HF_TOKEN`` set, ``HF_HUB_DISABLE_XET=1``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_ICARE_ASSETS = _ICARE_DIR / "assets"
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TARGET = "bouvier des flandres dog"
_TASK: Literal["breeds"] = "breeds"
_METHOD: Literal["distil"] = "distil"
_RAM_ABORT_GB = 0.6  # hard floor: below this free RAM, kill the process before the OS thrashes/crashes


class ResourceMonitor:
    """Background 3-axis (CPU/RAM/VRAM) monitor with a hard RAM-abort floor."""

    def __init__(self, log_path: Path, interval_s: float = 20.0) -> None:
        import psutil
        import torch

        self._psutil = psutil
        self._torch = torch
        self._log_path = log_path
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_ram_used_gb = 0.0
        self.peak_vram_used_gb = 0.0
        self.min_ram_free_gb = float("inf")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> str:
        vm = self._psutil.virtual_memory()
        cpu = self._psutil.cpu_percent(interval=None)
        free_v, tot_v = self._torch.cuda.mem_get_info(0)
        used_v = (tot_v - free_v) / 1024 ** 3
        ram_free = vm.available / 1024 ** 3
        self.peak_ram_used_gb = max(self.peak_ram_used_gb, vm.used / 1024 ** 3)
        self.peak_vram_used_gb = max(self.peak_vram_used_gb, used_v)
        self.min_ram_free_gb = min(self.min_ram_free_gb, ram_free)
        line = f"[monitor] CPU {cpu:.0f}% | RAM {vm.percent:.0f}% ({ram_free:.2f}GB free) | VRAM {used_v:.2f}/{tot_v / 1024 ** 3:.2f}GB"
        if ram_free < _RAM_ABORT_GB:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + f"\n[monitor] FREE RAM {ram_free:.2f}GB < {_RAM_ABORT_GB}GB -> HARD ABORT to avoid a crash\n")
            os._exit(137)
        return line

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            line = self._sample()
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def _safetensors_lora_keys(path: Path) -> List[str]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:  # type: ignore[no-untyped-call]
        return list(f.keys())


def _max_abs_diff_between_adapters(a: Path, b: Path) -> float:
    from safetensors.torch import load_file
    import torch

    ta, tb = load_file(str(a)), load_file(str(b))
    if set(ta.keys()) != set(tb.keys()):
        return float("inf")
    m = 0.0
    for k in ta:
        m = max(m, torch.max(torch.abs(ta[k].float() - tb[k].float())).item())
    return m


def main() -> int:
    import torch
    from PIL import Image

    from vision_unlearning.unlearner import UnlearnerLoraDistillation
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple
    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("every_epoch_spike")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="W3 ROCm feasibility spike (local training go/no-go).")
    parser.add_argument("--epochs", type=int, default=2, help="Small epoch count for the spike (>=2).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="per_device_train_batch_size. Canonical 12GB config is 2; use 1 for the "
                             "low-VRAM fallback the pipeline_03 else-branch documents ('# 1').")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="gradient_accumulation_steps. Canonical is 2; use 4 with --batch-size 1 to "
                             "keep the effective batch size at 4 (numerically equivalent for SD's groupnorm unet).")
    args = parser.parse_args()
    n_epochs = args.epochs
    assert n_epochs >= 2

    _OUT.mkdir(parents=True, exist_ok=True)
    monitor_log = _OUT / "spike_monitor.log"
    monitor_log.write_text("", encoding="utf-8")
    result: Dict[str, Any] = {
        "task": _TASK, "method": _METHOD, "target": _TARGET, "epochs": n_epochs,
        "per_device_train_batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum, "dataloader_num_workers": 0,
    }

    # target strings + canonical eval prompts (breeds branch of pipeline_03)
    metadata = json.loads((_ICARE_ASSETS / f"metadata_{_TASK}_2_enriched_filtered.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in metadata]
    idx = names.index(_TARGET)
    target_pre, target_over = get_target_overwrite(_TASK, _METHOD, _TARGET)
    target_pre_next, _ = get_target_overwrite(_TASK, _METHOD, names[(idx + 1) % 100])
    logger.info("target_preprocessed=%r overwrite=%r (idx %d)", target_pre, target_over, idx)

    example_prompts_forget = [
        f"An image of {target_pre}",
        f'Photograph of {target_pre.replace("_", " ")}; high definition',
        f"An picture of {target_pre} in the rain",
        f"An picture of {target_pre} running",
    ]
    example_prompts_retain = [
        f"An image of {target_pre_next}",
        f"Photograph of {target_pre_next}; high definition",
        f"An picture of {target_pre_next} in the rain",
        f"An picture of {target_pre_next} running",
    ]

    split_base = _ICARE_ASSETS / "datasets" / "taras_breeds_splits_filtered" / _TARGET
    forget_dir = split_base / "train_forget"
    retain_dir = split_base / "train_retain"
    model_dir = _OUT / "models" / f"{_TASK}_bouvier_spike_{_METHOD}_{n_epochs}"
    if model_dir.exists():
        shutil.rmtree(model_dir)  # fresh spike; adapter snapshots are not resumable checkpoints

    free_b, _ = torch.cuda.mem_get_info(0)
    # canonical distil batch config for 12GB (< 14e9 free branch of pipeline_03)
    hyperparameters: Dict[str, Any] = {
        "output_dir": str(model_dir),
        "hub_model_id": None,
        "final_eval_prompts_forget": example_prompts_forget,
        "final_eval_prompts_retain": example_prompts_retain,
        "model_name_or_path": _MODEL_ID,
        "dataset_forget_name": str(forget_dir),
        "dataset_retain_name": str(retain_dir),
        "validation_prompt": f"An image of {target_pre}",
        # Canonical uses 2; forced to 0 on Windows because DataLoader workers use `spawn` (not Linux's
        # `fork`) and must pickle the dataset transform, a local closure that cannot be pickled
        # (AttributeError on _prepare_dataloaders.<locals>.preprocess_forget). num_workers=0 loads in
        # the main process -> no pickling; it does not change the training result (same data, order, seed),
        # only removes data-loading parallelism (negligible for the 35-image forget set).
        "dataloader_num_workers": 0,
        "resolution": 512,
        "num_validation_images": 1,
        "mixed_precision": "no",
        "learning_rate": 6e-4,
        "max_grad_norm": 5.0,
        "num_train_epochs": n_epochs,
        "validation_epochs": n_epochs + 1,
        "checkpointing_steps": 10000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "random_flip": True,
        "lora_r": 16,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": 4,
        "lora_dropout": 0.2,
        "seed": 42,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "overwritting_concept": target_over,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        "save_lora_at_epochs": list(range(1, n_epochs + 1)),
    }

    unlearner = UnlearnerLoraDistillation(**hyperparameters)
    effective_dropout = unlearner._get_lora_config().lora_dropout  # F7: observe, do not change
    result["effective_lora_dropout"] = effective_dropout
    logger.info("F7 effective LoRA dropout (LoraConfig) = %s (passed lora_dropout=0.2 is ignored)", effective_dropout)

    monitor = ResourceMonitor(monitor_log)
    monitor.start()
    try:
        t0 = time.time()
        eval_results = unlearner.train()
        train_s = time.time() - t0
    finally:
        monitor.stop()
    result["train_seconds_total"] = round(train_s, 1)
    result["train_seconds_per_epoch"] = round(train_s / n_epochs, 1)
    result["eval_results"] = {r.metric_name: r.metric_value for r in eval_results}
    logger.info("Training done in %.1fs (%.1fs/epoch)", train_s, train_s / n_epochs)

    del unlearner
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # --- content validation of each epoch adapter + final-equivalence ---------------------------
    epoch_checks: Dict[str, Any] = {}
    for n in range(1, n_epochs + 1):
        p = model_dir / f"epoch-{n}" / "pytorch_lora_weights.safetensors"
        keys = _safetensors_lora_keys(p) if p.exists() else []
        epoch_checks[f"epoch-{n}"] = {"exists": p.exists(), "n_lora_keys": len(keys)}
    root_final = model_dir / "pytorch_lora_weights.safetensors"
    last_epoch = model_dir / f"epoch-{n_epochs}" / "pytorch_lora_weights.safetensors"
    final_equiv_maxabs = (
        _max_abs_diff_between_adapters(root_final, last_epoch)
        if root_final.exists() and last_epoch.exists() else float("inf")
    )
    result["epoch_adapter_checks"] = epoch_checks
    result["final_equivalence_max_abs_diff"] = final_equiv_maxabs
    all_epochs_valid = all(v["exists"] and v["n_lora_keys"] > 0 for v in epoch_checks.values())
    final_equiv_ok = final_equiv_maxabs == 0.0
    logger.info("epoch adapters valid=%s | final-equivalence max|diff|=%s", all_epochs_valid, final_equiv_maxabs)

    # --- generation with the epoch-{max} adapter + F6 load-vs-marginal timing --------------------
    gen_dir = _OUT / "spike_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    epoch_adapter = str(model_dir / f"epoch-{n_epochs}")
    prompts_all = [f"An image of {get_target_overwrite(_TASK, _METHOD, m['name'])[0]}" for m in metadata]

    def _gen(prompts: List[str], tag: str) -> float:
        filenames = [f"on_42_{tag}_{i}.png" for i, _ in enumerate(prompts)]
        t = time.time()
        generate_dataset(
            model_base_name=_MODEL_ID,
            lora_name=epoch_adapter,
            prompts=prompts,
            output_path=str(gen_dir),
            seeds=[42],
            filenames=filenames,
            batch_size=25,
            lora_requires_inversion=False,
        )
        return time.time() - t

    t_small = _gen(prompts_all[:2], "small")   # 2 images = load + 2*marginal
    t_big = _gen(prompts_all[:20], "big")      # 20 images = load + 20*marginal
    marginal_s = (t_big - t_small) / 18.0
    load_s = t_small - 2 * marginal_s
    result["gen_timing"] = {
        "call_2img_s": round(t_small, 1),
        "call_20img_s": round(t_big, 1),
        "fixed_load_s": round(load_s, 1),
        "marginal_per_image_s": round(marginal_s, 2),
    }
    logger.info("F6 timing: fixed load=%.1fs, marginal/image=%.2fs", load_s, marginal_s)

    gen_imgs = sorted(gen_dir.glob("on_42_*.png"))
    rgb_ok = 0
    for p in gen_imgs[:4]:
        with Image.open(p) as im:
            im.convert("RGB").load()
            rgb_ok += 1
    result["generated_images"] = {"count": len(gen_imgs), "sample_rgb_ok": rgb_ok}

    # --- section 5 seed/baseline characterization (report only, does not gate) -------------------
    baseline_char: Dict[str, Any] = {"attempted": False}
    baseline_dir = _ICARE_ASSETS / "datasets" / f"generated_{_TASK}_baseline"
    off_imgs = sorted(baseline_dir.glob("off_42_*.png")) if baseline_dir.exists() else []
    if off_imgs:
        stored = off_imgs[0]
        prompt = stored.name[len("off_42_"):-len(".png")]
        chk_dir = _OUT / "spike_baseline_check"
        chk_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset(
            model_base_name=_MODEL_ID,
            lora_name=None,
            prompts=[prompt],
            output_path=str(chk_dir),
            seeds=[42],
            filenames=[stored.name],
            batch_size=1,
            lora_requires_inversion=False,
        )
        regen = chk_dir / stored.name
        import numpy as np
        a = np.asarray(Image.open(stored).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(regen).convert("RGB"), dtype=np.float64)
        max_abs = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        ssim_val: Optional[float] = None
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_val = float(ssim(a, b, channel_axis=2, data_range=255.0))
        except Exception as exc:  # noqa: BLE001
            logger.info("SSIM unavailable (%s); reporting max abs pixel diff only", type(exc).__name__)
        baseline_char = {
            "attempted": True, "prompt": prompt, "max_abs_pixel_diff": max_abs,
            "ssim": ssim_val, "reproduces_pixel_identically": max_abs == 0.0,
        }
        logger.info("baseline characterization: max|diff|=%s ssim=%s", max_abs, ssim_val)
    result["baseline_characterization"] = baseline_char

    result["resources"] = {
        "peak_ram_used_gb": round(monitor.peak_ram_used_gb, 2),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 2),
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 2),
    }

    go = all_epochs_valid and final_equiv_ok and rgb_ok > 0 and len(gen_imgs) >= 20
    result["go_no_go"] = "proceed" if go else "stop"
    (_OUT / "spike_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("spike_result.json written")

    if go:
        print("SPIKE_OK", json.dumps(result["gen_timing"]))
        return 0
    print("SPIKE_GO_NO_GO_FAIL", json.dumps({k: result[k] for k in ("epoch_adapter_checks", "final_equivalence_max_abs_diff", "generated_images")}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
