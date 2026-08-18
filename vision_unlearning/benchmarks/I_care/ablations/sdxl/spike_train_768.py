'''Can the SPARE training run at 768 pixels on this card, and what does a step cost?

The adapters were trained at 512 and, measured in `adapter_transfer.py`, they do not act at the 768
the images will be generated at. Retraining under the deployed configuration is therefore on the
table, and the first question is whether the machine can do it at all: training at 512 already
peaked at 10.837 GB of 11.98 (`assets/campaign_train_seed42.json`), and activations scale with the
latent area, so 768 is 2.25 times that unless something gives.

The something is gradient checkpointing, which the trainer already exposes: activations are
recomputed in the backward pass instead of being kept, trading roughly a third more time for a large
memory saving. This spike runs a handful of real training steps -- real splits, real checkpoint,
same hyperparameters as the campaign -- at 768 with checkpointing on, and reports peak video memory
and seconds per step. It writes no adapter anyone will use.

Three outcomes, all useful:
  * it fits          -> retraining at 768 is possible; the cost per seed follows from the step time.
  * it does not fit  -> retraining at 768 is out, and the choice is between training at 512 with the
                        DEPLOYED micro-conditioning declared, or generating at 512 and accepting
                        what section 3 of the validation report says about that.
  * it fits only just -> report the margin rather than the verdict.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python spike_train_768.py --resolution 768
'''
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from run_campaign import (_FORGET_WEIGHT, _LEARNING_RATE, _LORA_ALPHA, _LORA_R, _METHOD, _MODEL_ID,
                          _SPLIT_BASE, _TASK, _TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE,
                          _TRAIN_MICRO_CONDITIONING_TARGET_SIZE, _VAE_ID, _VARIANT,
                          _generation_order)
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_MODEL_DIR = _OUT / "spike_train_768_model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Feasibility of SPARE training above 512 pixels.")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--gradient-checkpointing", type=int, default=1)
    arguments = parser.parse_args()

    check_headroom()

    from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    from step_check_support import StopAfterTraining, restore_post_training, stub_post_training

    order = _generation_order()
    target_name = next(e["name"] for e in order if e["is_target"])
    target_overwrite = next(e["overwrite_concept"] for e in order if e["is_target"])
    split_base = _SPLIT_BASE / target_name
    assert (split_base / "train_forget").is_dir(), f"forget split missing under {split_base}"

    if _MODEL_DIR.exists():
        import shutil
        shutil.rmtree(_MODEL_DIR)

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(_MODEL_DIR),
        "hub_model_id": None,
        "final_eval_prompts_forget": [],
        "final_eval_prompts_retain": [],
        "model_name_or_path": _MODEL_ID,
        "pretrained_vae_model_name_or_path": _VAE_ID,
        "variant": _VARIANT,
        "dataset_forget_name": str(split_base / "train_forget"),
        "dataset_retain_name": str(split_base / "train_retain"),
        "validation_prompt": None,
        "dataloader_num_workers": 0,
        "resolution": arguments.resolution,
        # The campaign's own declarations, imported rather than retyped: the spike is only evidence
        # about the campaign if it trains under the same conditioning (D14).
        "micro_conditioning_original_size": _TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE,
        "micro_conditioning_target_size": _TRAIN_MICRO_CONDITIONING_TARGET_SIZE,
        "device": "cuda",
        "mixed_precision": "fp16",
        "learning_rate": _LEARNING_RATE,
        "max_grad_norm": 5.0,
        "num_train_epochs": 1,
        "max_train_steps": arguments.steps,
        "validation_epochs": 10_000,
        "checkpointing_steps": 100_000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "random_flip": True,
        "lora_r": _LORA_R,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": _LORA_ALPHA,
        "lora_dropout": 0.2,
        "seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": bool(arguments.gradient_checkpointing),
        "overwritting_concept": target_overwrite,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=_FORGET_WEIGHT, retain_weight=1.0),
        "compute_runtimes": False,
        "compute_memory": True,
    }

    label = f"{arguments.resolution}_checkpointing{arguments.gradient_checkpointing}"
    monitor = ResourceMonitor(_OUT / f"spike_train_{label}_monitor.log", interval_s=10.0)
    original_unlearn_lora = stub_post_training()
    t0 = time.time()
    failure = None
    try:
        unlearner = UnlearnerLoraDistillation(**hyperparameters)
        monitor.start()
        try:
            unlearner.train()
        except StopAfterTraining:
            pass
    except Exception as error:  # the point of a feasibility spike is to record how it fails
        failure = f"{type(error).__name__}: {error}"
    finally:
        monitor.stop()
        restore_post_training(original_unlearn_lora)
    seconds = round(time.time() - t0, 1)

    results = {
        "task": _TASK, "method": _METHOD, "target": target_name,
        "resolution": arguments.resolution,
        "steps_requested": arguments.steps,
        "gradient_checkpointing": bool(arguments.gradient_checkpointing),
        "micro_conditioning_original_size": list(_TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE),
        "micro_conditioning_target_size": list(_TRAIN_MICRO_CONDITIONING_TARGET_SIZE),
        "seconds_total": seconds,
        "seconds_per_step": round(seconds / arguments.steps, 2),
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": (round(monitor.min_ram_free_gb, 3)
                            if monitor.min_ram_free_gb != float("inf") else None),
        "failure": failure,
    }
    (_OUT / f"spike_train_{label}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"SPIKE_TRAIN_DONE resolution={arguments.resolution} failed={failure is not None}")


if __name__ == "__main__":
    main()
