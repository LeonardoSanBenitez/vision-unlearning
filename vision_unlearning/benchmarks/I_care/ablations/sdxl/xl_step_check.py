'''
Does one SPARE training step run on a Stable Diffusion XL checkpoint, through the library trainer,
inside this machine's video memory?

Two questions, and they are separate:

1. **Does the Stable Diffusion XL path work at all?** Every other check in this stage runs on Stable
   Diffusion 1.4 and asks whether the refactor left it alone. None of them exercises the new branch.
   This script is the only evidence that the second text encoder is loaded, that the conditioning is
   built in the shape the denoiser expects, and that a step completes end to end. It prints the
   shapes of the three conditioning tensors so that a reader can check them against Stable Diffusion
   XL's own contract rather than trusting that no exception means correct.
2. **Does it fit?** The feasibility probe measured 10.010 GB for a step with the split backward, on
   hand-written code that drove the components directly. The campaign runs the library trainer, which
   is not the same code, and the headroom is 1.97 GB of 11.984 GB. So the peak is measured here, on
   this exact path, and the probe's figure is not reused as evidence.

The run is one optimizer step, at the resolution and hyperparameters the campaign will use, under the
S1 watchdog -- a monitor that hard-aborts if free system memory falls below its floor, and a one-shot
headroom check before any model is loaded. Both exist because the second feasibility probe took the
host down.

    python xl_step_check.py

Writes assets/xl_step_check.json and prints every number it records. Like every check in this stage
it prints measurements, never a verdict: the pass condition lives in the plan and in the job entry,
not in this file's exit code.
'''
import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from vision_unlearning.unlearner.spare import UnlearnerSpare
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

from step_check_support import StopAfterTraining, restore_post_training, stub_post_training, write_image_folder
from watchdog import ResourceMonitor, check_headroom

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

SEED = 42
RESOLUTION = 512
FORGET_CAPTION = "Fixture_Person"
RETAIN_CAPTIONS = ["Other_Person_A", "Other_Person_B"]
OVERWRITING_CONCEPT = "An image of a child"

ASSETS = Path(__file__).resolve().parent / "assets"


def _record_conditioning_shapes(unlearner: UnlearnerSpare, sink: Dict[str, Any]) -> None:
    '''
    Wraps `_encode_conditioning` so that the first call of each role records what it produced.

    The shapes are the only observable evidence that the Stable Diffusion XL branch built the right
    thing: nothing downstream would fail loudly on a plausible-but-wrong `time_ids`, and the loss is
    a single number that cannot distinguish one conditioning from another.
    '''
    original = unlearner._encode_conditioning

    def wrapper(batch: Dict[str, Any], role: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        hidden_states, added = original(batch, role)  # type: ignore[arg-type]
        if role not in sink:
            record: Dict[str, Any] = {
                "prompt_embeds": list(hidden_states.shape),
                "prompt_embeds_dtype": str(hidden_states.dtype),
            }
            for name, value in added.items():
                record[name] = list(value.shape)
            if "time_ids" in added:
                # Printed as values, not only as a shape: the six numbers are the original size, the
                # crop offset and the target size, and a plausible shape with the wrong contents is
                # exactly the failure no exception would report.
                record["time_ids_values"] = [round(float(v), 1) for v in added["time_ids"].flatten().tolist()]
            sink[role] = record
        return hidden_states, added

    unlearner._encode_conditioning = wrapper  # type: ignore[assignment,method-assign]


def train_one_step(work_dir: Path, monitor_log: Path) -> Dict[str, Any]:
    '''
    Runs a single optimizer step on the Stable Diffusion XL checkpoint and returns what it measured.
    '''
    rng = np.random.default_rng(SEED)
    forget_dir = work_dir / "train_forget"
    retain_dir = work_dir / "train_retain"
    write_image_folder(forget_dir, [FORGET_CAPTION, FORGET_CAPTION], rng, RESOLUTION)
    write_image_folder(retain_dir, RETAIN_CAPTIONS, rng, RESOLUTION)

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(work_dir / "adapter"),
        "hub_model_id": None,
        "final_eval_prompts_forget": [],
        "final_eval_prompts_retain": [],
        "model_name_or_path": MODEL_ID,
        "pretrained_vae_model_name_or_path": VAE_ID,
        # The half-precision weight files, not the full-precision ones: reading the 10.3 GB
        # full-precision denoiser into system memory is what the watchdog aborted on the first
        # attempt at this check.
        "variant": "fp16",
        "dataset_forget_name": str(forget_dir),
        "dataset_retain_name": str(retain_dir),
        "validation_prompt": None,
        "dataloader_num_workers": 0,
        "resolution": RESOLUTION,
        "device": "cuda",
        # Half precision with the half-precision-safe autoencoder: plan D4/D4b. Full precision is
        # measured impossible here -- the denoiser weights alone are about 13.9 GB against 11.984.
        "mixed_precision": "fp16",
        # The campaign's own hyperparameters (learning rate 6e-4, rank 16), so this measures what will run.
        "learning_rate": 6e-4,
        "max_grad_norm": 5.0,
        "num_train_epochs": 1,
        "max_train_steps": 1,
        "validation_epochs": 2,
        "checkpointing_steps": 100000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "random_flip": False,
        "center_crop": True,
        "lora_r": 16,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": 4,
        "seed": SEED,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "overwritting_concept": OVERWRITING_CONCEPT,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        "compute_runtimes": False,
        "compute_memory": True,
    }

    conditioning: Dict[str, Any] = {}
    monitor = ResourceMonitor(monitor_log, interval_s=10.0)
    original_unlearn_lora = stub_post_training()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        unlearner = UnlearnerSpare(**hyperparameters)
        _record_conditioning_shapes(unlearner, conditioning)
        monitor.start()
        try:
            unlearner.train()
        except StopAfterTraining:
            pass
    finally:
        monitor.stop()
        restore_post_training(original_unlearn_lora)

    adapter_path = work_dir / "adapter" / "pytorch_lora_weights.safetensors"
    return {
        "is_xl": bool(unlearner._is_xl),
        "seconds_total": round(time.time() - t0, 1),
        "peak_torch_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "peak_torch_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
        "adapter_written": adapter_path.is_file(),
        "adapter_bytes": adapter_path.stat().st_size if adapter_path.is_file() else None,
        "conditioning": conditioning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One SPARE training step on Stable Diffusion XL, measured.")
    parser.add_argument("--output", default=str(ASSETS / "xl_step_check.json"))
    args = parser.parse_args()

    check_headroom()

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    with tempfile.TemporaryDirectory(prefix="xl_step_") as tmp:
        payload = train_one_step(Path(tmp), ASSETS / "xl_step_check_monitor.log")

    payload.update({
        "checkpoint": MODEL_ID,
        "autoencoder": VAE_ID,
        "resolution": RESOLUTION,
        "seed": SEED,
        "total_vram_gb": round(total_vram_gb, 3),
        "headroom_gb": round(total_vram_gb - payload["peak_torch_allocated_gb"], 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("")
    print(json.dumps(payload, indent=2))
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
