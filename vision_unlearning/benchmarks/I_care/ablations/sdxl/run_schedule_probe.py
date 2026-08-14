'''
S4 of PLAN-TASK-2026-08-12-SDXL: the schedule probe. Does 5 epochs of SPARE unlearning under the
D8 "Class A" (inherited Stable Diffusion 1.4) hyperparameters move the target concept on Stable
Diffusion XL, and does it do so gradually rather than collapsing by epoch 1?

Task/target: people / `Mark_Philippoussis` (plan D1), read from
`ablations/every_epoch/assets/selection_people.json` rather than retyped (plan D2's convention).
Trains `output_dir/epoch-{1,3,5}/` LoRA adapters against the real
`assets/datasets/lfw_splits_filtered/Mark_Philippoussis/` split (S0), then generates the target
prompt only, at each of those three epochs plus the off (base-model) baseline, and reads `clip_diff`
(`clip_on - clip_off`, the same primitive `pipeline_06`/`metric_progression.py`/`run_demo_trajectory.py`
use). Writes `assets/schedule_probe.json`.

**Three separate processes, one script, three `--stage` values** -- this is not a style choice, it is
required by a measured constraint (S2, `check_generation_gate.py`): building a second Stable Diffusion
XL pipeline inside a process that already holds one drives free system memory to ~1.3 GB even after
`del` and `empty_cache()`. So training and every individual generation call are separate invocations,
extending the plan's C11 (training and generation in separate processes) to the generation stage
itself, exactly as S5's `run_campaign.py` is designed to. The CLIP scoring stage needs no GPU pipeline
at all and is safe to run in-process after every image exists.

    python run_schedule_probe.py --stage train
    python run_schedule_probe.py --stage generate --off
    python run_schedule_probe.py --stage generate --epoch 1
    python run_schedule_probe.py --stage generate --epoch 3
    python run_schedule_probe.py --stage generate --epoch 5
    python run_schedule_probe.py --stage report

Run from this directory, GPU-capable interpreter, PYTHONPATH at the vision-unlearning repo root,
`HF_HUB_DISABLE_XET=1` for the training stage (SDXL VAE `madebyollin/sdxl-vae-fp16-fix` is fetched
once and then cached).

**The sign-off gate's numeric interpretation, stated explicitly because the plan states it in words
("meaningfully more negative than the noise floor's scale", "not already at the schedule-probe's
floor") rather than as numbers (`CONTRIBUTING.md` working-method item 10: satisfy an ambiguous
instruction's hard constraints, then state the interpretation used).** Two conditions, both read off
`clip_diff`:

1. `clip_diff` at epoch 5 is at or below -5.0. This mirrors, not invents, a number already in the
   plan: D1's own comparison table uses "target crosses -5" as `Mark_Philippoussis`'s
   Stable-Diffusion-1.4 unlearning threshold, and Mark_Philippoussis's own single-entity noise floor
   is 2.51 (`every_epoch/assets/noise_floor_people.json`), so -5.0 is about 2x that floor -- distinct
   from noise on the same target the plan already measured.
2. `clip_diff` at epoch 1 has NOT already reached -5.0. This is the "not already collapsed" check:
   D1 explicitly rejected breeds as this task's target because it crosses -5 by epoch 2, "over in the
   first fifth" of its window, leaving nothing to read across an intermediate-epoch experiment. The
   same reasoning applied to epoch 1 of this 5-epoch probe: if it is already past the threshold that
   defines "unlearned" for this target, the remaining 4 epochs cannot show a trajectory.

Both TRUE -> Class A (the plan's D8 default hyperparameters) passes and the campaign trains under
them, unchanged, at S5. Either FALSE -> the plan's own predeclared fallback: switch to D8's Class B
(the colleague's tuned) hyperparameters and rerun this stage once before touching S5.

**Class A failed on its first run (2026-08-14): the trajectory was non-monotonic** -- clip_diff moved
to -6.09 by epoch 3 (crossing the threshold) then partially recovered to -2.11 by epoch 5, so
condition 1 (epoch 5 at or below -5.0) failed. Every `--stage` therefore takes `--class {A,B}`
(default `A`), which selects D8's two predeclared hyperparameter vectors and keeps their artifacts
under separate paths (`schedule_probe_model_A/` vs `_B/`, etc.) so Class A's own trajectory survives
as the evidence for "the inherited procedure alone does not sustain unlearning here" -- D8 requires
this, since a Class B run does not by itself support the transfer claim and the report must be able to
show both.
'''
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_ICARE_DIR = _HERE.parents[1]
_SELECTION = _HERE.parent / "every_epoch" / "assets" / "selection_people.json"
_SPLIT_BASE = _ICARE_DIR / "assets" / "datasets" / "lfw_splits_filtered"

_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
_TASK = "people"
_METHOD = "distil"
_RESOLUTION = 512
_SEED = 42
_N_EPOCHS = 5
_CHECKPOINTS = [1, 3, 5]
_VARIANT = "fp16"

_HyperparameterClass = Literal["A", "B"]

# D8: Class A is the inherited Stable Diffusion 1.4 procedure, unchanged -- the only evidence for the
# methodological-transfer claim. Class B is the colleague's tuned fallback, with the two values D8
# does NOT adopt (batch size, the retain-driven step cap) held at Class A's/measured values instead,
# per D8's own stated reasons (memory headroom; the cap is meaningless under our forget-driven epoch).
_CLASS_HYPERPARAMETERS: Dict[_HyperparameterClass, Dict[str, Any]] = {
    "A": {"learning_rate": 6e-4, "lora_r": 16, "lora_alpha": 4, "forget_weight": 0.3},
    "B": {"learning_rate": 1e-4, "lora_r": 4, "lora_alpha": 4, "forget_weight": 0.5},
}


def _paths(hp_class: _HyperparameterClass) -> Dict[str, Path]:
    suffix = f"_{hp_class}"
    return {
        "model_dir": _OUT / f"schedule_probe_model{suffix}",
        "gen_dir": _OUT / f"schedule_probe_generated{suffix}",
        "output_json": _OUT / f"schedule_probe{suffix}.json",
        "train_json": _OUT / f"schedule_probe_train{suffix}.json",
        "train_monitor_log": _OUT / f"schedule_probe_train_monitor{suffix}.log",
    }


# Plan D1: -5.0 is the same threshold D1's own comparison table uses for this target's Stable
# Diffusion 1.4 trajectory ("target crosses -5"), about 2x Mark_Philippoussis's own noise floor
# (2.51, every_epoch/assets/noise_floor_people.json). See this module's docstring for the full
# derivation of both gate conditions.
_CROSSING_THRESHOLD = -5.0


def _target_and_prompt() -> Dict[str, str]:
    '''Reads the target and its overwrite concept from the every-epoch selection (plan D1/D2).'''
    from vision_unlearning.datasets.testbed import get_target_overwrite

    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    target_name = selection["target"]["name"]
    target_pre, target_over = get_target_overwrite(_TASK, _METHOD, target_name)  # type: ignore[arg-type]
    return {
        "target_name": target_name,
        "target_pre": target_pre,
        "target_overwrite": target_over,
        "prompt": f"An image of {target_pre}",
    }


def _stage_train(hp_class: _HyperparameterClass) -> None:
    '''Trains 5 epochs under D8 Class A or B, saving adapters at epochs 1, 3 and 5.'''
    check_headroom()

    from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    from step_check_support import StopAfterTraining, restore_post_training, stub_post_training

    paths = _paths(hp_class)
    hp = _CLASS_HYPERPARAMETERS[hp_class]
    info = _target_and_prompt()
    split_base = _SPLIT_BASE / info["target_name"]
    assert (split_base / "train_forget").is_dir() and (split_base / "train_retain").is_dir(), \
        f"forget/retain splits missing for {_TASK}/{info['target_name']} under {split_base}"

    model_dir = paths["model_dir"]
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(model_dir),
        "hub_model_id": None,
        # Post-training evaluation is stubbed below (same reason as every check in this stage):
        # building the pipelines `unlearn_lora` needs for it, on top of the training-time weights
        # still resident, is the exact three-simultaneous-pipeline condition C7/S2 measured
        # dangerous. So these are irrelevant to what actually runs, but kept explicit and empty
        # rather than omitted, matching every other script in this stage.
        "final_eval_prompts_forget": [],
        "final_eval_prompts_retain": [],
        "model_name_or_path": _MODEL_ID,
        "pretrained_vae_model_name_or_path": _VAE_ID,
        "variant": _VARIANT,
        "dataset_forget_name": str(split_base / "train_forget"),
        "dataset_retain_name": str(split_base / "train_retain"),
        "validation_prompt": None,
        "dataloader_num_workers": 0,
        "resolution": _RESOLUTION,
        "device": "cuda",
        # D4/D4b: half precision with the half-precision-safe autoencoder.
        "mixed_precision": "fp16",
        # D8: Class A is the inherited Stable Diffusion 1.4 procedure, unchanged; Class B is the
        # colleague's tuned fallback. Batch size, accumulation and epoch semantics are held at D8's
        # measured/unchanged values under BOTH classes -- see the module docstring and D8 itself.
        "learning_rate": hp["learning_rate"],
        "max_grad_norm": 5.0,
        "num_train_epochs": _N_EPOCHS,
        "validation_epochs": _N_EPOCHS + 1,
        "checkpointing_steps": 100000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "random_flip": True,
        "lora_r": hp["lora_r"],
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": hp["lora_alpha"],
        "lora_dropout": 0.2,
        "seed": _SEED,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "overwritting_concept": info["target_overwrite"],
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=hp["forget_weight"], retain_weight=1.0),
        "save_lora_at_epochs": _CHECKPOINTS,
        "compute_runtimes": False,
        "compute_memory": True,
    }

    monitor = ResourceMonitor(paths["train_monitor_log"], interval_s=15.0)
    original_unlearn_lora = stub_post_training()
    t0 = time.time()
    try:
        unlearner = UnlearnerLoraDistillation(**hyperparameters)
        monitor.start()
        try:
            unlearner.train()
        except StopAfterTraining:
            pass
    finally:
        monitor.stop()
        restore_post_training(original_unlearn_lora)
    train_s = round(time.time() - t0, 1)

    for n in _CHECKPOINTS:
        adapter_path = model_dir / f"epoch-{n}" / "pytorch_lora_weights.safetensors"
        assert adapter_path.is_file(), f"expected checkpoint not written: {adapter_path}"

    result = {
        "stage": "train",
        "hyperparameter_class": hp_class,
        "hyperparameters": hp,
        "train_seconds": train_s,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
        "checkpoints_written": _CHECKPOINTS,
        **info,
    }
    paths["train_json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _stage_generate(hp_class: _HyperparameterClass, epoch: Optional[int]) -> None:
    '''Generates exactly one image: the off-baseline (epoch=None) or one on-checkpoint.'''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    paths = _paths(hp_class)
    info = _target_and_prompt()
    gen_dir = paths["gen_dir"]
    gen_dir.mkdir(parents=True, exist_ok=True)
    label = "off" if epoch is None else f"on_epoch{epoch}"
    filename = f"{label}_seed{_SEED}.png"
    lora_name = None if epoch is None else str(paths["model_dir"] / f"epoch-{epoch}")

    monitor = ResourceMonitor(_OUT / f"schedule_probe_generate_{label}_{hp_class}_monitor.log", interval_s=10.0)
    monitor.start()
    t0 = time.time()
    try:
        generate_dataset(
            model_base_name=_MODEL_ID,
            lora_name=lora_name,
            prompts=[info["prompt"]],
            output_path=str(gen_dir),
            filenames=[filename],
            seeds=[_SEED],
            batch_size=1,
            # C9: SPARE (distil) leaves the unlearned pipeline as trained; only munba-family
            # methods require the inversion.
            lora_requires_inversion=False,
            # D10: explicit 512x512, otherwise Stable Diffusion XL defaults to 1024.
            height=_RESOLUTION,
            width=_RESOLUTION,
            variant=_VARIANT,
        )
    finally:
        monitor.stop()
    seconds = round(time.time() - t0, 1)

    print(json.dumps({
        "stage": "generate",
        "hyperparameter_class": hp_class,
        "label": label,
        "epoch": epoch,
        "filename": filename,
        "seconds": seconds,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }, indent=2))


def _stage_report(hp_class: _HyperparameterClass) -> Dict[str, Any]:
    '''Reads the four generated images, scores clip_diff, applies the sign-off gate, writes the result.'''
    from PIL import Image
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    paths = _paths(hp_class)
    gen_dir = paths["gen_dir"]
    info = _target_and_prompt()
    prompt = info["prompt"]

    off_path = gen_dir / f"off_seed{_SEED}.png"
    assert off_path.is_file(), f"missing off-baseline image: {off_path}"

    metric = MetricImageTextSimilarity(metrics=["clip"])
    off_img = Image.open(off_path).convert("RGB")
    clip_off = metric.score_batch_same_text([off_img], prompt)[0]["clip"]

    trajectory: List[Dict[str, Any]] = []
    for n in _CHECKPOINTS:
        on_path = gen_dir / f"on_epoch{n}_seed{_SEED}.png"
        assert on_path.is_file(), f"missing epoch-{n} image: {on_path}"
        on_img = Image.open(on_path).convert("RGB")
        clip_on = metric.score_batch_same_text([on_img], prompt)[0]["clip"]
        clip_diff = clip_on - clip_off
        trajectory.append({"epoch": n, "clip_on": clip_on, "clip_off": clip_off, "clip_diff": clip_diff})

    clip_diff_by_epoch = {row["epoch"]: row["clip_diff"] for row in trajectory}
    condition_1_epoch5_crosses = clip_diff_by_epoch[5] <= _CROSSING_THRESHOLD
    condition_2_epoch1_not_collapsed = clip_diff_by_epoch[1] > _CROSSING_THRESHOLD
    gate_passes = condition_1_epoch5_crosses and condition_2_epoch1_not_collapsed

    result = {
        "task": _TASK,
        "method": _METHOD,
        "target": info["target_pre"],
        "overwrite": info["target_overwrite"],
        "prompt": prompt,
        "seed": _SEED,
        "epochs_trained": _N_EPOCHS,
        "checkpoints": _CHECKPOINTS,
        "hyperparameter_class": hp_class,
        "hyperparameters": _CLASS_HYPERPARAMETERS[hp_class],
        "trajectory": trajectory,
        "gate": {
            "crossing_threshold": _CROSSING_THRESHOLD,
            "condition_1_epoch5_at_or_below_threshold": condition_1_epoch5_crosses,
            "condition_2_epoch1_above_threshold": condition_2_epoch1_not_collapsed,
            "gate_passes": gate_passes,
        },
    }
    paths["output_json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S4 schedule probe: 5-epoch SPARE unlearning sign-off gate.")
    parser.add_argument("--stage", choices=["train", "generate", "report"], required=True)
    parser.add_argument("--class", dest="hp_class", choices=["A", "B"], default="A",
                        help="D8 hyperparameter class: A (inherited Stable Diffusion 1.4) or B (colleague's tuned fallback)")
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch for --stage generate")
    parser.add_argument("--off", action="store_true", help="generate the off (base-model) baseline instead")
    args = parser.parse_args()

    if args.stage == "train":
        _stage_train(args.hp_class)
    elif args.stage == "generate":
        if args.off == (args.epoch is not None):
            raise SystemExit("--stage generate needs exactly one of --off or --epoch N")
        _stage_generate(args.hp_class, None if args.off else args.epoch)
    else:
        _stage_report(args.hp_class)


if __name__ == "__main__":
    main()
