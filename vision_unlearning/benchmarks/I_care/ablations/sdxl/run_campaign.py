'''
S5/S6 of PLAN-TASK-2026-08-12-SDXL: the full SPARE-unlearning campaign of `Mark_Philippoussis` on
Stable Diffusion XL, at D8 Class B hyperparameters (selected 2026-08-14 -- see the plan's `# State`
S4 row: both predeclared classes were run, the sign-off gate's own numeric threshold was found
miscalibrated after the fact, and the user's own judgement call, made from the raw numbers and the
visible-but-unscored degradation in Class B's images, is to proceed with Class B; a rerun is cheap if
this turns out wrong).

Two `--stage` values, always their own process, one call per epoch/off-baseline (C11, extended by S2's
one-pipeline-per-process measurement):

    python run_campaign.py --stage train --seed 42
    python run_campaign.py --stage generate --seed 42 --epochs off
    python run_campaign.py --stage generate --seed 42 --epochs 1,3,5
    python run_campaign.py --stage generate --seed 42 --epochs remaining
    python run_campaign.py --stage generate --seed 43 --epochs all

`--stage train` trains 200 epochs, saving adapters at the 13-entry checkpoint list read from
`every_epoch/assets/epoch_grid_campaign_people_seed42.json` (`[1,2,3,5,10,15,20,30,50,75,100,150,200]`,
never retyped). `--stage generate` renders ALL TEN of `selection_people.json`'s entities (the target
plus its nine receivers) in ONE `generate_dataset` call per epoch/off-baseline -- this is the call
shape `make_epoch_grid.py` uses (read from its source before this script was written, per the plan's
own instruction: "settle the call shape... from the code"), not `run_demo_trajectory.py`'s
one-prompt-per-call shape. Getting this wrong would put every entity at a different random-number-
generator position than the every-epoch grids used, silently breaking the whole cross-model
comparison. Generation order is fixed by `selection_people.json` alone, sorted by each entity's own
`clip_diff`/`self_clip_diff` -- the exact sort `make_epoch_grid.py` applies -- and is NEVER the
column/display order, which is a presentation choice made later at S8.

Appends one row per (epoch-or-off, entity) to `assets/campaign_seed{seed}.json`:
`{"epoch": n_or_null, "entity": name, "path": ..., "seed": seed}`, `epoch: null` marking an
off-baseline row. Read back at S6/S8, never re-derived.

Run from this directory, GPU-capable interpreter, PYTHONPATH at the vision-unlearning repo root,
`HF_HUB_DISABLE_XET=1`.
'''
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_ICARE_DIR = _HERE.parents[1]
_EVERY_EPOCH_ASSETS = _HERE.parent / "every_epoch" / "assets"
_SELECTION = _EVERY_EPOCH_ASSETS / "selection_people.json"
_CHECKPOINT_LIST_SOURCE = _EVERY_EPOCH_ASSETS / "epoch_grid_campaign_people_seed42.json"
_SPLIT_BASE = _ICARE_DIR / "assets" / "datasets" / "lfw_splits_filtered"

_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
_TASK = "people"
_METHOD = "distil"
_RESOLUTION = 512
_N_EPOCHS = 200
_VARIANT = "fp16"

# D8 Class B (selected 2026-08-14): the colleague's tuned fallback, with batch size, accumulation and
# epoch semantics held at D8's own measured/unchanged values (see run_schedule_probe.py's identical
# override and D8 itself for why).
_LEARNING_RATE = 1e-4
_LORA_R = 4
_LORA_ALPHA = 4
_FORGET_WEIGHT = 0.5

_MODEL_DIR = _OUT / "campaign_model"
_MANIFEST_TEMPLATE = "campaign_seed{seed}.json"


def _checkpoint_list() -> List[int]:
    '''Reads the 13-entry checkpoint list from the every-epoch campaign's own JSON. Never retyped.'''
    payload = json.loads(_CHECKPOINT_LIST_SOURCE.read_text(encoding="utf-8"))
    epochs = payload["epochs"]
    assert isinstance(epochs, list) and all(isinstance(e, int) for e in epochs), \
        f"unexpected 'epochs' shape in {_CHECKPOINT_LIST_SOURCE}: {epochs!r}"
    return epochs


def _generation_order() -> List[Dict[str, Any]]:
    '''The ten entities (target + nine receivers), in the fixed generation order `make_epoch_grid.py`
    uses: sorted by each entity's own clip_diff / self_clip_diff, ascending (most negative first).
    This is NOT the display/column order (a later, per-seed presentation choice at S8) -- it is what
    fixes each entity's position in the random-number sequence, and it must match the every-epoch
    grids' own ordering for the images to be comparable at all.
    '''
    from vision_unlearning.datasets.testbed import get_target_overwrite

    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    raw: List[Tuple[str, float]] = [(selection["target"]["name"], selection["target"]["self_clip_diff"])]
    raw += [(r["name"], r["clip_diff"]) for r in selection["receivers"]]
    raw.sort(key=lambda item: item[1])
    assert len(raw) == 10, f"expected 10 entities (target + 9 receivers), got {len(raw)}"

    order = []
    for name, sort_value in raw:
        target_pre, target_over = get_target_overwrite(_TASK, _METHOD, name)  # type: ignore[arg-type]
        order.append({
            "name": name,
            "prompt": f"An image of {target_pre}",
            "overwrite_concept": target_over,
            "sort_value": sort_value,
            "is_target": name == selection["target"]["name"],
        })
    return order


def _append_manifest(seed: int, rows: List[Dict[str, Any]]) -> None:
    manifest_path = _OUT / _MANIFEST_TEMPLATE.format(seed=seed)
    existing: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.extend(rows)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _stage_train(seed: int) -> None:
    '''Trains 200 epochs under D8 Class B, saving adapters at the 13-entry checkpoint list.'''
    check_headroom()

    from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    from step_check_support import StopAfterTraining, restore_post_training, stub_post_training

    checkpoints = _checkpoint_list()
    order = _generation_order()
    target_name = next(e["name"] for e in order if e["is_target"])
    target_overwrite = next(e["overwrite_concept"] for e in order if e["is_target"])
    split_base = _SPLIT_BASE / target_name
    assert (split_base / "train_forget").is_dir() and (split_base / "train_retain").is_dir(), \
        f"forget/retain splits missing for {_TASK}/{target_name} under {split_base}"

    model_dir = _MODEL_DIR / f"seed{seed}"
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(model_dir),
        "hub_model_id": None,
        # Post-training evaluation is stubbed below, same reason as every check in this task: building
        # the pipelines `unlearn_lora` needs for it, on top of the training-time weights still
        # resident, is the three-simultaneous-pipeline condition C7/S2 measured dangerous.
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
        "mixed_precision": "fp16",
        "learning_rate": _LEARNING_RATE,
        "max_grad_norm": 5.0,
        "num_train_epochs": _N_EPOCHS,
        "validation_epochs": _N_EPOCHS + 1,
        "checkpointing_steps": 100000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "random_flip": True,
        "lora_r": _LORA_R,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": _LORA_ALPHA,
        "lora_dropout": 0.2,
        "seed": seed,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "overwritting_concept": target_overwrite,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=_FORGET_WEIGHT, retain_weight=1.0),
        "save_lora_at_epochs": checkpoints,
        "compute_runtimes": False,
        "compute_memory": True,
    }

    monitor = ResourceMonitor(_OUT / f"campaign_train_seed{seed}_monitor.log", interval_s=30.0)
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

    for n in checkpoints:
        adapter_path = model_dir / f"epoch-{n}" / "pytorch_lora_weights.safetensors"
        assert adapter_path.is_file(), f"expected checkpoint not written: {adapter_path}"

    result = {
        "stage": "train",
        "seed": seed,
        "hyperparameter_class": "B",
        "hyperparameters": {
            "learning_rate": _LEARNING_RATE, "lora_r": _LORA_R, "lora_alpha": _LORA_ALPHA,
            "forget_weight": _FORGET_WEIGHT,
        },
        "checkpoints": checkpoints,
        "train_seconds": train_s,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }
    (_OUT / f"campaign_train_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _resolve_epochs(seed: int, epochs_arg: str) -> List[Any]:
    '''Turns --epochs into a list mixing the string "off" and/or ints, per the requested selector.'''
    checkpoints = _checkpoint_list()
    manifest_path = _OUT / _MANIFEST_TEMPLATE.format(seed=seed)
    already_generated: set = set()
    if manifest_path.is_file():
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        already_generated = {row["epoch"] for row in rows}

    if epochs_arg == "off":
        return ["off"] if None not in already_generated else []
    if epochs_arg == "remaining":
        return [e for e in checkpoints if e not in already_generated]
    if epochs_arg == "all":
        result: List[Any] = [] if None in already_generated else ["off"]
        result += [e for e in checkpoints if e not in already_generated]
        return result
    requested = [int(x) for x in epochs_arg.split(",")]
    for e in requested:
        assert e in checkpoints, f"epoch {e} is not in the checkpoint list {checkpoints}"
    return requested


def _stage_generate(seed: int, epochs_arg: str) -> None:
    '''One generate_dataset call per requested epoch/off, over all ten entities in fixed order.'''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    order = _generation_order()
    prompts = [e["prompt"] for e in order]
    gen_dir = _OUT / f"campaign_seed{seed}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _MODEL_DIR / f"seed{seed}"

    to_run = _resolve_epochs(seed, epochs_arg)
    if not to_run:
        print(json.dumps({
            "stage": "generate", "seed": seed, "requested": epochs_arg, "rows_generated": 0,
            "note": "nothing to do -- already in the manifest",
        }, indent=2))
        return

    for item in to_run:
        is_off = item == "off"
        label = "off" if is_off else f"epoch{item}"
        lora_name = None if is_off else str(model_dir / f"epoch-{item}")
        filenames = [f"{label}_{e['name']}_seed{seed}.png" for e in order]

        monitor = ResourceMonitor(_OUT / f"campaign_generate_seed{seed}_{label}_monitor.log", interval_s=15.0)
        monitor.start()
        t0 = time.time()
        try:
            generate_dataset(
                model_base_name=_MODEL_ID,
                lora_name=lora_name,
                prompts=prompts,
                output_path=str(gen_dir),
                filenames=filenames,
                seeds=[seed],
                batch_size=1,
                # C9: SPARE (distil) leaves the unlearned pipeline as trained.
                lora_requires_inversion=False,
                # D10: explicit 512x512, otherwise Stable Diffusion XL defaults to 1024.
                height=_RESOLUTION,
                width=_RESOLUTION,
                variant=_VARIANT,
            )
        finally:
            monitor.stop()
        seconds = round(time.time() - t0, 1)

        rows = [
            {"epoch": None if is_off else item, "entity": e["name"],
             "path": str((gen_dir / fname)), "seed": seed}
            for e, fname in zip(order, filenames)
        ]
        _append_manifest(seed, rows)

        print(json.dumps({
            "stage": "generate", "seed": seed, "label": label, "rows_written": len(rows),
            "seconds": seconds,
            "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
            "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
        }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="S5/S6 campaign: train and generate the Stable Diffusion XL ablation.")
    parser.add_argument("--stage", choices=["train", "generate"], required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 43])
    parser.add_argument("--epochs", type=str, default=None,
                        help="'off', 'remaining', 'all', or a comma-separated list of checkpoint epochs; "
                             "required for --stage generate")
    args = parser.parse_args()

    if args.stage == "train":
        _stage_train(args.seed)
    else:
        assert args.epochs is not None, "--stage generate requires --epochs"
        _stage_generate(args.seed, args.epochs)


if __name__ == "__main__":
    main()
