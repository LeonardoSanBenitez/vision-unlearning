'''
S5/S6 of PLAN-TASK-2026-08-12-SDXL: the full SPARE-unlearning campaign of `Mark_Philippoussis` on
Stable Diffusion XL.

TRAINING HYPERPARAMETERS ARE NOT SETTLED. The first campaign's selection (Class B, 2026-08-14) was
made from images generated at 512 and is VOID with the rest of that campaign; plan stage S4 re-runs
the choice on 768-pixel images. The constants below are still Class B's, so this script trains what
the void campaign trained until S4 replaces them -- do not start stage S5 before it has.

Two `--stage` values, always their own process, and `--stage generate` does EXACTLY ONE epoch/off-
baseline per invocation, never a batch inside one process (C11, extended by S2's one-pipeline-per-
process measurement -- a second Stable Diffusion XL pipeline built in a process that already holds one
drives free system memory to ~1.3 GB even after `del` and `empty_cache()`). `--epochs` accepts a
single integer or `"off"`; `"remaining"`/`"all"` are a convenience that report what is still missing
and generate only the NEXT one -- the caller re-invokes this script, once per process, for each
epoch that remains:

    python run_campaign.py --stage train --seed 42
    python run_campaign.py --stage generate --seed 42 --epochs off
    python run_campaign.py --stage generate --seed 42 --epochs 1
    python run_campaign.py --stage generate --seed 42 --epochs 3
    python run_campaign.py --stage generate --seed 42 --epochs 5
    python run_campaign.py --stage generate --seed 42 --epochs remaining   # repeat until remaining_after_this is 0
    python run_campaign.py --stage generate --seed 43 --epochs all        # repeat until remaining_after_this is 0

A comma-separated `--epochs` list is refused outright (not silently truncated to its first entry) --
see `_resolve_epochs`.

`--stage train` trains 200 epochs, saving adapters at the 13-entry checkpoint list read from
`every_epoch/assets/epoch_grid_campaign_people_seed42.json` (`[1,2,3,5,10,15,20,30,50,75,100,150,200]`,
never retyped). `--stage generate` renders ALL TEN of `selection_people.json`'s entities (the target
plus its nine receivers) for that one epoch/off-baseline, as TEN SEPARATE `generate_dataset` calls at
the frozen generation configuration of plan section 2.1 -- 768 pixels, size micro-conditioning
declared as 1024, guidance 7.5, one entity per call with the generator reseeded. The one-call-ten-
prompts shape the first campaign used (copied from `make_epoch_grid.py`, to hold each entity at the
same random-number-generator position as the Stable Diffusion 1.4 grids) is abandoned on purpose: at
512 that position decides whether the image depicts the person at all, and the configuration that
renders all ten correctly generates each alone. See `_stage_generate` and `assets/VALIDATION_REPORT_01.md`.
Generation order is still fixed by `selection_people.json` alone, sorted by each entity's own
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
from typing import Any, Dict, List, Optional, Tuple

from vision_unlearning.utils.data_generation import GenerationRuntime

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
# Training resolution: the same 768 the images are generated at (D3). At 512 the peak was 10.837 GB
# of 11.98 without gradient checkpointing; at 768 with it the spike measured 9.435 GB, so
# checkpointing is what makes this fit and it is not optional here.
_TRAIN_RESOLUTION = 768
_GRADIENT_CHECKPOINTING = True
# D14: training declares the same size micro-conditioning generation declares. Fitting the adapter
# under the photographs' own 250x250 and using it under 1024 cost half the measured effect
# (adapter_transfer.json), and moving to 768 removed the rest.
_TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE = (1024, 1024)
_TRAIN_MICRO_CONDITIONING_TARGET_SIZE = (1024, 1024)
_N_EPOCHS = 200
_VARIANT = "fp16"

# Class B of the void campaign, kept as a placeholder until plan stage S4 chooses the hyperparameters
# again on 768-pixel images (see the module docstring). Batch size, accumulation and epoch semantics
# are held at D8's own measured/unchanged values (see run_schedule_probe.py's identical override).
_LEARNING_RATE = 1e-4
_LORA_R = 4
_LORA_ALPHA = 4
_FORGET_WEIGHT = 0.5

_MODEL_DIR = _OUT / "campaign_model"
_MANIFEST_TEMPLATE = "campaign_seed{seed}.json"

# --- The frozen generation configuration (plan section 2.1) ------------------------------------- #
# Validated in assets/VALIDATION_REPORT_01.md and fixed for the rest of the task: 20 of 20
# off-baselines depict the right person at both campaign seeds. Every image the campaign produces --
# off-baselines, on-images, controls -- is generated with exactly these values, one entity per call.
_GENERATION_RESOLUTION = 768
_GENERATION_KWARGS: Dict[str, Any] = {
    "guidance_scale": 7.5,
    "original_size": (1024, 1024),
    "target_size": (1024, 1024),
    "crops_coords_top_left": (0, 0),
}
# Card settings required at 768 on this machine, measured in spike_768.py: without tiling forced to
# 512-pixel tiles the autoencoder never tiles below 1024 pixels, and with deterministic algorithms on
# a convolution dies with `HIP error: unspecified launch failure`.
_GENERATION_RUNTIME = GenerationRuntime(
    attention_slice_size=1,
    vae_slicing=True,
    vae_tiling=True,
    vae_tile_sample_min_size=512,
    deterministic_algorithms=False,
    device_map="balanced",
)


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
        "resolution": _TRAIN_RESOLUTION,
        "gradient_checkpointing": _GRADIENT_CHECKPOINTING,
        "micro_conditioning_original_size": _TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE,
        "micro_conditioning_target_size": _TRAIN_MICRO_CONDITIONING_TARGET_SIZE,
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
    print(f"CAMPAIGN_TRAIN_DONE seed={seed} checkpoints={len(checkpoints)} seconds={train_s} "
          f"peak_vram_used_gb={result['peak_vram_used_gb']}")


def _resolve_epochs(seed: int, epochs_arg: str) -> List[Any]:
    '''Turns --epochs into an ORDERED list of what remains to generate, per the requested selector.

    `_stage_generate` only ever acts on the FIRST element -- see its own docstring for why -- so this
    function's job is to compute the full remaining set (for the "how many are left" figure it
    prints), never to hand back something the caller is expected to loop over inside one process.
    '''
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
    # A single epoch is the only other accepted form -- a comma list is refused rather than silently
    # generating only its first entry, which is what taking to_run[0] on a multi-item list would do.
    assert "," not in epochs_arg, (
        f"--epochs {epochs_arg!r}: a comma-separated list is refused, not silently truncated to its "
        "first entry. Invoke this script once per epoch, each its own process (S2's one-pipeline-"
        "per-process constraint) -- see this module's docstring."
    )
    epoch = int(epochs_arg)
    assert epoch in checkpoints, f"epoch {epoch} is not in the checkpoint list {checkpoints}"
    return [] if epoch in already_generated else [epoch]


def _build_generation_pipeline(lora_name: Optional[str]) -> Any:
    '''Builds the one pipeline the ten per-entity calls share, adapted or not.

    `generate_dataset` builds its own pipeline whenever it is given `model_base_name` or `lora_name`,
    which for ten calls would mean ten builds inside one process -- the condition S2 measured driving
    free system memory to ~1.3 GB, and about eleven hours of pure loading over the campaign. So the
    build is hoisted here and handed to every call as `model_pipeline`. The two branches are the same
    two the library itself takes (`utils/data_generation.py`), with the same arguments; the runtime
    settings are NOT applied here, because `generate_dataset` applies them to whatever pipeline it is
    given.

    @param lora_name: the adapter directory for an on-image epoch, or None for the off-baseline.
    @return: the pipeline, already placed by the loader's own device map.
    '''
    import torch
    from diffusers import AutoPipelineForText2Image

    from vision_unlearning.unlearner.lora import unlearn_lora

    if lora_name is not None:
        _, _, pipeline = unlearn_lora(
            _MODEL_ID, lora_name, device="cuda",
            weight_name="pytorch_lora_weights.safetensors",
            # C9: SPARE (distil) leaves the unlearned pipeline as trained.
            requires_inversion=False, return_original=False, return_learned=False,
            variant=_VARIANT, device_map=_GENERATION_RUNTIME.device_map,
        )
        return pipeline
    return AutoPipelineForText2Image.from_pretrained(
        _MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
        variant=_VARIANT, device_map=_GENERATION_RUNTIME.device_map,
    )


def _stage_generate(seed: int, epochs_arg: str) -> None:
    '''ONE epoch (or the off-baseline) per process, as TEN `generate_dataset` calls -- one per entity.

    Two constraints shape this, and they pull in opposite directions.

    Per PROCESS: never more than one epoch. S2 measured that a second Stable Diffusion XL pipeline
    built in a process that already holds one drives free system memory to ~1.3 GB even after `del`
    and `empty_cache()`. So "remaining"/"all" only pick the NEXT missing epoch and generate that one,
    printing how many are still outstanding; the caller invokes this as a fresh process for each.

    Per CALL: never more than one entity. The void campaign put all ten prompts in a single call,
    copying `make_epoch_grid.py`'s shape so that each entity sat at the same random-number-generator
    position as the Stable Diffusion 1.4 grids. At 768 that shape is abandoned deliberately (plan
    section 2.1): whether a Stable Diffusion XL render depicts the person asked for depends on the
    initial noise sample, so every image is generated alone with the generator reseeded, which is the
    configuration `validate_generation_768.py` signed off with 20 of 20 correct off-baselines. The
    cross-model comparison this gives up was never available anyway -- a different model, autoencoder
    and resolution make pixel-level comparability impossible (plan section 3), and the comparison that
    survives is of shape and ordering.

    The ten calls SHARE ONE PIPELINE (`_build_generation_pipeline`), passed as `model_pipeline`:
    letting each call build its own would be ten Stable Diffusion XL loads inside one process, which
    is both the memory condition above and about eleven hours of loading over the campaign. Sharing
    the pipeline does not share the random-number generator -- `generate_dataset` reseeds the global
    sources and builds a fresh generator on every call, which is exactly the per-entity reseeding
    this shape exists for.

    Images already on disk are skipped, so an abort costs the image in flight and nothing else. The
    manifest rows are appended once, after all ten images exist, so a partial epoch is retried whole
    rather than recorded as done.

    One consequence of one call per image: `generate_dataset` rewrites `metadata.jsonl` in the output
    directory at the end of every call, so the copy left in `campaign_seed<seed>/` describes only the
    last image generated. It is not read by anything here -- `campaign_seed<seed>.json`, written
    below, is this campaign's record of what was generated.
    '''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    order = _generation_order()
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

    item = to_run[0]
    remaining_after_this = len(to_run) - 1
    is_off = item == "off"
    label = "off" if is_off else f"epoch{item}"
    lora_name = None if is_off else str(model_dir / f"epoch-{item}")
    if lora_name is not None:
        adapter_file = Path(lora_name) / "pytorch_lora_weights.safetensors"
        assert adapter_file.is_file(), f"adapter missing for epoch {item}, seed {seed}: {adapter_file}"

    monitor = ResourceMonitor(_OUT / f"campaign_generate_seed{seed}_{label}_monitor.log", interval_s=15.0)
    monitor.start()
    t0 = time.time()
    per_entity: List[Dict[str, Any]] = []
    # Built on the first image that actually has to be generated, so a rerun whose images are all
    # already on disk costs no model load at all.
    pipeline: Optional[Any] = None
    try:
        for entity in order:
            filename = f"{label}_{entity['name']}_seed{seed}.png"
            path = gen_dir / filename
            if path.is_file():
                per_entity.append({"entity": entity["name"], "seconds": None, "skipped": True})
                print(f"already on disk, skipping: {filename}", flush=True)
                continue
            t_image = time.time()
            if pipeline is None:
                pipeline = _build_generation_pipeline(lora_name)
            generate_dataset(
                model_base_name=None,
                lora_name=None,
                model_pipeline=pipeline,
                prompts=[entity["prompt"]],
                output_path=str(gen_dir),
                filenames=[filename],
                seeds=[seed],
                batch_size=1,
                height=_GENERATION_RESOLUTION,
                width=_GENERATION_RESOLUTION,
                runtime=_GENERATION_RUNTIME,
                **_GENERATION_KWARGS,
            )
            assert path.is_file(), f"generate_dataset did not write {path}"
            per_entity.append({"entity": entity["name"], "seconds": round(time.time() - t_image, 1),
                               "skipped": False})
            print(f"{label} {entity['name']} seed {seed}: {per_entity[-1]['seconds']} s "
                  f"({len(per_entity)} of {len(order)})", flush=True)
    finally:
        monitor.stop()
    seconds = round(time.time() - t0, 1)

    rows = [
        {"epoch": None if is_off else item, "entity": entity["name"],
         "path": str(gen_dir / f"{label}_{entity['name']}_seed{seed}.png"), "seed": seed}
        for entity in order
    ]
    _append_manifest(seed, rows)

    print(json.dumps({
        "stage": "generate", "seed": seed, "label": label, "rows_written": len(rows),
        "calls": len(order), "seconds": seconds, "per_entity": per_entity,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
        "remaining_after_this": remaining_after_this,
    }, indent=2))
    print(f"CAMPAIGN_GENERATE_DONE seed={seed} label={label} images={len(rows)} "
          f"remaining_after_this={remaining_after_this}")


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
