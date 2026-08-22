'''
S5/S6 of PLAN-TASK-2026-08-12-SDXL: the full SPARE-unlearning campaign of `Mark_Philippoussis` on
Stable Diffusion XL.

TRAINING HYPERPARAMETERS: settled by plan stage S4 on 2026-08-19 (learning rate 6e-4, rank 16, alpha
4, forget weight 0.3 -- the inherited Stable Diffusion 1.4 values, unchanged). The first campaign's
selection was made from images generated at 512 and is void with the rest of that campaign; the
constants below are the ten-epoch, 768-pixel selection that replaced it. See the constants' own
comment for the evidence and for the alternative that was rejected.

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
from typing import Any, Dict, List, Optional

import campaign_configuration as cfg
from campaign_configuration import (
    GENERATION_KWARGS,
    GENERATION_RESOLUTION,
    GENERATION_RUNTIME,
    build_generation_pipeline,
    checkpoint_list,
    generation_order,
)
from watchdog import ResourceMonitor, check_headroom

_OUT = cfg.OUT
_MODEL_DIR = _OUT / "campaign_model"
_MANIFEST_TEMPLATE = "campaign_seed{seed}.json"
_N_EPOCHS = 200

# The hyperparameters plan stage S4 selected on 2026-08-19: the inherited Stable Diffusion 1.4 values,
# unchanged, which is what makes the base model the only variable against the existing Stable Diffusion
# 1.4 curves. Evidence, on 768-pixel images at the frozen generation configuration: the target's
# clip_diff runs +2.37 / +4.22 / -6.58 / -6.00 over epochs 1/3/5/10 against a 2.258 noise floor, and the
# epoch-5 and epoch-10 images show the target replaced by the overwrite concept (a child) rather than
# collapsing. The alternative tried and rejected -- learning rate 1e-4, rank 4, forget weight 0.5 --
# left the target recognisably himself at every checkpoint (assets/schedule_probe_lr1e-04_r4_a4_fw0.5.json).
# Everything else about the training and generation configuration lives in `campaign_configuration.py`,
# which `run_schedule_probe.py` reads too, so the selection was made under what this script runs.
_LEARNING_RATE = 6e-4
_LORA_R = 16
_LORA_ALPHA = 4
_FORGET_WEIGHT = 0.3


def _append_manifest(seed: int, rows: List[Dict[str, Any]]) -> None:
    manifest_path = _OUT / _MANIFEST_TEMPLATE.format(seed=seed)
    existing: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.extend(rows)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _stage_train(seed: int) -> None:
    '''Trains 200 epochs at the S4-selected hyperparameters, saving adapters at the 13-entry checkpoint list.

    A seed whose training has ALREADY finished is not retrained: this returns after printing the
    completion marker. "Finished" is two conditions together, and both are needed — the result record
    exists, which is only written after the last checkpoint, and every checkpoint file is on disk. A
    run interrupted halfway leaves the checkpoints without the record, so it retrains, which is what
    should happen.

    This is not a convenience. The driver re-runs every stage from the top after any failure, and
    without this guard a generation bug costs an hour of retraining that produces different adapters
    from the ones the earlier stages were validated against. To retrain deliberately, delete
    `assets/campaign_train_seed{seed}.json`.
    '''
    record_path = _OUT / f"campaign_train_seed{seed}.json"
    checkpoints_expected = checkpoint_list()
    existing = [n for n in checkpoints_expected
                if (_MODEL_DIR / f"seed{seed}" / f"epoch-{n}" / "pytorch_lora_weights.safetensors").is_file()]
    if record_path.is_file() and len(existing) == len(checkpoints_expected):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        print(json.dumps({"stage": "train", "seed": seed, "note": "already trained -- not retraining",
                          "checkpoints_on_disk": len(existing),
                          "train_seconds": record.get("train_seconds")}, indent=2))
        print(f"CAMPAIGN_TRAIN_DONE seed={seed} checkpoints={len(existing)} "
              f"seconds={record.get('train_seconds')} "
              f"peak_vram_used_gb={record.get('peak_vram_used_gb')}")
        return

    check_headroom()

    from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
    from step_check_support import StopAfterTraining, restore_post_training, stub_post_training

    checkpoints = checkpoint_list()
    order = generation_order()
    target_name = next(e["name"] for e in order if e["is_target"])
    target_overwrite = next(e["overwrite_concept"] for e in order if e["is_target"])
    split_base = cfg.SPLIT_BASE / target_name
    assert (split_base / "train_forget").is_dir() and (split_base / "train_retain").is_dir(), \
        f"forget/retain splits missing for {cfg.TASK}/{target_name} under {split_base}"

    model_dir = _MODEL_DIR / f"seed{seed}"
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    hyperparameters = cfg.training_hyperparameters(
        output_dir=model_dir,
        split_base=split_base,
        overwrite_concept=target_overwrite,
        seed=seed,
        n_epochs=_N_EPOCHS,
        save_lora_at_epochs=checkpoints,
        learning_rate=_LEARNING_RATE,
        lora_r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        forget_weight=_FORGET_WEIGHT,
    )

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
    checkpoints = checkpoint_list()
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

    order = generation_order()
    gen_dir = _OUT / f"campaign_seed{seed}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _MODEL_DIR / f"seed{seed}"

    to_run = _resolve_epochs(seed, epochs_arg)
    if not to_run:
        print(json.dumps({
            "stage": "generate", "seed": seed, "requested": epochs_arg, "rows_generated": 0,
            "note": "nothing to do -- already in the manifest",
        }, indent=2))
        # The completion marker is printed here too, so that asking for an epoch the manifest already
        # holds is a SUCCESS rather than a stage that run_stage.sh retries eight times and then fails.
        # A resumable stage has to be idempotent to be resumable at all: a driver that walks the
        # checkpoint list after an abort re-asks for every epoch, and the ones already generated must
        # answer "done, nothing to do" instead of looking like a failure.
        print(f"CAMPAIGN_GENERATE_DONE seed={seed} label=none images=0 remaining_after_this=0")
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
                pipeline = build_generation_pipeline(lora_name)
            generate_dataset(
                model_base_name=None,
                lora_name=None,
                model_pipeline=pipeline,
                prompts=[entity["prompt"]],
                output_path=str(gen_dir),
                filenames=[filename],
                seeds=[seed],
                batch_size=1,
                height=GENERATION_RESOLUTION,
                width=GENERATION_RESOLUTION,
                runtime=GENERATION_RUNTIME,
                **GENERATION_KWARGS,
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


def _stage_labels() -> None:
    '''Prints the generation labels of one seed, in order, one line each: `off` then each checkpoint.

    This exists so that a shell driver can walk the campaign one epoch per process WITHOUT a
    hand-written epoch list -- the list comes from `checkpoint_list()`, which reads the every-epoch
    campaign JSON. It is also what gives each stage its own log file: `run_campaign_stage.sh` names
    the log after the epoch selector it is given, so driving the campaign with the `remaining`
    selector names every stage `remaining` and each one overwrites the last (which is exactly what
    happened to the ten logs of the seed-42 second half on 2026-08-19).
    '''
    # Line endings matter here in a way they do not for a human-read log: this output is consumed
    # by a shell driver through command substitution, and on Windows the default text-mode stdout
    # translates every newline into a carriage return plus a newline. Bash strips the trailing
    # newline and keeps the carriage return, so the driver ends up asking for an epoch whose name
    # carries an invisible control character, every stage dies with a ValueError, and the failure
    # presents as eight identical attempts with no explanation. Writing the newline explicitly
    # removes the translation. Measured on 2026-08-22, at the cost of the seed-43 generation half.
    import sys
    sys.stdout.reconfigure(newline=chr(10))  # type: ignore[union-attr]
    print("off")
    for epoch in checkpoint_list():
        print(epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="S5/S6 campaign: train and generate the Stable Diffusion XL ablation.")
    parser.add_argument("--stage", choices=["train", "generate", "labels"], required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 43])
    parser.add_argument("--epochs", type=str, default=None,
                        help="'off', 'remaining', 'all', or a comma-separated list of checkpoint epochs; "
                             "required for --stage generate")
    args = parser.parse_args()

    if args.stage == "train":
        _stage_train(args.seed)
    elif args.stage == "labels":
        _stage_labels()
    else:
        assert args.epochs is not None, "--stage generate requires --epochs"
        _stage_generate(args.seed, args.epochs)


if __name__ == "__main__":
    main()
