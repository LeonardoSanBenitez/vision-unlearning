'''
S4 of PLAN-TASK-2026-08-12-SDXL: choose the training hyperparameters, on valid 768-pixel images.

Trains the target `Mark_Philippoussis` for TEN epochs at a given learning rate, adapter rank, alpha and
forget weight, saving adapters at epochs 1, 3, 5 and 10; then renders three entities -- the target and
two receivers -- at those four checkpoints plus the base-model off-baseline, at the frozen generation
configuration of plan section 2.1; then reads `clip_diff = clip_on - clip_off` from the same
`MetricImageTextSimilarity(metrics=['clip'])` primitive `pipeline_06` and the every-epoch ablation
use, and builds a contact sheet. 15 images per run.

**The hyperparameters are four command-line numbers with defaults, not a named set.** The defaults are
the inherited Stable Diffusion 1.4 values (learning rate 6e-4, rank 16, alpha 4, forget weight 0.3),
which is the setting the plan's methodological-transfer claim rests on: keep them and the base model is
the only variable against the existing Stable Diffusion 1.4 curves. To try the colleague's tuned values
instead, pass them:

    python run_schedule_probe.py --stage train --learning-rate 1e-4 --lora-r 4 --forget-weight 0.5

Every artifact of a run is suffixed with a tag derived from those four numbers
(`lr6e-04_r16_a4_fw0.3`), so two settings never overwrite each other and the filename says what it was
without a lookup table anywhere.

**Ten epochs, not five.** The 2026-08-14 version of this probe trained five and judged the result
against a -5.0 bar. That was miscalibrated: the Stable Diffusion 1.4 reference trajectory for this
target does not cross -5 until epoch 10, so a five-epoch window judged a setting against a bar the
reference model itself needs ten epochs to clear. The window has to contain the reference's own
crossing point or the comparison means nothing.

**No numeric pass/fail gate.** The plan states how the choice is made and it is a human reading, in
this order: (1) the images at all four checkpoints -- does the target change, and toward the overwrite
concept rather than into noise; (2) `clip_diff` on the target at epoch 10 against the measured noise
floor, with epochs 1/3/5 read for shape only; (3) the receivers, to check the setting does not simply
destroy everything. This script therefore reports numbers and builds the contact sheet; it does not
decide. Prefer the inherited defaults whenever their target moves outside the floor. If nothing tried
moves the target, that is a finding about how far the methodology transfers, not something a third
setting invented here fixes.

**The two receivers are chosen by rule from the selection file, never retyped**: the most damaged and
the least damaged receiver by the Stable Diffusion 1.4 `clip_diff` the every-epoch ablation measured.
Two images per checkpoint cannot represent nine receivers, so they are the two ends of the range --
one where collateral damage is expected if the training works at all, one where destroying it would
mean the training destroys everything.

**Positive control, free.** The off-baselines are the base model at the frozen configuration, seed 42,
for entities the S1 gate already generated. The report stage diffs them against
`assets/campaign_seed42/off_<entity>_seed42.png` and reports the per-entity mean absolute difference.
It must land at the cross-process reproducibility noise of about 0.0003 of 255; anything larger means
this script is not asking for what the campaign asks for, and every number below it is about a
different configuration.

Stages, each its own process -- required, not stylistic: building a second Stable Diffusion XL
pipeline inside a process that already holds one drives free system memory to ~1.3 GB even after
`del` and `empty_cache()` (measured, S2/`check_generation_gate.py`).

    python run_schedule_probe.py --stage train
    python run_schedule_probe.py --stage generate --off
    python run_schedule_probe.py --stage generate --epoch 1
    python run_schedule_probe.py --stage generate --epoch 3
    python run_schedule_probe.py --stage generate --epoch 5
    python run_schedule_probe.py --stage generate --epoch 10
    python run_schedule_probe.py --stage report

Any hyperparameter arguments must be repeated identically on every stage of the same run, since they
are what identifies the run's artifacts. `run_s4_schedule_probe.sh` drives all seven in order, is
resumable, and takes the same arguments once.

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
from watchdog import ResourceMonitor, check_headroom

_OUT = cfg.OUT
_SEED = 42
_N_EPOCHS = 10
_CHECKPOINTS = [1, 3, 5, 10]

# The inherited Stable Diffusion 1.4 hyperparameters, unchanged. They are the defaults because they
# are the setting that supports the transfer claim; anything else is passed on the command line.
_DEFAULT_LEARNING_RATE = 6e-4
_DEFAULT_LORA_R = 16
_DEFAULT_LORA_ALPHA = 4
_DEFAULT_FORGET_WEIGHT = 0.3

# The S1 off-baselines this probe's own off-baselines are checked against (see the module docstring's
# positive control), and the tolerance that check applies. 0.0003 of 255 is the cross-process
# agreement measured in `assets/verify_refactored_base.json`; 0.01 is the same slack the S1 gate used.
_CONTROL_DIR = _OUT / "campaign_seed42"
_CONTROL_TOLERANCE = 0.01


def _run_tag(hyperparameters: Dict[str, Any]) -> str:
    '''The filename tag identifying one setting of the four hyperparameters, e.g. `lr6e-04_r16_a4_fw0.3`.

    Derived from the values themselves so that a reader of any artifact knows what produced it without
    consulting a table, and so that two settings cannot overwrite each other's files.
    '''
    return (f"lr{hyperparameters['learning_rate']:.0e}"
            f"_r{hyperparameters['lora_r']}"
            f"_a{hyperparameters['lora_alpha']}"
            f"_fw{hyperparameters['forget_weight']:g}")


def _paths(hyperparameters: Dict[str, Any]) -> Dict[str, Path]:
    tag = _run_tag(hyperparameters)
    return {
        "model_dir": _OUT / f"schedule_probe_model_{tag}",
        "gen_dir": _OUT / f"schedule_probe_generated_{tag}",
        "output_json": _OUT / f"schedule_probe_{tag}.json",
        "train_json": _OUT / f"schedule_probe_train_{tag}.json",
        "train_monitor_log": _OUT / f"schedule_probe_train_monitor_{tag}.log",
        "sheet": _OUT / f"schedule_probe_sheet_{tag}.png",
    }


def _probe_entities() -> List[Dict[str, Any]]:
    '''The target and the two receivers, in the campaign's own generation order.

    The receivers are picked by rule from the every-epoch selection: the most damaged and the least
    damaged by Stable Diffusion 1.4 `clip_diff`. `generation_order()` is already sorted ascending by
    that value, so they are its first and last non-target entries. Returning them in that same order
    keeps this probe's per-entity ordering a sub-sequence of the campaign's.
    '''
    order = cfg.generation_order()
    target = next(e for e in order if e["is_target"])
    receivers = [e for e in order if not e["is_target"]]
    chosen = {target["name"], receivers[0]["name"], receivers[-1]["name"]}
    selected = [e for e in order if e["name"] in chosen]
    assert len(selected) == 3, f"expected 3 probe entities, got {[e['name'] for e in selected]}"
    return selected


def _stage_train(hyperparameters: Dict[str, Any]) -> None:
    '''Trains 10 epochs at the given hyperparameters, saving adapters at epochs 1, 3, 5 and 10.'''
    check_headroom()

    from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation

    from step_check_support import StopAfterTraining, restore_post_training, stub_post_training

    paths = _paths(hyperparameters)
    target = next(e for e in _probe_entities() if e["is_target"])
    split_base = cfg.SPLIT_BASE / target["name"]
    assert (split_base / "train_forget").is_dir() and (split_base / "train_retain").is_dir(), \
        f"forget/retain splits missing for {cfg.TASK}/{target['name']} under {split_base}"

    model_dir = paths["model_dir"]
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    arguments = cfg.training_hyperparameters(
        output_dir=model_dir,
        split_base=split_base,
        overwrite_concept=target["overwrite_concept"],
        seed=_SEED,
        n_epochs=_N_EPOCHS,
        save_lora_at_epochs=_CHECKPOINTS,
        **hyperparameters,
    )

    monitor = ResourceMonitor(paths["train_monitor_log"], interval_s=15.0)
    original_unlearn_lora = stub_post_training()
    t0 = time.time()
    try:
        unlearner = UnlearnerLoraDistillation(**arguments)
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
        "run_tag": _run_tag(hyperparameters),
        "hyperparameters": hyperparameters,
        "target": target["name"],
        "overwrite_concept": target["overwrite_concept"],
        "resolution": cfg.TRAIN_RESOLUTION,
        "gradient_checkpointing": cfg.GRADIENT_CHECKPOINTING,
        "micro_conditioning_original_size": list(cfg.TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE),
        "micro_conditioning_target_size": list(cfg.TRAIN_MICRO_CONDITIONING_TARGET_SIZE),
        "epochs": _N_EPOCHS,
        "checkpoints_written": _CHECKPOINTS,
        "train_seconds": train_s,
        "seconds_per_epoch": round(train_s / _N_EPOCHS, 1),
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }
    paths["train_json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"SCHEDULE_PROBE_TRAIN_DONE tag={result['run_tag']} checkpoints={len(_CHECKPOINTS)} "
          f"seconds={train_s} peak_vram_used_gb={result['peak_vram_used_gb']}")


def _stage_generate(hyperparameters: Dict[str, Any], epoch: Optional[int]) -> None:
    '''Generates the three probe entities at ONE checkpoint (or the off-baseline), one call each.

    One process per checkpoint and one `generate_dataset` call per entity, for the two reasons plan
    section 2.1 and `run_campaign.py` record: a second pipeline in one process exhausts system
    memory, and at the frozen configuration each image is generated alone with the generator reseeded.
    The three calls share one pipeline. Images already on disk are skipped, so an abort costs the
    image in flight and nothing else.
    '''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    paths = _paths(hyperparameters)
    entities = _probe_entities()
    gen_dir = paths["gen_dir"]
    gen_dir.mkdir(parents=True, exist_ok=True)
    label = "off" if epoch is None else f"epoch{epoch}"
    lora_name = None if epoch is None else str(paths["model_dir"] / f"epoch-{epoch}")
    if lora_name is not None:
        adapter_file = Path(lora_name) / "pytorch_lora_weights.safetensors"
        assert adapter_file.is_file(), f"adapter missing for epoch {epoch}: {adapter_file}"

    tag = _run_tag(hyperparameters)
    monitor = ResourceMonitor(_OUT / f"schedule_probe_generate_{label}_{tag}_monitor.log", interval_s=15.0)
    monitor.start()
    t0 = time.time()
    per_entity: List[Dict[str, Any]] = []
    pipeline: Optional[Any] = None
    try:
        for entity in entities:
            filename = f"{label}_{entity['name']}_seed{_SEED}.png"
            path = gen_dir / filename
            if path.is_file():
                per_entity.append({"entity": entity["name"], "seconds": None, "skipped": True})
                print(f"already on disk, skipping: {filename}", flush=True)
                continue
            t_image = time.time()
            if pipeline is None:
                pipeline = cfg.build_generation_pipeline(lora_name)
            generate_dataset(
                model_base_name=None,
                lora_name=None,
                model_pipeline=pipeline,
                prompts=[entity["prompt"]],
                output_path=str(gen_dir),
                filenames=[filename],
                seeds=[_SEED],
                batch_size=1,
                height=cfg.GENERATION_RESOLUTION,
                width=cfg.GENERATION_RESOLUTION,
                runtime=cfg.GENERATION_RUNTIME,
                **cfg.GENERATION_KWARGS,
            )
            assert path.is_file(), f"generate_dataset did not write {path}"
            per_entity.append({"entity": entity["name"], "seconds": round(time.time() - t_image, 1),
                               "skipped": False})
            print(f"{label} {entity['name']}: {per_entity[-1]['seconds']} s "
                  f"({len(per_entity)} of {len(entities)})", flush=True)
    finally:
        monitor.stop()
    seconds = round(time.time() - t0, 1)

    print(json.dumps({
        "stage": "generate", "run_tag": tag, "label": label, "epoch": epoch,
        "images": len(entities), "seconds": seconds, "per_entity": per_entity,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }, indent=2))
    print(f"SCHEDULE_PROBE_GENERATE_DONE tag={tag} label={label} images={len(entities)}")


def _off_baseline_control(entities: List[Dict[str, Any]], gen_dir: Path) -> List[Dict[str, Any]]:
    '''Diffs this probe's off-baselines against the S1 campaign off-baselines of the same entities.

    Both are the base model at the frozen configuration and seed 42, so they must agree to the
    cross-process reproducibility noise. This is the one check in the stage whose correct answer is
    known before it runs.
    '''
    from image_difference import mean_abs_difference

    rows: List[Dict[str, Any]] = []
    for entity in entities:
        ours = gen_dir / f"off_{entity['name']}_seed{_SEED}.png"
        theirs = _CONTROL_DIR / f"off_{entity['name']}_seed{_SEED}.png"
        difference = mean_abs_difference(ours, theirs)
        rows.append({
            "entity": entity["name"],
            "campaign_image": str(theirs),
            # None means one of the two files is missing or they differ in shape -- an answer to
            # report, never something to compare against a tolerance.
            "mean_absolute_difference": difference,
            "within_tolerance": None if difference is None else difference <= _CONTROL_TOLERANCE,
        })
    return rows


def _stage_report(hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    '''Scores every image, runs the off-baseline control, writes the JSON and builds the contact sheet.'''
    from PIL import Image
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    paths = _paths(hyperparameters)
    gen_dir = paths["gen_dir"]
    entities = _probe_entities()
    noise_floor = cfg.noise_floor_standard_deviation()

    metric = MetricImageTextSimilarity(metrics=["clip"])
    rows: List[Dict[str, Any]] = []
    for entity in entities:
        off_path = gen_dir / f"off_{entity['name']}_seed{_SEED}.png"
        assert off_path.is_file(), f"missing off-baseline image: {off_path}"
        clip_off = metric.score_batch_same_text([Image.open(off_path).convert("RGB")], entity["prompt"])[0]["clip"]

        trajectory: List[Dict[str, Any]] = []
        for n in _CHECKPOINTS:
            on_path = gen_dir / f"epoch{n}_{entity['name']}_seed{_SEED}.png"
            assert on_path.is_file(), f"missing epoch-{n} image: {on_path}"
            clip_on = metric.score_batch_same_text([Image.open(on_path).convert("RGB")], entity["prompt"])[0]["clip"]
            clip_diff = clip_on - clip_off
            trajectory.append({
                "epoch": n, "clip_on": round(clip_on, 3), "clip_diff": round(clip_diff, 3),
                "outside_noise_floor": abs(clip_diff) > noise_floor,
            })
        rows.append({
            "entity": entity["name"], "prompt": entity["prompt"], "is_target": entity["is_target"],
            "sd14_clip_diff": round(entity["sort_value"], 3),
            "clip_off": round(clip_off, 3), "trajectory": trajectory,
        })

    result = {
        "stage": "report",
        "task": cfg.TASK,
        "method": cfg.METHOD,
        "seed": _SEED,
        "epochs_trained": _N_EPOCHS,
        "checkpoints": _CHECKPOINTS,
        "generation_resolution": cfg.GENERATION_RESOLUTION,
        "run_tag": _run_tag(hyperparameters),
        "hyperparameters": hyperparameters,
        "noise_floor_standard_deviation": noise_floor,
        "noise_floor_source": str(cfg.NOISE_FLOOR_SOURCE),
        "off_baseline_control": _off_baseline_control(entities, gen_dir),
        "entities": rows,
        "contact_sheet": str(paths["sheet"]),
    }
    paths["output_json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    _build_contact_sheet(hyperparameters, result)
    print(json.dumps(result, indent=2))
    print(f"SCHEDULE_PROBE_REPORT_DONE tag={result['run_tag']} entities={len(rows)} "
          f"checkpoints={len(_CHECKPOINTS)} sheet={paths['sheet']}")
    return result


def _build_contact_sheet(hyperparameters: Dict[str, Any], result: Dict[str, Any]) -> None:
    '''Rows = the three entities, columns = off-baseline then each checkpoint, one image per cell.

    Cold parameter-dense titles, the number under the image it was computed from (the user's own S9
    requirement, applied here because this sheet is what the hyperparameter choice is made on).
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    paths = _paths(hyperparameters)
    gen_dir = paths["gen_dir"]
    entities = result["entities"]
    columns = ["off"] + [f"epoch{n}" for n in _CHECKPOINTS]

    figure, axes = plt.subplots(len(entities), len(columns),
                                figsize=(3.0 * len(columns), 3.35 * len(entities)))
    for row_index, row in enumerate(entities):
        by_epoch = {item["epoch"]: item for item in row["trajectory"]}
        for column_index, column in enumerate(columns):
            axis = axes[row_index][column_index]
            filename = f"{column}_{row['entity']}_seed{_SEED}.png"
            axis.imshow(Image.open(gen_dir / filename).convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            if column == "off":
                caption = f"off, clip {row['clip_off']:.2f}"
            else:
                item = by_epoch[int(column.removeprefix("epoch"))]
                caption = (f"epoch {item['epoch']}, clip {item['clip_on']:.2f}\n"
                           f"clip_diff {item['clip_diff']:+.2f}")
            axis.set_title(caption, fontsize=9)
            if column_index == 0:
                role = "target" if row["is_target"] else "receiver"
                axis.set_ylabel(f"{row['entity'].replace('_', ' ')}\n{role}", fontsize=9)

    figure.suptitle(
        f"Stable Diffusion XL, SPARE: learning rate {hyperparameters['learning_rate']}, "
        f"rank {hyperparameters['lora_r']}, alpha {hyperparameters['lora_alpha']}, "
        f"forget weight {hyperparameters['forget_weight']}, "
        f"{_N_EPOCHS} epochs, seed {_SEED}, {cfg.GENERATION_RESOLUTION} pixels, "
        f"clip_diff noise floor {result['noise_floor_standard_deviation']:.2f}",
        fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(paths["sheet"], dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="S4 schedule probe: choose the training hyperparameters.")
    parser.add_argument("--stage", choices=["train", "generate", "report", "tag"], required=True,
                        help="'tag' prints the hyperparameter tag this run's artifacts carry and exits; "
                             "it is how the launcher learns the tag without a second copy of the naming rule")
    parser.add_argument("--learning-rate", type=float, default=_DEFAULT_LEARNING_RATE)
    parser.add_argument("--lora-r", type=int, default=_DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=_DEFAULT_LORA_ALPHA)
    parser.add_argument("--forget-weight", type=float, default=_DEFAULT_FORGET_WEIGHT)
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch for --stage generate")
    parser.add_argument("--off", action="store_true", help="generate the off (base-model) baseline instead")
    args = parser.parse_args()

    hyperparameters: Dict[str, Any] = {
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "forget_weight": args.forget_weight,
    }

    if args.stage == "tag":
        print(_run_tag(hyperparameters))
    elif args.stage == "train":
        _stage_train(hyperparameters)
    elif args.stage == "generate":
        if args.off == (args.epoch is not None):
            raise SystemExit("--stage generate needs exactly one of --off or --epoch N")
        if args.epoch is not None and args.epoch not in _CHECKPOINTS:
            raise SystemExit(f"--epoch {args.epoch} is not one of the checkpoints {_CHECKPOINTS}")
        _stage_generate(hyperparameters, None if args.off else args.epoch)
    else:
        _stage_report(hyperparameters)


if __name__ == "__main__":
    main()
