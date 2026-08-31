'''S6.5 of PLAN-TASK-2026-08-12-SDXL: the random-ten control, measured exactly like the campaign.

The campaign measures ten entities that were chosen *because* Stable Diffusion 1.4 damaged them (the
target plus its nine strongest receivers). That selection makes one reading impossible: if those nine
barely move under Stable Diffusion XL, "Stable Diffusion XL spreads less interference" and "the
interference went somewhere we did not look" produce the same table. This control breaks the tie by
measuring ten entities nobody selected -- drawn at random from the same 100-entity people metadata,
excluding the target and the nine receivers -- under the same adapters, the same seeds and the same
frozen generation configuration.

**It generates what the campaign generates**: the off-baseline plus every one of the thirteen saved
checkpoints, at both seeds, so the control set can be drawn in the same grids and curves and read
against the same noise floor. The first version of this script generated the final checkpoint alone.
That was enough to show that collateral damage exists on unselected entities, and not enough to say
anything about the transient behaviour that dominates the middle of training -- which is where this
campaign's largest effects are.

Stages, each its own process (S2's one-pipeline-per-process constraint):

    python random_ten_control.py --stage draw                        # no GPU; writes the drawn set
    python random_ten_control.py --stage labels                      # prints off, then each checkpoint
    python random_ten_control.py --stage generate --seed 42 --epoch off
    python random_ten_control.py --stage generate --seed 42 --epoch 5
    python random_ten_control.py --stage score --seeds 42,43         # no Stable Diffusion XL; CLIP only

Run from this directory, GPU-capable interpreter, PYTHONPATH at the vision-unlearning repository
root, HF_HUB_DISABLE_XET=1.
'''
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import campaign_configuration as cfg
import clip_scoring
from campaign_configuration import (
    GENERATION_KWARGS,
    GENERATION_RESOLUTION,
    GENERATION_RUNTIME,
    build_generation_pipeline,
    generation_order,
)
from watchdog import ResourceMonitor, check_headroom

# The drawing seed is a module constant so the control set is reproducible from the repository alone,
# and it is written into the artifact beside the names it produced. 20260812 is the date the control
# was approved in the plan; any fixed integer would do, and the point is only that it is not chosen
# after seeing which entities came out.
_DRAW_SEED = 20260812
_N_CONTROL = 10

_OUT = cfg.OUT
_ENTITIES_PATH = _OUT / "random_ten_control_entities.json"
_SCORES_PATH = _OUT / "clip_diff_random_ten_control.json"


def _manifest_path(seed: int) -> Path:
    return _OUT / f"random_ten_control_seed{seed}.json"


def _generated_directory(seed: int) -> Path:
    return _OUT / f"random_ten_control_seed{seed}"


def _selected_entities() -> List[str]:
    '''The ten entities the campaign already measures: the target and its nine receivers.'''
    return [entity["name"] for entity in generation_order()]


def _stage_draw() -> None:
    '''Draws the control set and writes it. Deterministic, cheap, and refuses to contradict itself.

    Sorting the pool by name before drawing makes the set depend on the metadata CONTENT and not on
    its row order, so a later regeneration of the metadata file that preserves the same 100 entities
    reproduces the same ten.
    '''
    from vision_unlearning.datasets.testbed import get_metadata_filtered

    metadata = get_metadata_filtered(clip_scoring.TASK, base_folder=str(cfg.ICARE_DIR / "assets"))
    all_names = sorted(entry["name"] for entry in metadata)
    selected = _selected_entities()
    missing_from_metadata = sorted(set(selected) - set(all_names))
    assert not missing_from_metadata, \
        f"campaign entities absent from the people metadata: {missing_from_metadata}"

    pool = [name for name in all_names if name not in set(selected)]
    drawn = sorted(random.Random(_DRAW_SEED).sample(pool, _N_CONTROL))
    assert not set(drawn) & set(selected), "the control set overlaps the campaign selected ten"

    payload = {
        "draw_seed": _DRAW_SEED,
        "task": clip_scoring.TASK,
        "n_metadata": len(all_names),
        "n_excluded": len(selected),
        "excluded": selected,
        "pool_size": len(pool),
        "entities": drawn,
        "prompts": clip_scoring.prompts_for(drawn)[0],
    }
    if _ENTITIES_PATH.is_file():
        existing = json.loads(_ENTITIES_PATH.read_text(encoding="utf-8"))
        assert existing["entities"] == drawn, (
            f"{_ENTITIES_PATH} holds a different control set than this draw produces "
            f"({existing['entities']} against {drawn}); the images on disk belong to the stored set."
        )
    _ENTITIES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"RANDOM_TEN_DRAW_DONE entities={len(drawn)} pool={len(pool)} = {len(all_names)} metadata "
          f"minus {len(selected)} campaign entities")


def control_entities() -> List[str]:
    '''The drawn control set, read back from the artifact the draw stage wrote. Never re-drawn here.'''
    assert _ENTITIES_PATH.is_file(), f"run --stage draw first: {_ENTITIES_PATH} does not exist"
    payload = json.loads(_ENTITIES_PATH.read_text(encoding="utf-8"))
    entities: List[str] = payload["entities"]
    assert len(entities) == _N_CONTROL, f"expected {_N_CONTROL} control entities, got {len(entities)}"
    return entities


def _stage_labels() -> None:
    '''Prints the generation labels in order -- `off`, then every checkpoint -- one per line.

    The same device `run_campaign.py` uses, and for the same reason: a shell driver walks the whole
    set one epoch per process without a hand-written epoch list, and each stage gets a log named after
    the epoch it generated rather than after the selector that asked for it.
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
    for epoch in cfg.checkpoint_list():
        print(epoch)


def _append_manifest(seed: int, rows: List[Dict[str, Any]]) -> None:
    path = _manifest_path(seed)
    existing: List[Dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.extend(rows)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _stage_generate(seed: int, epoch_arg: str) -> None:
    '''Generates the ten control images for ONE label -- the off-baseline or one checkpoint.

    The shape is the campaign runner's `_stage_generate`, for the same measured reasons: one Stable
    Diffusion XL pipeline per process, one `generate_dataset` call per entity with the generator
    reseeded, images already on disk skipped, manifest rows appended only once all ten exist. The
    prompts come from `clip_scoring.prompts_for`, the helper the campaign generation order uses, so a
    control image and a campaign image of the same entity are asked for in exactly the same words.

    A label the manifest already holds prints the completion marker and exits, so a driver can walk
    the whole checkpoint list after an interruption and pay only for what is missing.
    '''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    entities = control_entities()
    own_prompt, _ = clip_scoring.prompts_for(entities)
    checkpoints = cfg.checkpoint_list()
    is_off = epoch_arg == "off"
    epoch: Optional[int] = None if is_off else int(epoch_arg)
    assert is_off or epoch in checkpoints, \
        f"--epoch {epoch_arg!r} is not in the checkpoint list {checkpoints}"
    label = "off" if is_off else f"epoch{epoch}"

    manifest = _manifest_path(seed)
    already: set = set()
    if manifest.is_file():
        already = {row["epoch"] for row in json.loads(manifest.read_text(encoding="utf-8"))}
    if epoch in already:
        print(json.dumps({"stage": "generate", "seed": seed, "label": label, "rows_generated": 0,
                          "note": "nothing to do -- already in the manifest"}, indent=2))
        print(f"RANDOM_TEN_GENERATE_DONE seed={seed} label={label} images=0")
        return

    lora_name: Optional[str] = None
    if not is_off:
        adapter_dir = _OUT / "campaign_model" / f"seed{seed}" / f"epoch-{epoch}"
        adapter_file = adapter_dir / "pytorch_lora_weights.safetensors"
        assert adapter_file.is_file(), f"adapter missing for epoch {epoch}, seed {seed}: {adapter_file}"
        lora_name = str(adapter_dir)

    generated_directory = _generated_directory(seed)
    generated_directory.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(_OUT / f"random_ten_control_seed{seed}_{label}_monitor.log",
                              interval_s=15.0)
    monitor.start()
    t0 = time.time()
    per_entity: List[Dict[str, Any]] = []
    pipeline: Optional[Any] = None
    try:
        for name in entities:
            filename = f"{label}_{name}_seed{seed}.png"
            path = generated_directory / filename
            if path.is_file():
                per_entity.append({"entity": name, "seconds": None, "skipped": True})
                print(f"already on disk, skipping: {filename}", flush=True)
                continue
            t_image = time.time()
            if pipeline is None:
                pipeline = build_generation_pipeline(lora_name)
            generate_dataset(
                model_base_name=None,
                lora_name=None,
                model_pipeline=pipeline,
                prompts=[own_prompt[name]],
                output_path=str(generated_directory),
                filenames=[filename],
                seeds=[seed],
                batch_size=1,
                height=GENERATION_RESOLUTION,
                width=GENERATION_RESOLUTION,
                runtime=GENERATION_RUNTIME,
                **GENERATION_KWARGS,
            )
            assert path.is_file(), f"generate_dataset did not write {path}"
            per_entity.append({"entity": name, "seconds": round(time.time() - t_image, 1),
                               "skipped": False})
            print(f"{label} {name} seed {seed}: {per_entity[-1]['seconds']} s "
                  f"({len(per_entity)} of {len(entities)})", flush=True)
    finally:
        monitor.stop()
    seconds = round(time.time() - t0, 1)

    rows = [
        {"epoch": epoch, "entity": name,
         "path": str(generated_directory / f"{label}_{name}_seed{seed}.png"), "seed": seed}
        for name in entities
    ]
    _append_manifest(seed, rows)

    print(json.dumps({
        "stage": "generate", "seed": seed, "label": label, "rows_written": len(rows),
        "seconds": seconds, "per_entity": per_entity,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }, indent=2))
    print(f"RANDOM_TEN_GENERATE_DONE seed={seed} label={label} images={len(rows)}")


def _outside_floor(values: Dict[str, float], floor: float) -> List[str]:
    '''The entities whose `clip_diff` magnitude exceeds the noise floor, most negative first.'''
    return [name for name, value in sorted(values.items(), key=lambda item: item[1])
            if abs(value) > floor]


def _below_floor(values: Dict[str, float], floor: float) -> List[str]:
    '''The entities whose `clip_diff` is more NEGATIVE than the floor, most negative first.

    This is the count that means damage, and it is not the same as the two-sided one above. A
    `clip_diff` above +floor says the adapted image agrees with the entity's prompt BETTER than the
    original did, which is not collateral damage however far outside the floor it sits -- and reading
    the contact sheet is what makes that obvious: at epoch 200 the two control entities with the
    largest positive values are plainly still themselves, while the one at -9.82 has been replaced by
    a child. Both counts are reported; only this one is evidence of interference.
    '''
    return [name for name, value in sorted(values.items(), key=lambda item: item[1])
            if value < -floor]


def _worst(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''The most negative point of one trajectory, with the epoch it happened at.'''
    point = min(trajectory, key=lambda item: item["clip_diff"])
    return {"epoch": point["epoch"], "clip_diff": point["clip_diff"]}


def _stage_score(seeds: List[int]) -> None:
    '''Scores every control image and prints the counts that ARE the finding, per seed.

    Two counts per seed, and the pair is the point: how many control entities are damaged at the last
    checkpoint, and how many were damaged at their own worst checkpoint. The campaign's numbers come
    from `assets/clip_diff_campaign.json` rather than being recomputed, so no campaign image is ever
    re-scored here.
    '''
    entities = control_entities()
    floor = cfg.noise_floor_standard_deviation()
    campaign = json.loads((_OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    target_name = campaign["target"]

    result: Dict[str, Any] = {
        "task": clip_scoring.TASK, "method": clip_scoring.METHOD, "model": clip_scoring.MODEL_ID,
        "draw_seed": _DRAW_SEED, "noise_floor_standard_deviation": floor,
        "entities": entities, "seeds": seeds, "per_seed": {}, "comparison": {},
    }
    total_scored = 0
    for seed in seeds:
        manifest = _manifest_path(seed)
        assert manifest.is_file(), f"no control manifest for seed {seed}: {manifest}"
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        per_entity, trained_epochs, n_scored = clip_scoring.score_entities(rows, entities)
        total_scored += n_scored
        result["per_seed"][str(seed)] = {"epochs": trained_epochs, "per_entity": per_entity}

        final_epoch = trained_epochs[-1]
        control_final = {name: per_entity[name]["trajectory"][-1]["clip_diff"] for name in entities}
        control_worst = {name: _worst(per_entity[name]["trajectory"])["clip_diff"] for name in entities}

        campaign_block = campaign["per_seed"][str(seed)]["per_entity"]
        receiver_final: Dict[str, float] = {}
        receiver_worst: Dict[str, float] = {}
        for name, payload in campaign_block.items():
            if name == target_name:
                continue
            receiver_final[name] = payload["trajectory"][-1]["clip_diff"]
            receiver_worst[name] = _worst(payload["trajectory"])["clip_diff"]

        comparison = {
            "final_epoch": final_epoch,
            "control_final": control_final, "control_worst": control_worst,
            "receiver_final": receiver_final, "receiver_worst": receiver_worst,
            "control_damaged_at_final": _below_floor(control_final, floor),
            "control_damaged_at_worst": _below_floor(control_worst, floor),
            "receivers_damaged_at_final": _below_floor(receiver_final, floor),
            "receivers_damaged_at_worst": _below_floor(receiver_worst, floor),
            "control_outside_floor_at_final": _outside_floor(control_final, floor),
        }
        result["comparison"][str(seed)] = comparison

        print(f"--- seed {seed}: {len(trained_epochs)} checkpoints, {len(entities)} control entities")
        print(f"damaged (clip_diff below -{floor:.3f}) at the worst checkpoint: "
              f"control {len(comparison['control_damaged_at_worst'])} of {len(entities)}, "
              f"selected receivers {len(comparison['receivers_damaged_at_worst'])} of {len(receiver_final)}")
        print(f"damaged at epoch {final_epoch}: "
              f"control {len(comparison['control_damaged_at_final'])} of {len(entities)}, "
              f"selected receivers {len(comparison['receivers_damaged_at_final'])} of {len(receiver_final)}")
        for name in sorted(control_worst, key=lambda item: control_worst[item]):
            worst = _worst(per_entity[name]["trajectory"])
            print(f"  control  {name:<28} worst {worst['clip_diff']:+7.2f} at epoch "
                  f"{str(worst['epoch']):<4} final {control_final[name]:+7.2f}")

    _SCORES_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    expected = sum(len(entities) * (1 + len(result["per_seed"][str(seed)]["epochs"]))
                   for seed in seeds)
    print(f"images scored: {total_scored}; expected {expected}")
    print(f"images scored equals expected: {total_scored == expected}")
    print(f"written: {_SCORES_PATH}")
    print(f"RANDOM_TEN_SCORE_DONE images={total_scored} seeds={sorted(result['per_seed'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S6.5: the random-ten control, measured like the campaign.")
    parser.add_argument("--stage", choices=["draw", "labels", "generate", "score"], required=True)
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43])
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds for --stage score; defaults to --seed")
    parser.add_argument("--epoch", type=str, default=None,
                        help="'off' or a checkpoint epoch; required for --stage generate")
    args = parser.parse_args()

    if args.stage == "draw":
        _stage_draw()
    elif args.stage == "labels":
        _stage_labels()
    elif args.stage == "generate":
        assert args.epoch is not None, "--stage generate requires --epoch"
        _stage_generate(args.seed, args.epoch)
    else:
        seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else [args.seed]
        _stage_score(seeds)


if __name__ == "__main__":
    main()
