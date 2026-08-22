'''S6.5 of PLAN-TASK-2026-08-12-SDXL: the random-ten control for the seed-42 campaign.

The campaign measures ten entities that were chosen *because* Stable Diffusion 1.4 damaged them (the
target plus its nine strongest receivers). That selection makes one reading impossible: if those nine
barely move under Stable Diffusion XL, "Stable Diffusion XL spreads less interference" and "the
interference went somewhere we did not look" produce the same table. This control breaks the tie by
measuring ten entities nobody selected -- drawn at random from the same 100-entity people metadata,
excluding the target and the nine receivers -- under the same adapter, the same seed and the same
frozen generation configuration.

Only the FINAL checkpoint is generated (epoch 200, seed 42) plus each entity's off-baseline: 20
images. The question is whether collateral damage exists elsewhere at the end of training, not how it
evolves, and a full trajectory for ten more entities would cost thirteen times as much for a
refinement of an answer this already gives.

Four stages, each its own process (S2's one-pipeline-per-process constraint):

    python random_ten_control.py --stage draw                 # no GPU; writes the drawn set
    python random_ten_control.py --stage generate --epoch off
    python random_ten_control.py --stage generate --epoch 200
    python random_ten_control.py --stage score                # no Stable Diffusion XL; CLIP only

Run from this directory, GPU-capable interpreter, PYTHONPATH at the vision-unlearning repository
root, HF_HUB_DISABLE_XET=1.
'''
from __future__ import annotations

import argparse
import json
import random
import time
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
_SEED = 42
_EPOCH = 200

_OUT = cfg.OUT
_ENTITIES_PATH = _OUT / "random_ten_control_entities.json"
_MANIFEST_PATH = _OUT / f"random_ten_control_seed{_SEED}.json"
_GEN_DIR = _OUT / f"random_ten_control_seed{_SEED}"
_SCORES_PATH = _OUT / "clip_diff_random_ten_control.json"


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


def _append_manifest(rows: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if _MANIFEST_PATH.is_file():
        existing = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    existing.extend(rows)
    _MANIFEST_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _stage_generate(epoch_arg: str) -> None:
    '''Generates the ten control images for ONE label -- the off-baseline or epoch 200.

    The shape is the campaign runner's `_stage_generate`, for the same measured reasons: one Stable
    Diffusion XL pipeline per process, one `generate_dataset` call per entity with the generator
    reseeded, images already on disk skipped, manifest rows appended only once all ten exist. The
    prompts come from `clip_scoring.prompts_for`, which is the helper the campaign generation order
    uses, so a control image and a campaign image of the same entity would be asked for in exactly
    the same words.
    '''
    check_headroom()

    from vision_unlearning.utils.data_generation import generate_dataset

    entities = control_entities()
    own_prompt, _ = clip_scoring.prompts_for(entities)
    is_off = epoch_arg == "off"
    assert is_off or int(epoch_arg) == _EPOCH, \
        f"--epoch {epoch_arg!r}: this control generates only the off-baseline and epoch {_EPOCH}"
    label = "off" if is_off else f"epoch{_EPOCH}"

    already: set = set()
    if _MANIFEST_PATH.is_file():
        already = {row["epoch"] for row in json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))}
    if (None if is_off else _EPOCH) in already:
        print(json.dumps({"stage": "generate", "label": label, "rows_generated": 0,
                          "note": "nothing to do -- already in the manifest"}, indent=2))
        print(f"RANDOM_TEN_GENERATE_DONE label={label} images=0 (already in the manifest)")
        return

    lora_name: Optional[str] = None
    if not is_off:
        adapter_dir = _OUT / "campaign_model" / f"seed{_SEED}" / f"epoch-{_EPOCH}"
        adapter_file = adapter_dir / "pytorch_lora_weights.safetensors"
        assert adapter_file.is_file(), f"adapter missing for epoch {_EPOCH}: {adapter_file}"
        lora_name = str(adapter_dir)

    _GEN_DIR.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(_OUT / f"random_ten_control_{label}_monitor.log", interval_s=15.0)
    monitor.start()
    t0 = time.time()
    per_entity: List[Dict[str, Any]] = []
    pipeline: Optional[Any] = None
    try:
        for name in entities:
            filename = f"{label}_{name}_seed{_SEED}.png"
            path = _GEN_DIR / filename
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
                output_path=str(_GEN_DIR),
                filenames=[filename],
                seeds=[_SEED],
                batch_size=1,
                height=GENERATION_RESOLUTION,
                width=GENERATION_RESOLUTION,
                runtime=GENERATION_RUNTIME,
                **GENERATION_KWARGS,
            )
            assert path.is_file(), f"generate_dataset did not write {path}"
            per_entity.append({"entity": name, "seconds": round(time.time() - t_image, 1),
                               "skipped": False})
            print(f"{label} {name} seed {_SEED}: {per_entity[-1]['seconds']} s "
                  f"({len(per_entity)} of {len(entities)})", flush=True)
    finally:
        monitor.stop()
    seconds = round(time.time() - t0, 1)

    rows = [
        {"epoch": None if is_off else _EPOCH, "entity": name,
         "path": str(_GEN_DIR / f"{label}_{name}_seed{_SEED}.png"), "seed": _SEED}
        for name in entities
    ]
    _append_manifest(rows)

    print(json.dumps({
        "stage": "generate", "label": label, "rows_written": len(rows), "seconds": seconds,
        "per_entity": per_entity,
        "peak_vram_used_gb": round(monitor.peak_vram_used_gb, 3),
        "min_ram_free_gb": round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None,
    }, indent=2))
    print(f"RANDOM_TEN_GENERATE_DONE label={label} images={len(rows)}")


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


def _stage_score() -> None:
    '''Scores the 20 control images and prints the two counts that ARE the finding.

    The comparison is deliberately narrow: the number of control entities outside the noise floor at
    epoch 200, beside the number of selected receivers outside it at epoch 200, both read from the
    artifacts rather than recomputed. The campaign numbers come from `assets/clip_diff_campaign.json`,
    so this stage never re-scores a campaign image.
    '''
    entities = control_entities()
    rows = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    per_entity, trained_epochs, n_scored = clip_scoring.score_entities(rows, entities)
    assert trained_epochs == [_EPOCH], \
        f"expected only epoch {_EPOCH} in the control, got {trained_epochs}"

    floor = cfg.noise_floor_standard_deviation()
    control_diff = {name: per_entity[name]["trajectory"][0]["clip_diff"] for name in entities}

    campaign = json.loads((_OUT / "clip_diff_campaign.json").read_text(encoding="utf-8"))
    campaign_seed = campaign["per_seed"][str(_SEED)]
    target_name = campaign["target"]
    receiver_diff: Dict[str, float] = {}
    target_diff = float("nan")
    for name, payload in campaign_seed["per_entity"].items():
        final = [point for point in payload["trajectory"] if point["epoch"] == _EPOCH]
        assert len(final) == 1, f"campaign trajectory for {name} has no epoch {_EPOCH}"
        if name == target_name:
            target_diff = final[0]["clip_diff"]
        else:
            receiver_diff[name] = final[0]["clip_diff"]

    control_outside = _outside_floor(control_diff, floor)
    receivers_outside = _outside_floor(receiver_diff, floor)
    control_damaged = _below_floor(control_diff, floor)
    receivers_damaged = _below_floor(receiver_diff, floor)
    result = {
        "task": clip_scoring.TASK,
        "method": clip_scoring.METHOD,
        "model": clip_scoring.MODEL_ID,
        "seed": _SEED,
        "epoch": _EPOCH,
        "draw_seed": _DRAW_SEED,
        "noise_floor_standard_deviation": floor,
        "entities": entities,
        # The same envelope the campaign scores use (`per_seed -> {epochs, per_entity}`), so that one
        # reader -- the grid figure below all of this -- works on either artifact without sniffing
        # which shape it was handed.
        "per_seed": {str(_SEED): {"epochs": trained_epochs, "per_entity": per_entity}},
        "comparison_at_final_epoch": {
            "control_clip_diff": control_diff,
            "receiver_clip_diff": receiver_diff,
            "target_clip_diff": target_diff,
            "control_outside_floor": control_outside,
            "receivers_outside_floor": receivers_outside,
            "control_below_negative_floor": control_damaged,
            "receivers_below_negative_floor": receivers_damaged,
            "n_control": len(entities),
            "n_receivers": len(receiver_diff),
        },
    }
    _SCORES_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"images scored: {n_scored}; expected {len(entities)} x (1 off + 1 epoch) = {len(entities) * 2}")
    print(f"images scored equals expected: {n_scored == len(entities) * 2}")
    print(f"noise floor (one standard deviation, one entity over six seeds): {floor:.3f}")
    print(f"target {target_name}: clip_diff {target_diff:+.2f} at epoch {_EPOCH}")
    print(f"control entities outside the floor at epoch {_EPOCH}: {len(control_outside)} of "
          f"{len(entities)} {control_outside}")
    print(f"selected receivers outside the floor at epoch {_EPOCH}: {len(receivers_outside)} of "
          f"{len(receiver_diff)} {receivers_outside}")
    print(f"control entities BELOW the negative floor (the count that means damage): "
          f"{len(control_damaged)} of {len(entities)} {control_damaged}")
    print(f"selected receivers BELOW the negative floor: {len(receivers_damaged)} of "
          f"{len(receiver_diff)} {receivers_damaged}")
    for name in sorted(control_diff, key=lambda entity: control_diff[entity]):
        print(f"  control  {name:<28} clip_diff {control_diff[name]:+7.2f}")
    for name in sorted(receiver_diff, key=lambda entity: receiver_diff[entity]):
        print(f"  receiver {name:<28} clip_diff {receiver_diff[name]:+7.2f}")
    print(f"written: {_SCORES_PATH}")
    print(f"RANDOM_TEN_SCORE_DONE control_outside={len(control_outside)} of {len(entities)} "
          f"receivers_outside={len(receivers_outside)} of {len(receiver_diff)} "
          f"control_damaged={len(control_damaged)} receivers_damaged={len(receivers_damaged)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S6.5: the random-ten control for the seed-42 campaign.")
    parser.add_argument("--stage", choices=["draw", "generate", "score"], required=True)
    parser.add_argument("--epoch", type=str, default=None,
                        help="'off' or 200; required for --stage generate")
    args = parser.parse_args()

    if args.stage == "draw":
        _stage_draw()
    elif args.stage == "generate":
        assert args.epoch is not None, "--stage generate requires --epoch"
        _stage_generate(args.epoch)
    else:
        _stage_score()


if __name__ == "__main__":
    main()
