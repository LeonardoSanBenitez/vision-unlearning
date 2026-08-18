'''Overnight evidence run for the Stable Diffusion XL generation defect (2026-08-15).

Authorised by the user: "if u wnat to keep some simple things runnign just o see if something brings new
evidence, like generating lots iamges with different parameters, feel free to do so."

Nothing here fixes anything or regenerates the campaign -- those are tomorrow's decisions. This only
produces evidence, and it is ordered so that the blocks which would most change tomorrow's decisions run
FIRST. Each block writes its own JSON as soon as it finishes, so an interruption costs only the block in
flight.

Every image is 512 pixels. 1024 is settled as infeasible on this card (HIP launch failure, 2026-08-15)
and is deliberately not retried.

BLOCK 1 -- corrected off-baselines, 10 entities x 5 seeds, each generated ALONE.
    Useful whatever tomorrow decides: it is the multi-seed baseline that fixes `clip_diff`'s one-sample
    denominator, and it re-measures the noise floor on images that actually depict the entities. 50 images.

BLOCK 2 -- the position sweep, and the cleanest test in the file.
    One call whose prompt list is the SAME prompt repeated ten times. Content is therefore held exactly
    constant and only the position in the call varies. If position is the defect, image 0 is good and
    images 1-9 degrade. 3 entities x 10 = 30 images.

BLOCK 3 -- the same repeated-prompt call, but the generator is RESEEDED before every prompt.
    Separates two explanations that block 2 cannot: "the generator's later draws are bad" against
    "something in the pipeline degrades after the first call in a process". If block 3 is clean while
    block 2 degrades, it is the generator state and the fix is a reseed (or one call per entity). If
    block 3 degrades too, it is pipeline or process state and the fix is different. 30 images.

BLOCK 4 -- parameter grid on entities generated alone: guidance 5.0 vs 7.5 x original_size default vs
    1024. Re-measures yesterday's micro-conditioning finding in the CORRECT regime, since it was first
    measured on batched, degenerate renders. 4 entities x 4 = 16 images.

126 images, roughly 45 minutes at the measured ~20 s each.

    python overnight_evidence.py
'''
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_DIAG = _OUT / "overnight_evidence"
_RESOLUTION = 512
_BASELINE_SEEDS = [42, 43, 44, 45, 46]
_SWEEP_ENTITIES = ["Mark_Philippoussis", "Juan_Carlos_Ferrero", "Andy_Roddick"]
_GRID_ENTITIES = ["Mark_Philippoussis", "Juan_Carlos_Ferrero", "Andy_Roddick", "Ian_Thorpe"]


def _seed_everything(seed: int) -> Any:
    """The campaign's seeding site, replayed. Returns the generator the pipeline call should use."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cuda").manual_seed(seed)


def main() -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    parser = argparse.ArgumentParser(description="Overnight evidence for the generation defect.")
    parser.add_argument("--blocks", default="1,2,3,4")
    args = parser.parse_args()
    blocks = {int(b) for b in args.blocks.split(",")}

    check_headroom()
    _DIAG.mkdir(parents=True, exist_ok=True)

    order = _generation_order()
    prompt_of = {entry["name"]: entry["prompt"] for entry in order}
    names = [entry["name"] for entry in order]

    monitor = ResourceMonitor(_OUT / "overnight_evidence_monitor.log", interval_s=15.0)
    monitor.start()
    results: Dict[str, Any] = {"resolution": _RESOLUTION, "model": _MODEL_ID, "blocks": {}}

    def save(block: str, payload: Dict[str, Any], seconds: float, images: int) -> None:
        results["blocks"][block] = {"seconds": round(seconds, 1), "images": images, **payload}
        results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                      if monitor.min_ram_free_gb != float("inf") else None)
        (_OUT / "overnight_evidence.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"BLOCK {block} DONE: {images} images in {seconds:.1f} s", flush=True)

    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, safety_checker=None, variant=_VARIANT,
        ).to("cuda")

        # --- BLOCK 1: corrected off-baselines, every entity alone, five seeds -------------------- #
        if 1 in blocks:
            t0 = time.time()
            per_seed: Dict[str, Any] = {}
            count = 0
            for seed in _BASELINE_SEEDS:
                per_entity: Dict[str, str] = {}
                for name in names:
                    generator = _seed_everything(seed)
                    image = pipeline([prompt_of[name]], generator=generator,
                                     height=_RESOLUTION, width=_RESOLUTION).images[0]
                    path = _DIAG / f"baseline_alone_{name}_seed{seed}.png"
                    image.save(path, "PNG")
                    per_entity[name] = str(path)
                    count += 1
                per_seed[str(seed)] = per_entity
                print(f"block 1: seed {seed} done ({count} images so far)", flush=True)
            save("1_baselines_alone", {"seeds": _BASELINE_SEEDS, "per_seed": per_seed,
                                       "note": "each image generated alone, so every entity is at position 0"},
                 time.time() - t0, count)

        # --- BLOCK 2: same prompt ten times in ONE call; only position varies -------------------- #
        if 2 in blocks:
            t0 = time.time()
            per_entity_positions: Dict[str, Any] = {}
            count = 0
            for name in _SWEEP_ENTITIES:
                generator = _seed_everything(42)
                repeated = [prompt_of[name]] * 10
                paths: List[str] = []
                for position, prompt in enumerate(repeated):
                    image = pipeline([prompt], generator=generator,
                                     height=_RESOLUTION, width=_RESOLUTION).images[0]
                    path = _DIAG / f"position_{position:02d}_{name}_seed42.png"
                    image.save(path, "PNG")
                    paths.append(str(path))
                    count += 1
                per_entity_positions[name] = paths
                print(f"block 2: {name} done ({count} images so far)", flush=True)
            save("2_position_sweep", {"seed": 42, "per_entity_by_position": per_entity_positions,
                                      "note": "identical prompt at all ten positions, ONE generator advanced "
                                              "across them, exactly as generate_dataset does"},
                 time.time() - t0, count)

        # --- BLOCK 3: the same, but reseeding before every prompt -------------------------------- #
        if 3 in blocks:
            t0 = time.time()
            per_entity_reseeded: Dict[str, Any] = {}
            count = 0
            for name in _SWEEP_ENTITIES:
                paths = []
                for position in range(10):
                    generator = _seed_everything(42)  # the only difference from block 2
                    image = pipeline([prompt_of[name]], generator=generator,
                                     height=_RESOLUTION, width=_RESOLUTION).images[0]
                    path = _DIAG / f"reseeded_{position:02d}_{name}_seed42.png"
                    image.save(path, "PNG")
                    paths.append(str(path))
                    count += 1
                per_entity_reseeded[name] = paths
                print(f"block 3: {name} done ({count} images so far)", flush=True)
            save("3_reseeded_control", {"seed": 42, "per_entity_by_call_index": per_entity_reseeded,
                                        "note": "same ten calls in the same process, generator reseeded each "
                                                "time; all ten should be identical if only generator state matters"},
                 time.time() - t0, count)

        # --- BLOCK 4: parameter grid, entities alone --------------------------------------------- #
        if 4 in blocks:
            t0 = time.time()
            grid: Dict[str, Any] = {}
            count = 0
            for name in _GRID_ENTITIES:
                per_setting: Dict[str, str] = {}
                for guidance in (5.0, 7.5):
                    for original in (None, (1024, 1024)):
                        extra: Dict[str, Any] = {"guidance_scale": guidance}
                        if original is not None:
                            extra["original_size"] = original
                        label = f"guidance{guidance}_original{'default' if original is None else '1024'}"
                        generator = _seed_everything(42)
                        image = pipeline([prompt_of[name]], generator=generator,
                                         height=_RESOLUTION, width=_RESOLUTION, **extra).images[0]
                        path = _DIAG / f"grid_{label}_{name}_seed42.png"
                        image.save(path, "PNG")
                        per_setting[label] = str(path)
                        count += 1
                grid[name] = per_setting
                print(f"block 4: {name} done ({count} images so far)", flush=True)
            save("4_parameter_grid", {"seed": 42, "per_entity": grid,
                                      "note": "each generated alone; guidance and micro-conditioning varied"},
                 time.time() - t0, count)
    finally:
        monitor.stop()

    total = sum(block["images"] for block in results["blocks"].values())
    print(f"OVERNIGHT_EVIDENCE_DONE blocks={sorted(results['blocks'])} total_images={total}")
    print(f"peak video memory {results.get('peak_vram_used_gb')} GB, "
          f"minimum free system memory {results.get('min_ram_free_gb')} GB")
    print(f"written: {_OUT / 'overnight_evidence.json'}")


if __name__ == "__main__":
    main()
