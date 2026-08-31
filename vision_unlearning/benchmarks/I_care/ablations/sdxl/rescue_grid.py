'''Does Stable Diffusion XL's size micro-conditioning rescue the seeds whose renders collapse at 512?

WHY THIS RUN EXISTS. Block 1 of the overnight evidence run generated the same ten entities ALONE --
so every one of them at draw index 0, the position the proposed fix gives everybody -- at five seeds.
Seed 42's ten renders are photographic; seed 43's ten and seed 45's ten are not. The collapse is
therefore NOT a property of the draw index and is not repaired by giving every entity index 0: it is
a property of the initial noise sample, and at 512 pixels most samples lose. That kills the standing
explanation and the standing fix, and it makes the question "what makes a 512 render collapse".

THE HYPOTHESIS UNDER TEST. Stable Diffusion XL is conditioned on the size of the image it is being
asked for (its paper's section 2.2; the diffusers docstring at
`pipeline_stable_diffusion_xl.py:959` says `original_size` "defaults to `(height, width)` if not
specified"). Asking for 512 pixels therefore also TELLS the model "this is a 512-pixel image", which
in its training data meant a small, low-quality picture -- and flat posterised vector art is exactly
what the collapsed renders look like. If that is the mechanism, then declaring `original_size` (and
`target_size`) as 1024 while still rendering 512 pixels should rescue the losing seeds, at no extra
memory cost, because only three numbers in a conditioning vector change.

DESIGN. 3 entities x 3 seeds x 4 conditions = 36 images, every one generated alone with the
generator reseeded, at 512 pixels.

* seeds: 42 (the seed whose block-1 renders are all good -- the control that must NOT be broken by
  the conditioning change), 43 (the campaign's own second seed, all ten collapsed) and 45 (the worst
  of the five, all ten collapsed).
* conditions:
    - `campaign_defaults`            guidance 5.0, no size arguments -- what the campaign did.
    - `original1024`                 original_size=(1024, 1024) only.
    - `original1024_target1024`      also target_size=(1024, 1024).
    - `original1024_target1024_g7.5` the same, at guidance 7.5.

HARNESS CONTROL. `campaign_defaults` regenerates images that already exist as block 1's
`baseline_alone_<entity>_seed<seed>.png`. The script diffs the two and prints the mean absolute pixel
difference per entity, so a difference caused by this harness cannot be mistaken for a difference
caused by a condition. The two determinism flags `generate_dataset` sets are set here as well, which
block 1 did not do, so a small non-zero residual is expected and its size is the thing to read.

Runs under the ResourceMonitor with the 1.5 GB free-system-memory abort floor, and rewrites its JSON
after every condition, so an interruption costs one condition at most.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python rescue_grid.py
'''
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_IMAGES = _OUT / "rescue_grid"
_BLOCK1 = _OUT / "overnight_evidence"
_RESULT = _OUT / "rescue_grid.json"

_RESOLUTION = 512
_SEEDS = [42, 43, 45]
_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]

# Condition label -> extra keyword arguments handed to the pipeline call.
_CONDITIONS: List[Tuple[str, Dict[str, Any]]] = [
    ("campaign_defaults", {}),
    ("original1024", {"original_size": (1024, 1024)}),
    ("original1024_target1024", {"original_size": (1024, 1024), "target_size": (1024, 1024)}),
    ("original1024_target1024_g7.5", {"original_size": (1024, 1024), "target_size": (1024, 1024),
                                      "guidance_scale": 7.5}),
]


def _seed_everything(seed: int) -> Any:
    '''The campaign's seeding site, replayed; returns the generator the pipeline call should use.'''
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cuda").manual_seed(seed)


def _mean_abs_difference(left: Path, right: Path) -> Optional[float]:
    import numpy as np
    from PIL import Image
    if not (left.is_file() and right.is_file()):
        return None
    with Image.open(left) as a, Image.open(right) as b:
        first = np.asarray(a.convert("RGB"), dtype=np.int64)
        second = np.asarray(b.convert("RGB"), dtype=np.int64)
    if first.shape != second.shape:
        return None
    return round(float(np.mean(np.abs(first - second))), 4)


def main() -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    check_headroom()
    _IMAGES.mkdir(parents=True, exist_ok=True)

    prompt_of = {entry["name"]: entry["prompt"] for entry in _generation_order()}
    for name in _ENTITIES:
        assert name in prompt_of, f"{name} is not one of the campaign's ten entities"

    monitor = ResourceMonitor(_OUT / "rescue_grid_monitor.log", interval_s=15.0)
    monitor.start()
    results: Dict[str, Any] = {
        "resolution": _RESOLUTION, "model": _MODEL_ID, "seeds": _SEEDS, "entities": _ENTITIES,
        "conditions": [label for label, _ in _CONDITIONS], "images": [],
        "harness_control_mean_abs_difference_against_block1": [],
    }
    rows: List[Dict[str, Any]] = results["images"]

    t_start = time.time()
    try:
        # The two flags generate_dataset sets before generating, so this harness runs the same
        # kernels the production path does.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # device_map="balanced" instead of the usual from_pretrained(...).to("cuda"), because on
        # this machine today the default placement assembles the pipeline in system memory first and
        # the watchdog hard-aborted the first two attempts of this run at 1.45 and 1.38 GB free.
        # Measured by probe_load_memory.py: balanced placement bottoms out at 1.77 GB free, above the
        # 1.5 GB floor, and puts the denoiser on cuda:0 exactly as before. It is a placement change
        # only -- there is one graphics card, so every component still ends up in the same place --
        # and its effect on the pixels is measured by this script's harness control rather than
        # assumed to be nil.
        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, variant=_VARIANT, device_map="balanced",
        )

        for label, extra in _CONDITIONS:
            t0 = time.time()
            for seed in _SEEDS:
                for entity in _ENTITIES:
                    generator = _seed_everything(seed)
                    image = pipeline([prompt_of[entity]], generator=generator,
                                     height=_RESOLUTION, width=_RESOLUTION, **extra).images[0]
                    path = _IMAGES / f"{label}_{entity}_seed{seed}.png"
                    image.save(path, "PNG")
                    rows.append({"condition": label, "seed": seed, "entity": entity,
                                 "path": str(path)})
                    if label == "campaign_defaults":
                        block1 = _BLOCK1 / f"baseline_alone_{entity}_seed{seed}.png"
                        results["harness_control_mean_abs_difference_against_block1"].append({
                            "entity": entity, "seed": seed,
                            "mean_abs_difference": _mean_abs_difference(path, block1),
                            "block1_path": str(block1),
                        })
            results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
            results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                          if monitor.min_ram_free_gb != float("inf") else None)
            _RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"condition {label} done in {time.time() - t0:.1f} s "
                  f"({len(rows)} images so far)", flush=True)
    finally:
        monitor.stop()
        torch.use_deterministic_algorithms(False)

    for record in results["harness_control_mean_abs_difference_against_block1"]:
        print(f"harness control {record['entity']} seed {record['seed']}: "
              f"mean absolute pixel difference against block 1 = {record['mean_abs_difference']}")
    print(f"RESCUE_GRID_DONE images={len(rows)} seconds={time.time() - t_start:.1f} "
          f"peak_vram_used_gb={results.get('peak_vram_used_gb')} "
          f"min_ram_free_gb={results.get('min_ram_free_gb')} written={_RESULT}")


if __name__ == "__main__":
    main()
