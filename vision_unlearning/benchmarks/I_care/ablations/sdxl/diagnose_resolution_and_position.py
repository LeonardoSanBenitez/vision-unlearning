'''Separate the two remaining explanations for base-model renders that do not depict the entity.

The first diagnostic (`diagnose_render_quality.py`) showed the micro-conditioning `original_size` is a
real but partial cause. Reading its figure raised a sharper question: the TARGET renders correctly under
every condition while three receivers do not, and a plain "512 is too small for this model" explanation
does not predict that -- it would degrade all ten prompts about equally.

Two variables were never separated, because the campaign varied them together:

* RESOLUTION. The campaign renders 512 pixels, so the denoiser runs on 64x64 latents against the
  128x128 it was trained for. This is the standard cause of duplicated subjects and flat vector-art
  compositions.
* POSITION IN THE PROMPT LIST. `generate_dataset` advances ONE generator across the ten prompts in
  order, so each entity's initial noise is determined by its position. The target is index 0 and gets
  the first draw; the failing receivers are all later. Whether an entity collapses may be a property of
  its noise draw rather than of its prompt.

Each entity here is generated ALONE, i.e. always at position 0 with a freshly seeded generator, so
comparing these against the campaign's own images isolates position. Resolution is then varied on top.

    python diagnose_resolution_and_position.py --seed 42

WARNING, and the reason this script exists as its own registered run: the 1024 condition is the first
1024 work attempted since 2026-08-12, when a 1024 stage crashed the host from inside the autoencoder.
It runs under the S1 ResourceMonitor with the 1.5 GB free-system-memory abort floor, and 1024 is the
LAST condition so that every cheaper measurement is already written to disk if the host does go down.
'''
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_DIAG = _OUT / "diagnose_resolution_and_position"

# (label, render size, extra pipeline keyword arguments). Ordered cheapest-and-safest first; the 1024
# condition is last on purpose.
_CONDITIONS: List[Tuple[str, int, Dict[str, Any]]] = [
    ("alone_512", 512, {}),
    ("alone_512_original_size_1024", 512, {"original_size": (1024, 1024)}),
    ("alone_1024", 1024, {}),
]


def main() -> None:
    import numpy as np
    import torch
    from diffusers import AutoPipelineForText2Image

    parser = argparse.ArgumentParser(description="Separate resolution from prompt-list position.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entities", default="Mark_Philippoussis,Juan_Carlos_Ferrero,Andy_Roddick,Ian_Thorpe")
    args = parser.parse_args()
    seed = args.seed
    wanted = args.entities.split(",")

    check_headroom()
    _DIAG.mkdir(parents=True, exist_ok=True)

    order = _generation_order()
    prompt_of = {entry["name"]: entry["prompt"] for entry in order}
    position_of = {entry["name"]: index for index, entry in enumerate(order)}
    missing = [name for name in wanted if name not in prompt_of]
    assert not missing, f"unknown entities: {missing}"

    monitor = ResourceMonitor(_OUT / f"diagnose_resolution_and_position_seed{seed}_monitor.log", interval_s=5.0)
    monitor.start()
    results: Dict[str, Any] = {
        "seed": seed,
        "entities": wanted,
        "campaign_position": {name: position_of[name] for name in wanted},
        "note": "every image here is generated alone, so the entity is always at position 0 of its own call",
        "conditions": {},
    }
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, safety_checker=None, variant=_VARIANT,
        ).to("cuda")

        for label, size, extra in _CONDITIONS:
            t0 = time.time()
            per_entity: Dict[str, Any] = {}
            for name in wanted:
                # Fresh seeding per image: this is what puts every entity at position 0.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipeline([prompt_of[name]], generator=generator,
                                 height=size, width=size, **extra).images[0]
                out_path = _DIAG / f"{label}_{name}_seed{seed}.png"
                image.save(out_path, "PNG")
                per_entity[name] = {"path": str(out_path), "size": size}
                print(f"{label}: {name} written ({size}x{size})", flush=True)
            results["conditions"][label] = {
                "render_size": size,
                "extra_arguments": {k: list(v) if isinstance(v, tuple) else v for k, v in extra.items()},
                "seconds": round(time.time() - t0, 1),
                "per_entity": per_entity,
            }
            # Written after every condition, not only at the end: if the 1024 stage takes the host
            # down, the cheaper measurements survive.
            (_OUT / f"diagnose_resolution_and_position_seed{seed}.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            print(f"condition {label} finished in {results['conditions'][label]['seconds']} s", flush=True)
    finally:
        monitor.stop()

    results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
    results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                  if monitor.min_ram_free_gb != float("inf") else None)
    (_OUT / f"diagnose_resolution_and_position_seed{seed}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"peak video memory {results['peak_vram_used_gb']} GB, "
          f"minimum free system memory {results['min_ram_free_gb']} GB")
    print(f"campaign positions of these entities: {results['campaign_position']}")
    print(f"written: {_OUT / f'diagnose_resolution_and_position_seed{seed}.json'}")


if __name__ == "__main__":
    main()
