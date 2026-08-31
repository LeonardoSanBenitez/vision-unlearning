'''How much SYSTEM memory does building the Stable Diffusion XL pipeline transiently cost, and does
placing the components directly on the graphics card reduce it?

Why this exists: on 2026-08-17 two attempts at `rescue_grid.py` were hard-aborted by the watchdog
during the pipeline load, both at the 1.5 GB free-system-memory floor (`assets/rescue_grid_monitor.log`:
"RAM 91% (1.45GB free)" then "1.38GB free"), while the same load on 2026-08-15 left 5.75 GB free. The
machine is simply carrying more resident processes today. Before asking anyone to close applications,
measure whether the load can be made cheaper.

Two placements, each in its own process (never two pipelines in one process -- that is the S2
measurement this task has honoured throughout), reporting the minimum free system memory the monitor
saw during the load:

    --placement default    what every script here does: from_pretrained(...).to("cuda")
    --placement balanced   from_pretrained(..., device_map="balanced"), which asks diffusers to
                           place each component on the graphics card as it is read, instead of
                           assembling the whole pipeline in system memory first

The floor is NOT lowered for this probe. If a placement cannot load above the floor, the monitor
stops it and that is the measurement.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python probe_load_memory.py --placement default
'''
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from run_campaign import _MODEL_ID, _VARIANT
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"


def main() -> None:
    import psutil
    import torch
    from diffusers import AutoPipelineForText2Image

    parser = argparse.ArgumentParser(description="Measure the system-memory cost of the pipeline load.")
    parser.add_argument("--placement", choices=["default", "balanced"], required=True)
    args = parser.parse_args()

    check_headroom()
    free_before_gb = psutil.virtual_memory().available / 1024 ** 3
    monitor = ResourceMonitor(_OUT / f"probe_load_memory_{args.placement}_monitor.log", interval_s=2.0)
    monitor.start()
    t0 = time.time()
    try:
        if args.placement == "default":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                _MODEL_ID, torch_dtype=torch.float16, variant=_VARIANT,
            ).to("cuda")
        else:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                _MODEL_ID, torch_dtype=torch.float16, variant=_VARIANT, device_map="balanced",
            )
        load_s = round(time.time() - t0, 1)
        free_after_gb = psutil.virtual_memory().available / 1024 ** 3
        device_of_unet = str(next(pipeline.unet.parameters()).device)
    finally:
        monitor.stop()

    payload: Dict[str, Any] = {
        "placement": args.placement,
        "load_seconds": load_s,
        "free_system_memory_before_gb": round(free_before_gb, 3),
        "free_system_memory_after_gb": round(free_after_gb, 3),
        "minimum_free_system_memory_during_load_gb": (
            round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None),
        "peak_video_memory_gb": round(monitor.peak_vram_used_gb, 3),
        "denoiser_device": device_of_unet,
    }
    (_OUT / f"probe_load_memory_{args.placement}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PROBE_LOAD_MEMORY_DONE placement={args.placement}")


if __name__ == "__main__":
    main()
