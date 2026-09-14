'''One image at 768 pixels, to settle whether this card can render above 512 at all.

A spike, not a grid (`CONTRIBUTING_ABLATIONS.md` §4): the smallest end-to-end version of the thing,
whose only job is to say feasible or not feasible. Two images, both single, in this order:

  1. 512 pixels -- a smoke test. If this fails, the settings are broken and nothing about 768 can be
     concluded from the run.
  2. 768 pixels -- the question.

WHAT IS DIFFERENT FROM THE RUN THAT CRASHED. `resolution_probe.py` reached 768 with the autoencoder
decode tiled at 512 and attention slicing on, and died inside a convolution in a denoiser residual
block (`diffusers/models/resnet.py:341` -> `torch/nn/modules/conv.py:543`) with
`HIP error: unspecified launch failure`, video memory at 10.2 of 11.98 GB. That is a driver-level
launch failure inside a convolution, not a clean out-of-memory, so the levers that matter are the
ones that change which convolution kernel is selected and how much scratch space it asks for:

  * `torch.use_deterministic_algorithms(False)` -- the deterministic convolution algorithm can
    demand a substantially larger workspace, and every generation so far has run with it ON because
    `generate_dataset` sets it. If 768 succeeds only with it off, that is a real and reportable
    trade: reproducibility against resolution.
  * attention slicing at its most aggressive setting (slice size 1) and feed-forward chunking, so
    that whatever headroom exists goes to the convolutions.

If this spike fails as well, the conclusion for the report is that 768 is not reachable on this
machine as configured, and the resolution explanation cannot be tested here.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python spike_768.py
'''
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_IMAGES = _OUT / "spike_768"
_RESULT = _OUT / "spike_768.json"

_ENTITY = "Mark_Philippoussis"
_SEED = 43  # a seed whose 512 render collapses, so a success here is also informative about quality
_TILE_SAMPLE_MIN_SIZE = 512
_TILE_LATENT_MIN_SIZE = 64


def _seed_everything(seed: int) -> Any:
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

    check_headroom()
    _IMAGES.mkdir(parents=True, exist_ok=True)

    prompt = {entry["name"]: entry["prompt"] for entry in _generation_order()}[_ENTITY]
    monitor = ResourceMonitor(_OUT / "spike_768_monitor.log", interval_s=5.0)
    monitor.start()
    results: Dict[str, Any] = {
        "entity": _ENTITY, "seed": _SEED, "prompt": prompt,
        "deterministic_algorithms": False, "attention_slice_size": 1,
        "forward_chunking": False, "tile_sample_min_size": _TILE_SAMPLE_MIN_SIZE,
        "steps": [],
    }
    steps: List[Dict[str, Any]] = results["steps"]

    def write() -> None:
        results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                      if monitor.min_ram_free_gb != float("inf") else None)
        _RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    try:
        # Deliberately NOT enabling deterministic algorithms: that is the variable under test.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, variant=_VARIANT, device_map="balanced",
        )
        pipeline.enable_vae_tiling()
        pipeline.vae.tile_sample_min_size = _TILE_SAMPLE_MIN_SIZE
        pipeline.vae.tile_latent_min_size = _TILE_LATENT_MIN_SIZE
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing(1)
        # NOT chunking the feed-forward layers: UNet2DConditionModel in the installed diffusers
        # (0.30) has no enable_forward_chunking, so the first run of this spike died on an
        # AttributeError before reaching any image. Recorded rather than silently dropped.
        results["forward_chunking"] = False

        for resolution in (512, 768):
            torch.cuda.empty_cache()
            free_before_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
            print(f"starting {resolution} pixels with {free_before_gb:.2f} GB free video memory",
                  flush=True)
            t0 = time.time()
            generator = _seed_everything(_SEED)
            image = pipeline([prompt], generator=generator,
                             height=resolution, width=resolution).images[0]
            path = _IMAGES / f"spike_{resolution}_{_ENTITY}_seed{_SEED}.png"
            image.save(path, "PNG")
            steps.append({"resolution": resolution, "path": str(path),
                          "seconds": round(time.time() - t0, 1),
                          "free_video_memory_before_gb": round(free_before_gb, 3),
                          "peak_video_memory_gb": round(monitor.peak_vram_used_gb, 3)})
            write()
            print(f"{resolution} pixels OK in {steps[-1]['seconds']} s -> {path}", flush=True)
    finally:
        monitor.stop()
        write()

    print(f"SPIKE_768_DONE resolutions={[s['resolution'] for s in steps]} "
          f"peak_vram_used_gb={results.get('peak_vram_used_gb')} "
          f"min_ram_free_gb={results.get('min_ram_free_gb')} written={_RESULT}")


if __name__ == "__main__":
    main()
