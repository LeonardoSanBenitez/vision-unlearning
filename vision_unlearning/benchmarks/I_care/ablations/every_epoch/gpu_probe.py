"""Locate where the SD1.4 pipeline actually runs and measure one-image speed (diagnostic).

Loads SD1.4 fp16 explicitly to cuda, prints the pipeline's device and dedicated-VRAM usage before/after
load and during generation, and times a single 50-step image. This distinguishes fast dedicated-VRAM GPU
(~a few s/image), slow AMD shared-memory GPU (dedicated VRAM stays ~0 while system RAM holds the model),
and CPU. Run with PYTHONPATH at the repo root, HF_TOKEN set, HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import time


def main() -> int:
    import torch
    from diffusers import AutoPipelineForText2Image

    total = torch.cuda.mem_get_info(0)[1]

    def vram_used_gb() -> float:
        free, _ = torch.cuda.mem_get_info(0)
        return (total - free) / 1024 ** 3

    print(f"cuda_available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}")
    print(f"VRAM used before load: {vram_used_gb():.2f}GB")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    ).to("cuda")
    unet_dev = next(pipe.unet.parameters()).device
    unet_dtype = next(pipe.unet.parameters()).dtype
    print(f"pipe unet device={unet_dev} dtype={unet_dtype} | VRAM used after load: {vram_used_gb():.2f}GB")

    t0 = time.time()
    _ = pipe("An image of a cat", num_inference_steps=50).images[0]
    dt = time.time() - t0
    print(f"1 image (50 steps): {dt:.1f}s | VRAM used during: {vram_used_gb():.2f}GB")
    print(f"PROBE_RESULT dt_1img_s={dt:.1f} vram_after_load_gb={vram_used_gb():.2f} unet_device={unet_dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
