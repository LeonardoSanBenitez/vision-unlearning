"""Micro-benchmark: raw fp16 SD1.4 generation throughput on this GPU.

Isolates generation speed from training/LoRA. Loads the base SD1.4 pipeline via the real
``generate_dataset`` path (fp16), times a small and a larger call to separate fixed model-load time
from marginal per-image time, and reports the GPU device actually used. Run it with and without
``HSA_OVERRIDE_GFX_VERSION=10.3.0`` to see whether forcing the supported gfx1030 kernel target fixes
the slow, unoptimized gfx1031 path (the 'CK grouped conv library not found for device gfx1031' warning).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_OUT = Path(__file__).resolve().parent / "assets"


def main() -> int:
    import os
    import torch
    from vision_unlearning.utils.data_generation import generate_dataset

    parser = argparse.ArgumentParser(description="fp16 SD1.4 generation micro-benchmark.")
    parser.add_argument("--small", type=int, default=2)
    parser.add_argument("--big", type=int, default=6)
    parser.add_argument("--gen-batch", type=int, default=2, help="inference batch (25 OOMs the 12GB GPU).")
    parser.add_argument("--steps", type=int, default=50, help="(informational) diffusion steps generate_dataset uses.")
    args = parser.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    out_dir = _OUT / "bench_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '(unset)')}")

    def gen(n: int, tag: str) -> float:
        prompts = [f"An image of a cat number {i}" for i in range(n)]
        filenames = [f"{tag}_{i}.png" for i in range(n)]
        t = time.time()
        generate_dataset(
            model_base_name="CompVis/stable-diffusion-v1-4",
            lora_name=None,
            prompts=prompts,
            output_path=str(out_dir),
            seeds=[42],
            filenames=filenames,
            batch_size=args.gen_batch,
            lora_requires_inversion=False,
        )
        dt = time.time() - t
        f, t_v = torch.cuda.mem_get_info(0)
        print(f"[{tag}] n={n} took {dt:.1f}s | VRAM used {(t_v - f) / 1024 ** 3:.2f}GB")
        return dt

    t_small = gen(args.small, "small")
    t_big = gen(args.big, "big")
    marginal = (t_big - t_small) / (args.big - args.small)
    load = t_small - args.small * marginal
    res = {
        "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION", "(unset)"),
        "n_small": args.small, "t_small_s": round(t_small, 1),
        "n_big": args.big, "t_big_s": round(t_big, 1),
        "fixed_load_s": round(load, 1),
        "marginal_per_image_s": round(marginal, 2),
    }
    tag = "with_override" if os.environ.get("HSA_OVERRIDE_GFX_VERSION") else "baseline"
    (_OUT / f"bench_result_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("BENCH_RESULT", json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
