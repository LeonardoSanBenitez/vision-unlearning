'''Why do the base-model images at 512 not depict the requested person?

The validation report found that 5 of the 20 off-baseline images contain no human subject at all, and
that the rest range from photographic to flat poster art. "Too few denoising steps" is ruled out from
the installed source before spending any graphics-card time: `generate_dataset` never passes
`num_inference_steps`, and diffusers 0.30.0's `StableDiffusionXLPipeline.__call__` defaults it to 50
(line 827) with `guidance_scale` 5.0 (line 831).

That leaves two candidates, both consequences of generating at 512 rather than Stable Diffusion XL's
native 1024, and this script separates them WITHOUT going anywhere near 1024 -- every condition renders
at 512, so none of them can reproduce the 1024 autoencoder failure that once took the host down.

* MICRO-CONDITIONING. `pipeline_stable_diffusion_xl.py:1029` reads
  `original_size = original_size or (height, width)`. Passing height=width=512 and no `original_size`
  therefore tells the model the image it is producing came from a 512-pixel original. Stable Diffusion
  XL is trained with that size as a conditioning signal, so this is a request for the small-original
  bucket rather than a neutral instruction. Condition `original_size_1024` asks for a 1024 original
  while still rendering 512 pixels.
* PROMPT ADHERENCE. The pipeline's default guidance scale is 5.0, where every existing Stable Diffusion
  1.4 artifact in this project was generated at that pipeline's own default of 7.5. Condition
  `original_size_1024_guidance_7_5` varies both together, so a difference between the second and third
  conditions isolates guidance.

THE CONTROL IS THE POINT. Condition `campaign` replays exactly what the campaign did, and its images
must come out pixel-identical to the ones already on disk. If they do not, the comparison is worthless
and this script says so rather than reporting a difference it cannot attribute.

Reproducing the campaign requires replaying the WHOLE canonical prompt list in order, not just the
entity of interest: `generate_dataset` advances one generator across the prompts, so an entity's noise
draw is fixed by its POSITION in that list (`CONTRIBUTING_ICARE.md` section 2). Generating one prompt
alone would draw the first noise and match nothing.

The five seeding lines below are duplicated from `data_generation.py`'s SEEDING SITE deliberately and
under protest, because reproducing that site bit-for-bit is the entire purpose here. This script writes
no benchmark artifact and nothing under `assets/datasets/`; it is a diagnostic whose outputs live beside
it and are read once.

    python diagnose_render_quality.py --seed 42
'''
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_campaign import _MODEL_ID, _RESOLUTION, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_DIAG = _OUT / "diagnose_render_quality"

# name -> extra keyword arguments handed to the pipeline call, on top of the campaign's own.
_CONDITIONS: List[Tuple[str, Dict[str, Any]]] = [
    ("campaign", {}),
    ("original_size_1024", {"original_size": (1024, 1024)}),
    ("original_size_1024_guidance_7_5", {"original_size": (1024, 1024), "guidance_scale": 7.5}),
]


def main() -> None:
    import numpy as np
    import torch
    from diffusers import AutoPipelineForText2Image
    from PIL import Image

    parser = argparse.ArgumentParser(description="Isolate why 512 renders do not depict the requested entity.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    check_headroom()
    _DIAG.mkdir(parents=True, exist_ok=True)

    order = _generation_order()
    prompts = [entry["prompt"] for entry in order]
    names = [entry["name"] for entry in order]
    campaign_dir = _OUT / f"campaign_seed{seed}"

    monitor = ResourceMonitor(_OUT / f"diagnose_render_quality_seed{seed}_monitor.log", interval_s=5.0)
    monitor.start()
    results: Dict[str, Any] = {"seed": seed, "resolution": _RESOLUTION, "prompts": prompts,
                               "conditions": {}}
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, safety_checker=None, variant=_VARIANT,
        ).to("cuda")

        for condition, extra in _CONDITIONS:
            t0 = time.time()
            # The campaign's seeding site, replayed exactly.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)

            per_entity: Dict[str, Any] = {}
            for index, prompt in enumerate(prompts):
                image = pipeline(
                    [prompt], generator=generator,
                    height=_RESOLUTION, width=_RESOLUTION, **extra,
                ).images[0]
                out_path = _DIAG / f"{condition}_{names[index]}_seed{seed}.png"
                image.save(out_path, "PNG")
                record: Dict[str, Any] = {"path": str(out_path)}
                if condition == "campaign":
                    reference = campaign_dir / f"off_{names[index]}_seed{seed}.png"
                    a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float64)
                    b = np.asarray(image.convert("RGB"), dtype=np.float64)
                    record["reference"] = str(reference)
                    record["max_absolute_difference_against_campaign"] = float(np.abs(a - b).max())
                per_entity[names[index]] = record
            results["conditions"][condition] = {
                "extra_arguments": {k: list(v) if isinstance(v, tuple) else v for k, v in extra.items()},
                "seconds": round(time.time() - t0, 1),
                "per_entity": per_entity,
            }
            print(f"condition {condition}: {len(prompts)} images in "
                  f"{results['conditions'][condition]['seconds']} s")
    finally:
        monitor.stop()

    control = results["conditions"]["campaign"]["per_entity"]
    differences = [record["max_absolute_difference_against_campaign"] for record in control.values()]
    results["control"] = {
        "n_compared": len(differences),
        "max_absolute_difference_over_all_entities": max(differences),
        "n_identical": sum(1 for value in differences if value == 0.0),
    }
    results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
    results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                  if monitor.min_ram_free_gb != float("inf") else None)
    (_OUT / f"diagnose_render_quality_seed{seed}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # The check prints the comparison, never a verdict computed elsewhere.
    print(f"control: compared {results['control']['n_compared']} images against the campaign's own "
          f"off-baselines; identical {results['control']['n_identical']} of "
          f"{results['control']['n_compared']}; maximum absolute pixel difference "
          f"{results['control']['max_absolute_difference_over_all_entities']}")
    print(f"peak video memory {results['peak_vram_used_gb']} GB, "
          f"minimum free system memory {results['min_ram_free_gb']} GB")
    print(f"written: {_OUT / f'diagnose_render_quality_seed{seed}.json'}")


if __name__ == "__main__":
    main()
