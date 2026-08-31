'''Does the collapse disappear when Stable Diffusion XL is rendered nearer its native size?

This is the direct test of the last standing explanation in VALIDATION_REPORT_01.md: the model was
trained at 1024 pixels and we generate at 512, a quarter of the pixels and a 64x64 latent grid
instead of 128x128. Size micro-conditioning only TELLS the model the picture is large; resolution
gives it the room to make one.

WHY IT WAS NOT RUN BEFORE, AND WHAT IS DIFFERENT NOW. Two earlier 1024 attempts ended badly: the
2026-08-12 host crash and the 2026-08-15 `HIP error: unspecified launch failure`, both inside the
autoencoder, both with video memory at 11.98 of 11.98 GB. Neither had the two settings that target
exactly those two places: `enable_vae_tiling()`, which decodes the image in tiles instead of in one
allocation -- the autoencoder decode is where both crashes happened -- and
`enable_attention_slicing()`, which computes the denoiser's attention in slices. Arithmetic from the
measured 512 peak of 8.80 GB: weights account for about 6.9 GB, so activations are ~1.9 GB at 512,
~4.3 GB at 768 (2.25x the latent tokens) and ~7.6 GB at 1024 (4x). Slicing and tiling are what make
the last two figures survivable. The user authorised this run explicitly on 2026-08-17.

WHAT ATTEMPT 1 TAUGHT (2026-08-17, 15:59), because it changes the script rather than only the log.
The 512 condition completed and the first 768 image was killed by the watchdog with video memory at
11.98 of 11.98 GB and free system memory at 0.38 GB. Two causes, both fixed below:

* `enable_vae_tiling()` did nothing at either target resolution. `AutoencoderKL.decode` tiles only
  when the latent exceeds `tile_latent_min_size` (`autoencoder_kl.py:286`), and that threshold comes
  from the checkpoint's own `sample_size`, which is 1024 here, giving 1024/8 = 128. A 768-pixel
  image has a 96x96 latent and a 1024-pixel image has exactly 128 -- neither exceeds it. The
  thresholds are therefore set explicitly, so tiles are 512 pixels regardless of the output size.
* The two text encoders sit on the card for the whole run while contributing nothing after the
  prompt is encoded. The prompt embeddings are now computed once, up front, and both encoders are
  moved to the processor before any denoising starts, which returns their video memory to the
  denoiser -- the component that actually needs it at higher resolutions.

FIVE CONDITIONS, ordered cheapest and safest first, 9 images each (3 entities x 3 seeds: 42 which
renders well at 512, and 43 and 45 which do not):

    512_control        512 pixels, slicing and tiling ON, everything else as the campaign.
                       This is the HARNESS CONTROL: it regenerates images that already exist from
                       rescue_grid.py, so the mean absolute pixel difference against those says
                       whether slicing and tiling changed the picture. Without it, a difference at
                       768 could be the slicing rather than the resolution. Attempt 1 measured that
                       difference at 1.29 to 19.68 of 255 -- slicing DOES change the pixels, so the
                       comparison holds only within this run, where slicing is constant.
    512_control_2      the same, under the memory arrangement introduced for attempt 2 (encoders on
                       the processor, tiles forced to 512). Run again because the arrangement, not
                       only the resolution, changed; this is what the 768 and 1024 images are
                       compared against.
    768_default        768 pixels, no size arguments (so original_size becomes (768, 768)).
    768_original1024   768 pixels, original_size and target_size declared as 1024.
    1024_native        1024 pixels, no size arguments -- at this resolution the defaults ARE 1024,
                       which is the model's native training condition. Last, because it is the one
                       that has taken the machine down before.

Each image is written as soon as it is produced and the JSON is rewritten after every condition, so
an abort costs at most the image in flight. The ResourceMonitor's 1.5 GB free-system-memory floor is
in force and is not lowered; additionally each condition refuses to start unless the free video
memory before it exceeds the estimate for that resolution.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python resolution_probe.py
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
_IMAGES = _OUT / "resolution_probe"
_RESCUE = _OUT / "rescue_grid"
_RESULT = _OUT / "resolution_probe.json"

_SEEDS = [42, 43, 45]
_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]

# label -> (resolution, extra pipeline keyword arguments, required free video memory in GB before
# the condition starts). The requirement is the measured 512 peak of 8.8 GB scaled by the latent
# area, minus the weights that are already resident, with a margin.
_CONDITIONS: List[Tuple[str, int, Dict[str, Any], float]] = [
    ("512_control", 512, {}, 2.0),
    ("512_control_2", 512, {}, 2.0),
    ("768_default", 768, {}, 3.0),
    ("768_original1024", 768, {"original_size": (1024, 1024), "target_size": (1024, 1024)}, 3.0),
    ("1024_native", 1024, {}, 4.5),
]

# Tiles of 512 pixels, whatever the output size. The checkpoint's own thresholds (derived from its
# sample_size of 1024) never fire below 1024 pixels, which is why attempt 1's decode allocated in one
# piece and took the run down.
_TILE_SAMPLE_MIN_SIZE = 512
_TILE_LATENT_MIN_SIZE = 64


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
    import argparse

    import torch
    from diffusers import AutoPipelineForText2Image

    parser = argparse.ArgumentParser(description="Resolution probe for the Stable Diffusion XL base model.")
    parser.add_argument("--conditions", default=",".join(label for label, _, _, _ in _CONDITIONS),
                        help="comma-separated subset of the condition labels, in the order given")
    arguments = parser.parse_args()
    wanted = [item.strip() for item in arguments.conditions.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in {label for label, _, _, _ in _CONDITIONS}]
    assert not unknown, f"unknown condition label(s): {unknown}"
    conditions = [entry for entry in _CONDITIONS if entry[0] in wanted]

    check_headroom()
    _IMAGES.mkdir(parents=True, exist_ok=True)

    prompt_of = {entry["name"]: entry["prompt"] for entry in _generation_order()}
    monitor = ResourceMonitor(_OUT / "resolution_probe_monitor.log", interval_s=10.0)
    monitor.start()
    results: Dict[str, Any] = {
        "model": _MODEL_ID, "seeds": _SEEDS, "entities": _ENTITIES,
        "conditions": [label for label, _, _, _ in _CONDITIONS],
        "attention_slicing": True, "vae_tiling": True,
        "images": [], "conditions_completed": [], "conditions_refused": [],
        "harness_control_mean_abs_difference_against_rescue_grid": [],
    }
    rows: List[Dict[str, Any]] = results["images"]

    def write() -> None:
        results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                      if monitor.min_ram_free_gb != float("inf") else None)
        _RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        pipeline = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float16, variant=_VARIANT, device_map="balanced",
        )
        # The two settings that address the places both earlier 1024 runs died: the autoencoder
        # decode allocation and the denoiser's attention matrix. The thresholds are set explicitly
        # because the checkpoint's own never fire below 1024 pixels -- see the module docstring.
        pipeline.enable_vae_tiling()
        pipeline.vae.tile_sample_min_size = _TILE_SAMPLE_MIN_SIZE
        pipeline.vae.tile_latent_min_size = _TILE_LATENT_MIN_SIZE
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
        results["tile_sample_min_size"] = pipeline.vae.tile_sample_min_size
        results["tile_latent_min_size"] = pipeline.vae.tile_latent_min_size

        # The prompt embeddings are computed once, while the text encoders are still on the card;
        # then both encoders move to the processor for the rest of the run. They contribute nothing
        # after this point, and their video memory is what the denoiser is short of at 768 and 1024.
        embeddings: Dict[str, Tuple[Any, Any, Any, Any]] = {}
        for entity in _ENTITIES:
            embeddings[entity] = pipeline.encode_prompt(
                prompt=prompt_of[entity], device=pipeline.device if hasattr(pipeline, "device") else "cuda",
                num_images_per_prompt=1, do_classifier_free_guidance=True,
            )
        free_before_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        free_after_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        results["video_memory_freed_by_moving_the_text_encoders_gb"] = round(
            free_after_gb - free_before_gb, 3)
        print(f"text encoders moved to the processor: free video memory {free_before_gb:.2f} -> "
              f"{free_after_gb:.2f} GB", flush=True)
        write()

        for label, resolution, extra, required_free_vram_gb in conditions:
            torch.cuda.empty_cache()
            free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
            if free_vram_gb < required_free_vram_gb:
                results["conditions_refused"].append({
                    "condition": label, "free_video_memory_gb": round(free_vram_gb, 3),
                    "required_gb": required_free_vram_gb})
                print(f"REFUSED {label}: free video memory {free_vram_gb:.2f} GB is below the "
                      f"{required_free_vram_gb} GB this condition needs", flush=True)
                write()
                continue

            t0 = time.time()
            for seed in _SEEDS:
                for entity in _ENTITIES:
                    generator = _seed_everything(seed)
                    t_image = time.time()
                    prompt_embeds, negative_embeds, pooled, negative_pooled = embeddings[entity]
                    image = pipeline(
                        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                        pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled,
                        generator=generator, height=resolution, width=resolution, **extra,
                    ).images[0]
                    path = _IMAGES / f"{label}_{entity}_seed{seed}.png"
                    image.save(path, "PNG")
                    rows.append({"condition": label, "resolution": resolution, "seed": seed,
                                 "entity": entity, "path": str(path),
                                 "seconds": round(time.time() - t_image, 1)})
                    if label.startswith("512_control"):
                        reference = _RESCUE / f"campaign_defaults_{entity}_seed{seed}.png"
                        results["harness_control_mean_abs_difference_against_rescue_grid"].append({
                            "condition": label, "entity": entity, "seed": seed,
                            "mean_abs_difference": _mean_abs_difference(path, reference),
                            "reference_path": str(reference)})
                    write()
            results["conditions_completed"].append(label)
            write()
            print(f"condition {label} done in {time.time() - t0:.1f} s "
                  f"({len(rows)} images so far)", flush=True)
    finally:
        monitor.stop()
        torch.use_deterministic_algorithms(False)
        write()

    for record in results["harness_control_mean_abs_difference_against_rescue_grid"]:
        print(f"harness control {record['condition']} {record['entity']} seed {record['seed']}: "
              f"mean absolute pixel difference against the 512 rescue-grid image = "
              f"{record['mean_abs_difference']}")
    print(f"RESOLUTION_PROBE_DONE images={len(rows)} "
          f"completed={results['conditions_completed']} refused={results['conditions_refused']} "
          f"peak_vram_used_gb={results.get('peak_vram_used_gb')} "
          f"min_ram_free_gb={results.get('min_ram_free_gb')} written={_RESULT}")


if __name__ == "__main__":
    main()
