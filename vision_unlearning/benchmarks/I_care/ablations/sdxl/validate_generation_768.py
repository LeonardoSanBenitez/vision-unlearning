'''The sign-off run: does Stable Diffusion XL generation work on this machine, for the whole set we
actually need, at the settings we intend to freeze?

Everything before this was diagnosis on samples of three entities. This run covers the population the
campaign uses and produces the artifacts a human signs off on:

  A. OFF-BASELINES -- all TEN entities at BOTH campaign seeds (42 and 43), 20 images. Every image is
     generated alone, with the generator reseeded, at the frozen generation hyperparameters. If ten
     of ten depict the right person at both seeds, generation works; if not, it does not, and the
     report says so.
  B. ON-IMAGES -- the same pipeline with a trained adapter loaded (seed 42, epoch 200), three
     entities, 3 images. The off-image path working says nothing about the adapted path, and the
     campaign needs both. This also exercises loading an adapter into a pipeline that carries the
     memory settings below.
  C. REPRODUCIBILITY -- two of the stage A images regenerated in the same process and diffed against
     their originals. Deterministic algorithms are OFF at this resolution (see below), so
     "reproducible" is no longer an assumption; this measures what it is actually worth.

FROZEN GENERATION HYPERPARAMETERS (user's instruction, 2026-08-17: choose now, then keep them for
the rest of the experiment):

    768 x 768 pixels, original_size (1024, 1024), target_size (1024, 1024),
    crops_coords_top_left (0, 0), guidance 7.5, 50 steps, one call per entity.

MEMORY AND KERNEL SETTINGS, measured in spike_768.py and required for 768 on this card:

    autoencoder tiling forced to 512-pixel tiles (the checkpoint's own threshold is 128 latent
    units, so tiling never engages below 1024 pixels), autoencoder slicing, attention slicing at
    slice size 1, and deterministic algorithms OFF -- with them on, 768 dies in a convolution with
    `HIP error: unspecified launch failure`.

Cost: 25 images at about 140 s each, roughly one hour. JSON is rewritten after every image, and the
run is under the ResourceMonitor with the 1.5 GB free-system-memory floor.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python validate_generation_768.py
'''
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_IMAGES = _OUT / "validate_generation_768"
_RESULT = _OUT / "validate_generation_768.json"
_ADAPTER = _OUT / "campaign_model" / "seed42" / "epoch-200"

_RESOLUTION = 768
_SEEDS = [42, 43]
_ON_IMAGE_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]
_REPRODUCIBILITY_CASES = [(42, "Mark_Philippoussis"), (43, "Megawati_Sukarnoputri")]

_GENERATION_KWARGS: Dict[str, Any] = {
    "original_size": (1024, 1024),
    "target_size": (1024, 1024),
    "crops_coords_top_left": (0, 0),
    "guidance_scale": 7.5,
}
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
    import torch
    from diffusers import AutoPipelineForText2Image

    check_headroom()
    _IMAGES.mkdir(parents=True, exist_ok=True)
    assert (_ADAPTER / "pytorch_lora_weights.safetensors").is_file(), \
        f"adapter missing: {_ADAPTER / 'pytorch_lora_weights.safetensors'}"

    order = _generation_order()
    prompt_of = {entry["name"]: entry["prompt"] for entry in order}
    names = [entry["name"] for entry in order]

    monitor = ResourceMonitor(_OUT / "validate_generation_768_monitor.log", interval_s=15.0)
    monitor.start()
    results: Dict[str, Any] = {
        "resolution": _RESOLUTION, "model": _MODEL_ID, "seeds": _SEEDS,
        "generation_hyperparameters": {k: list(v) if isinstance(v, tuple) else v
                                       for k, v in _GENERATION_KWARGS.items()},
        "deterministic_algorithms": False, "attention_slice_size": 1,
        "tile_sample_min_size": _TILE_SAMPLE_MIN_SIZE,
        "adapter": str(_ADAPTER),
        "off_baselines": [], "on_images": [], "reproducibility": [],
        "stages_completed": [],
    }

    def write() -> None:
        results["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        results["min_ram_free_gb"] = (round(monitor.min_ram_free_gb, 3)
                                      if monitor.min_ram_free_gb != float("inf") else None)
        _RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def generate(pipeline: Any, entity: str, seed: int, path: Path) -> Optional[float]:
        '''Generates one image, or returns None if it is already on disk.

        Resumable by construction, because this machine's free system memory is not under this
        script's control: the watchdog has aborted two runs mid-flight when the WSL2 virtual machine
        re-inflated underneath them. With this skip, an abort costs the image in flight and nothing
        else, and the wrapper can simply run the script again.
        '''
        if path.is_file():
            print(f"already on disk, skipping: {path.name}", flush=True)
            return None
        t0 = time.time()
        generator = _seed_everything(seed)
        image = pipeline([prompt_of[entity]], generator=generator,
                         height=_RESOLUTION, width=_RESOLUTION, **_GENERATION_KWARGS).images[0]
        image.save(path, "PNG")
        return round(time.time() - t0, 1)

    try:
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

        # --- A: off-baselines, ten entities, both campaign seeds ------------------------------- #
        t_stage = time.time()
        for seed in _SEEDS:
            for entity in names:
                path = _IMAGES / f"off_{entity}_seed{seed}.png"
                seconds = generate(pipeline, entity, seed, path)
                results["off_baselines"].append({"entity": entity, "seed": seed,
                                                 "path": str(path), "seconds": seconds,
                                                 "from_earlier_attempt": seconds is None})
                write()
                print(f"off {entity} seed {seed}: {seconds} s "
                      f"({len(results['off_baselines'])} of 20)", flush=True)
        results["stages_completed"].append("A_off_baselines")
        print(f"STAGE A DONE in {time.time() - t_stage:.1f} s", flush=True)
        write()

        # --- C: reproducibility, before the adapter changes the pipeline ------------------------ #
        for seed, entity in _REPRODUCIBILITY_CASES:
            path = _IMAGES / f"repeat_{entity}_seed{seed}.png"
            # Always regenerated, never skipped: the question is whether the SAME PROCESS reproduces
            # its own image, so a repeat left behind by an earlier attempt would answer a different
            # question (across processes) under the same name.
            path.unlink(missing_ok=True)
            seconds = generate(pipeline, entity, seed, path)
            original = _IMAGES / f"off_{entity}_seed{seed}.png"
            results["reproducibility"].append({
                "entity": entity, "seed": seed, "path": str(path), "seconds": seconds,
                "mean_abs_difference_against_the_first_generation": _mean_abs_difference(path, original),
                "original_path": str(original)})
            write()
            print(f"repeat {entity} seed {seed}: mean absolute pixel difference "
                  f"{results['reproducibility'][-1]['mean_abs_difference_against_the_first_generation']}",
                  flush=True)
        results["stages_completed"].append("C_reproducibility")
        write()

        # --- B: on-images, the same pipeline with a trained adapter ----------------------------- #
        pipeline.load_lora_weights(str(_ADAPTER), weight_name="pytorch_lora_weights.safetensors")
        t_stage = time.time()
        for entity in _ON_IMAGE_ENTITIES:
            path = _IMAGES / f"on_epoch200_{entity}_seed42.png"
            seconds = generate(pipeline, entity, 42, path)
            results["on_images"].append({"entity": entity, "seed": 42, "epoch": 200,
                                         "path": str(path), "seconds": seconds})
            write()
            print(f"on {entity} seed 42 epoch 200: {seconds} s", flush=True)
        results["stages_completed"].append("B_on_images")
        print(f"STAGE B DONE in {time.time() - t_stage:.1f} s", flush=True)
        write()
    finally:
        monitor.stop()
        write()

    total = len(results["off_baselines"]) + len(results["on_images"]) + len(results["reproducibility"])
    print(f"VALIDATE_GENERATION_768_DONE images={total} "
          f"off={len(results['off_baselines'])} on={len(results['on_images'])} "
          f"repeat={len(results['reproducibility'])} "
          f"stages={results['stages_completed']} "
          f"peak_vram_used_gb={results.get('peak_vram_used_gb')} "
          f"min_ram_free_gb={results.get('min_ram_free_gb')} written={_RESULT}")


if __name__ == "__main__":
    main()
