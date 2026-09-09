"""S2 of PLAN-TASK-2026-08-12-SDXL: can the shared generation function drive SDXL, and is it deterministic?

Three questions, each of which would be expensive to discover during the campaign, and none of which is
answered by the feasibility probe (that probe drove ``StableDiffusionXLPipeline`` directly; the campaign drives
``vision_unlearning.utils.data_generation.generate_dataset``, which is what every existing Stable Diffusion 1.4
image in the every-epoch ablation came out of):

C7  Does the pipeline construction inside ``generate_dataset`` accept ``safety_checker=None`` for a Stable
    Diffusion XL checkpoint?  ``StableDiffusionXLPipeline.__init__`` has no such parameter, so this is either a
    warning-and-ignore or an exception, and reading diffusers 0.30.0's ``pipeline_utils.py`` (line 820) says
    warning -- measured here rather than trusted.
C6  Is generation reproducible?  ``generate_dataset`` enables ``torch.use_deterministic_algorithms(True)``
    (strict, not ``warn_only``), and whether every operation SDXL uses has a deterministic implementation on
    this ROCm build is unknown.  Two runs at the same seed are compared pixel by pixel.
--  What does an image cost, split into the fixed cost of loading the pipeline and the marginal cost per image
    (``CONTRIBUTING_ABLATIONS.md`` section 4)?  The campaign generates 280 images; whether the pipeline is
    loaded once per image or once per stage is a factor-of-three difference in the budget, and the numbers to
    decide that are measured here.

Resolution: 512x512 explicitly, per the plan's D3.  ``generate_dataset`` gained optional ``height``/``width``
parameters for this (plan D10) -- without them SDXL falls back to its own default of 1024, which is measured
infeasible on this machine.

**One pipeline per process.** Measured here, the hard way: building a second Stable Diffusion XL pipeline in a
process that had already held one drove free system memory to 1.31 GB and the watchdog killed the run, even
though ``del`` plus ``empty_cache`` had already returned the video memory. So the cross-reload comparison is a
second invocation, not a fourth phase of the first, and the campaign inherits the same constraint: one
generation process per epoch, which extends the plan's C11 (training and generation in separate processes) to
the generation stages themselves.

Run from this directory, with the GPU-capable interpreter and PYTHONPATH at the vision-unlearning repo root::

    PY=".../sd-interpretability/.venv/Scripts/python.exe"
    PYTHONPATH=/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning \
        HF_HUB_DISABLE_XET=1 "$PY" check_generation_gate.py --phase main
    PYTHONPATH=... HF_HUB_DISABLE_XET=1 "$PY" check_generation_gate.py --phase reload

Writes ``assets/generation_gate.json`` and the generated PNGs under ``assets/generation_gate/``.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_IMAGE_DIR = _OUT / "generation_gate"
_LOG = _OUT / "generation_gate_monitor.log"
_SELECTION = _HERE.parent / "every_epoch" / "assets" / "selection_people.json"

_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_RESOLUTION = 512
_SEED = 42
_N_PROMPTS = 5
_VARIANT = "fp16"


def _save(result: Dict[str, Any]) -> None:
    """Write the result JSON after every phase.

    The first attempt at this stage wrote it only in a ``finally`` block, and the watchdog's
    ``os._exit(137)`` bypasses ``finally`` -- so the run that proved the fp32 weight problem left no
    JSON behind at all, only the monitor log. Evidence has to survive a hard abort.
    """
    (_OUT / "generation_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _prompts() -> List[str]:
    """The campaign's own prompts, read from the every-epoch selection rather than retyped.

    The template is ``An image of {hf_name}``, identical to
    ``ablations/every_epoch/make_epoch_grid.py`` line 227 and ``metric_progression.py`` line 84, so the gate
    measures the cost and the determinism of exactly the calls the campaign will make.
    """
    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    names = [selection["target"]["hf_name"]] + [r["hf_name"] for r in selection["receivers"]]
    return [f"An image of {name}" for name in names[:_N_PROMPTS]]


def _load_pipeline() -> Dict[str, Any]:
    """Phase 1 -- the exact construction ``generate_dataset`` performs, timed, and C7 answered."""
    import torch
    from diffusers import AutoPipelineForText2Image

    started = time.time()
    pipeline = AutoPipelineForText2Image.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant=_VARIANT,
    ).to("cuda")
    seconds = round(time.time() - started, 1)
    return {
        "pipeline_class": type(pipeline).__name__,
        "variant": _VARIANT,
        "load_seconds": seconds,
        "unet_sample_size": int(pipeline.unet.config.sample_size),
        "vae_scale_factor": int(pipeline.vae_scale_factor),
        "default_resolution_without_height_width": int(pipeline.unet.config.sample_size) * int(pipeline.vae_scale_factor),
        "vae_dtype": str(pipeline.vae.dtype),
        "vae_force_upcast": bool(pipeline.vae.config.force_upcast),
        "pipeline": pipeline,
    }


def _generate(
    prompts: List[str],
    out_dir: Path,
    prefix: str,
    pipeline: Optional[Any] = None,
    model_base_name: Optional[str] = None,
) -> Dict[str, Any]:
    """One ``generate_dataset`` call at 512x512 and seed 42, timed, with the file names it produced."""
    from vision_unlearning.utils.data_generation import generate_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = [f"{prefix}_{index}.png" for index in range(len(prompts))]
    started = time.time()
    metadata = generate_dataset(
        model_base_name=model_base_name,
        lora_name=None,
        prompts=prompts,
        output_path=str(out_dir),
        filenames=filenames,
        seeds=[_SEED],
        batch_size=1,
        model_pipeline=pipeline,
        height=_RESOLUTION,
        width=_RESOLUTION,
        variant=None if model_base_name is None else _VARIANT,
    )
    seconds = round(time.time() - started, 1)
    return {
        "prompts": prompts,
        "filenames": [record["file_name"] for record in metadata],
        "total_seconds": seconds,
        "seconds_per_image": round(seconds / len(prompts), 1),
        "loaded_its_own_pipeline": model_base_name is not None,
    }


def _compare(dir_a: Path, names_a: List[str], dir_b: Path, names_b: List[str]) -> Dict[str, Any]:
    """Pixel comparison of two runs, image by image, plus the image size actually produced."""
    per_image: List[Dict[str, Any]] = []
    for name_a, name_b in zip(names_a, names_b):
        array_a = np.asarray(Image.open(dir_a / name_a).convert("RGB"))
        array_b = np.asarray(Image.open(dir_b / name_b).convert("RGB"))
        identical = bool(np.array_equal(array_a, array_b))
        per_image.append({
            "a": name_a,
            "b": name_b,
            "size": list(array_a.shape),
            "identical": identical,
            "max_absolute_difference": int(np.abs(array_a.astype(int) - array_b.astype(int)).max()),
        })
    return {
        "n_compared": len(per_image),
        "n_identical": sum(1 for entry in per_image if entry["identical"]),
        "all_identical": all(entry["identical"] for entry in per_image),
        "per_image": per_image,
    }


def main() -> None:
    import torch

    _OUT.mkdir(parents=True, exist_ok=True)
    check_headroom()
    # 5 s, not 15: the first attempt at this stage went from 4.51 GB free to 0.31 GB free inside a
    # single 15 s tick while reading full-precision weights. The floor is only as good as the sampling.
    monitor = ResourceMonitor(_LOG, interval_s=5.0)
    monitor.start()

    result: Dict[str, Any] = {
        "stage": "S2 generation gate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": _MODEL_ID,
        "resolution": _RESOLUTION,
        "seed": _SEED,
        "status": "started",
    }
    _save(result)
    prompts = _prompts()
    try:
        # Phase 1: pipeline construction with safety_checker=None (C7).
        loaded = _load_pipeline()
        pipeline = loaded.pop("pipeline")
        result["load"] = loaded
        result["safety_checker_accepted"] = True
        _save(result)

        # Phases 2 and 3: the same call twice on one loaded pipeline (C6, marginal cost).
        run_a = _generate(prompts, _IMAGE_DIR / "run_a", "a", pipeline=pipeline)
        result["run_a"] = run_a
        _save(result)
        run_b = _generate(prompts, _IMAGE_DIR / "run_b", "b", pipeline=pipeline)
        result["run_b"] = run_b
        result["determinism_same_pipeline"] = _compare(
            _IMAGE_DIR / "run_a", run_a["filenames"], _IMAGE_DIR / "run_b", run_b["filenames"],
        )
        _save(result)

        # Phase 4 is NOT here. Building a second Stable Diffusion XL pipeline in a process that has
        # already held one exhausted system memory and the watchdog killed the run at 1.31 GB free,
        # even after del + empty_cache had returned the video memory. That is a campaign constraint --
        # one pipeline per process -- so the reload comparison runs as its own invocation, below.
        del pipeline
        torch.cuda.empty_cache()

        result["peak_torch_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        result["peak_torch_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
        result["status"] = "ok"
    except Exception as error:  # the failure IS the measurement -- record it, do not swallow it silently
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        monitor.stop()
        result["peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        result["min_ram_free_gb"] = round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None
        _save(result)
        print(f"GENERATION_GATE_DONE {result['status']} {time.strftime('%H:%M:%S')}", flush=True)


def main_reload() -> None:
    """Phase 4, as its own process: is a run reproducible across a pipeline reload?

    The campaign generates each epoch's images from a different adapter, so it reloads between
    stages; if the noise draw depended on anything the reload changes, every cross-epoch comparison
    would be reading a difference that is not the adapter's. Compares one image, generated through
    the ``model_base_name`` branch, against the image the main phase already wrote for the same
    prompt and the same seed. Merges its result into the existing JSON instead of overwriting it.
    """
    import torch

    check_headroom()
    monitor = ResourceMonitor(_LOG, interval_s=5.0)
    monitor.start()

    result: Dict[str, Any] = json.loads((_OUT / "generation_gate.json").read_text(encoding="utf-8"))
    prompts = _prompts()
    try:
        run_c = _generate(prompts[:1], _IMAGE_DIR / "run_c", "c", model_base_name=_MODEL_ID)
        result["run_c"] = run_c
        result["determinism_across_reload"] = _compare(
            _IMAGE_DIR / "run_a", result["run_a"]["filenames"][:1], _IMAGE_DIR / "run_c", run_c["filenames"],
        )
        result["reload_status"] = "ok"
    except Exception as error:
        result["reload_status"] = "failed"
        result["reload_error"] = f"{type(error).__name__}: {error}"
        result["reload_traceback"] = traceback.format_exc()
        print(result["reload_traceback"], flush=True)
    finally:
        monitor.stop()
        result["reload_peak_vram_used_gb"] = round(monitor.peak_vram_used_gb, 3)
        result["reload_min_ram_free_gb"] = (
            round(monitor.min_ram_free_gb, 3) if monitor.min_ram_free_gb != float("inf") else None
        )
        result["reload_peak_torch_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        _save(result)
        print(f"GENERATION_GATE_DONE {result['reload_status']} {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["main", "reload"], default="main",
        help="'main' loads one pipeline and runs the load/determinism/cost phases; 'reload' is the "
             "cross-reload comparison, which MUST be a separate process (see the module docstring).",
    )
    if parser.parse_args().phase == "main":
        main()
    else:
        main_reload()
