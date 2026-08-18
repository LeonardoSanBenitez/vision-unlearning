'''Does the adapter trained at 512 still forget when used at 768?

The campaign trained its adapters at resolution 512, with the size micro-conditioning the trainer
derives from each training photograph (the source size, 250x250 for this dataset, presented at
512x512). Generation now happens at 768 with (1024, 1024) declared. The adapter is a low-rank delta
on the denoiser's attention projections, fitted under conditioning vectors that never occur at
generation time, so whether its effect survives the move is a question about the data rather than
about the theory.

`clip_diff = clip_on - clip_off` is computed in both regimes for the same three entities and the
same seed:

    512, as trained   base and adapted images at 512 with the campaign's own settings
    768, as deployed  base and adapted images at 768 with the frozen generation hyperparameters

Only the adapted 512 images are missing; everything else already exists (`rescue_grid` supplies the
512 baselines, `validate_generation_768` both halves at 768), so this script generates three images
and then reads the rest off disk.

    PYTHONPATH=<repo root> HF_HUB_DISABLE_XET=1 python adapter_transfer.py --stage generate
    PYTHONPATH=<repo root> python adapter_transfer.py --stage report
'''
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from run_campaign import _MODEL_ID, _VARIANT, _generation_order
from watchdog import ResourceMonitor, check_headroom

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_IMAGES = _OUT / "adapter_transfer"
_ADAPTER = _OUT / "campaign_model" / "seed42" / "epoch-200"
_RESULT = _OUT / "adapter_transfer.json"
_METRICS = _OUT / "render_quality_metrics.json"

_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]
_SEED = 42

# Where the three other cells of the comparison already live.
_OFF_512 = _OUT / "rescue_grid" / "campaign_defaults_{entity}_seed42.png"
_OFF_512_DEPLOYED = _OUT / "rescue_grid" / "original1024_target1024_g7.5_{entity}_seed42.png"
_OFF_768 = _OUT / "validate_generation_768" / "off_{entity}_seed42.png"
_ON_768 = _OUT / "validate_generation_768" / "on_epoch200_{entity}_seed42.png"


def _generate(entity: str, regime: str) -> None:
    """One entity, one call, one process.

    Each generate_dataset call builds its own pipeline, and two Stable Diffusion XL pipelines in one
    process is the condition measured to drive free system memory to about 1.3 GB -- so the caller
    invokes this once per entity. Generating the three in one call would also put entities at draw
    indices 1 and 2, while every image this is compared against was generated alone at index 0.
    """
    from vision_unlearning.utils.data_generation import GenerationRuntime, generate_dataset

    check_headroom()
    _IMAGES.mkdir(parents=True, exist_ok=True)
    assert (_ADAPTER / "pytorch_lora_weights.safetensors").is_file(), f"adapter missing: {_ADAPTER}"

    prompt_of = {entry["name"]: entry["prompt"] for entry in _generation_order()}
    # "as_trained" is the campaign's own 512 call, the regime the adapter was fitted in.
    # "deployed_conditioning" keeps 512 pixels but declares the sizes and guidance generation now
    # uses, which separates the two mismatches: if the adapter still acts here but not at 768, the
    # resolution is what breaks it; if it dies here too, the conditioning is.
    conditioning: Dict[str, Any] = ({} if regime == "as_trained" else
                                    {"guidance_scale": 7.5, "original_size": (1024, 1024),
                                     "target_size": (1024, 1024), "crops_coords_top_left": (0, 0)})
    prefix = "on512" if regime == "as_trained" else "on512deployed"
    filename = f"{prefix}_{entity}_seed{_SEED}.png"

    monitor = ResourceMonitor(_OUT / f"adapter_transfer_{prefix}_{entity}_monitor.log", interval_s=15.0)
    monitor.start()
    t0 = time.time()
    try:
        # The campaign's own 512 settings: no size arguments, default guidance, deterministic
        # kernels on -- matching the baselines this will be compared against.
        generate_dataset(
            model_base_name=_MODEL_ID,
            lora_name=str(_ADAPTER),
            prompts=[prompt_of[entity]],
            output_path=str(_IMAGES),
            filenames=[filename],
            seeds=[_SEED],
            batch_size=1,
            lora_requires_inversion=False,
            height=512, width=512,
            variant=_VARIANT,
            runtime=GenerationRuntime(device_map="balanced"),
            **conditioning,
        )
        print(f"generated {filename}", flush=True)
    finally:
        monitor.stop()
    print(f"ADAPTER_TRANSFER_GENERATE_DONE seconds={round(time.time() - t0, 1)} "
          f"peak_vram_used_gb={round(monitor.peak_vram_used_gb, 3)} "
          f"min_ram_free_gb={round(monitor.min_ram_free_gb, 3)}")


def _report() -> None:
    from PIL import Image

    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    prompt_of = {entry["name"]: entry["prompt"] for entry in _generation_order()}
    metric = MetricImageTextSimilarity(metrics=["clip"])

    def score(path: Path, prompt: str) -> float:
        with Image.open(path) as handle:
            return round(float(metric.score_batch_same_text([handle.convert("RGB")], prompt)[0]["clip"]), 3)

    rows: List[Dict[str, Any]] = []
    for entity in _ENTITIES:
        prompt = prompt_of[entity]
        cells = {
            "off_512": Path(str(_OFF_512).format(entity=entity)),
            "on_512": _IMAGES / f"on512_{entity}_seed{_SEED}.png",
            "off_512_deployed": Path(str(_OFF_512_DEPLOYED).format(entity=entity)),
            "on_512_deployed": _IMAGES / f"on512deployed_{entity}_seed{_SEED}.png",
            "off_768": Path(str(_OFF_768).format(entity=entity)),
            "on_768": Path(str(_ON_768).format(entity=entity)),
        }
        missing = [name for name, path in cells.items() if not Path(path).is_file()]
        if missing:
            print(f"{entity}: missing {missing}")
            continue
        scores = {name: score(Path(path), prompt) for name, path in cells.items()}
        rows.append({
            "entity": entity,
            **scores,
            "clip_diff_512_as_trained": round(scores["on_512"] - scores["off_512"], 3),
            "clip_diff_512_deployed_conditioning": round(
                scores["on_512_deployed"] - scores["off_512_deployed"], 3),
            "clip_diff_768_as_deployed": round(scores["on_768"] - scores["off_768"], 3),
        })

    print(f"{'entity':<24}{'512 as trained':>16}{'512 deployed':>16}{'768 deployed':>16}")
    for row in rows:
        print(f"{row['entity']:<24}{row['clip_diff_512_as_trained']:>16.2f}"
              f"{row['clip_diff_512_deployed_conditioning']:>16.2f}"
              f"{row['clip_diff_768_as_deployed']:>16.2f}")
    _RESULT.write_text(json.dumps({"seed": _SEED, "adapter": str(_ADAPTER), "rows": rows}, indent=2),
                       encoding="utf-8")
    print(f"ADAPTER_TRANSFER_REPORT_DONE entities={len(rows)} written={_RESULT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Does the 512-trained adapter still act at 768?")
    parser.add_argument("--stage", choices=["generate", "report"], required=True)
    parser.add_argument("--entity", choices=_ENTITIES, help="required for --stage generate")
    parser.add_argument("--regime", choices=["as_trained", "deployed_conditioning"],
                        default="as_trained")
    arguments = parser.parse_args()
    if arguments.stage == "generate":
        assert arguments.entity is not None, "--stage generate requires --entity"
        _generate(arguments.entity, arguments.regime)
    else:
        _report()


if __name__ == "__main__":
    main()
