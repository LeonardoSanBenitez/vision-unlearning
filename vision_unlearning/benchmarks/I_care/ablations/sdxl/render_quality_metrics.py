'''Two numbers per generated image, so "the render is good" and "the render collapsed" stop being
adjectives and become measurements.

Both are computed over images that already exist on disk; nothing is generated here.

1. `clip_own_prompt` -- the library's own `MetricImageTextSimilarity(metrics=['clip'])` between the
   image and the prompt it was generated from. This is the same primitive `pipeline_06`, the
   every-epoch ablation and `clip_diff_campaign.py` use, so the values sit on the familiar scale.
   It answers "does the image agree with the text", and it is deliberately NOT trusted on its own:
   a flat vector-art collage carrying the entity's surname in large letters can score well.

2. `flat_colour_fraction` -- the fraction of pixels that fall in the sixteen most populated colour
   bins, after quantising each channel to 32 levels (5 bits). A photograph spreads its pixels over
   thousands of bins because of shading and grain, so this fraction is small; the failed Stable
   Diffusion XL renders are flat posterised vector art with a handful of saturated fills, so it is
   large. It is a shape-of-the-histogram measure and it needs no model.

A metric is only worth using if it gets a case with a known answer right, so the script runs a
POSITIVE CONTROL before reporting anything else. Block 1 of the overnight evidence run generated the
same ten entities alone at five seeds; seed 42's ten renders are photographic and seed 45's ten are
degenerate, judged by eye from `assets/overnight_block1_baselines.png`. The control prints both
groups' ranges and computes -- rather than asserts -- whether they separate.

    python render_quality_metrics.py

Writes `assets/render_quality_metrics.json`: one record per image with its labels, plus the control.
'''
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_OVERNIGHT_JSON = _OUT / "overnight_evidence.json"
_RESCUE_JSON = _OUT / "rescue_grid.json"
_RESULT = _OUT / "render_quality_metrics.json"

# The positive control: two groups of ten images whose verdict was settled by eye on the contact
# sheet before any number existed here.
_CONTROL_GOOD_SEED = 42
_CONTROL_BAD_SEED = 45

_QUANTISATION_LEVELS = 32
_TOP_BINS = 16


def flat_colour_fraction(image_path: Path) -> float:
    '''Fraction of pixels living in the sixteen most populated 32-level colour bins.

    Near 0 for a photograph (pixel values spread over many bins), near 1 for flat vector art.
    '''
    from PIL import Image

    with Image.open(image_path) as handle:
        rgb = handle.convert("RGB")
        pixels = list(rgb.getdata())
    step = 256 // _QUANTISATION_LEVELS
    counts = Counter((r // step, g // step, b // step) for r, g, b in pixels)
    top = sum(count for _, count in counts.most_common(_TOP_BINS))
    return round(top / len(pixels), 4)


def _overnight_records(prompt_of: Dict[str, str]) -> List[Dict[str, Any]]:
    '''One record per image of the overnight evidence run, carrying the labels each block varies.'''
    payload = json.loads(_OVERNIGHT_JSON.read_text(encoding="utf-8"))
    blocks = payload["blocks"]
    records: List[Dict[str, Any]] = []

    for seed, per_entity in blocks["1_baselines_alone"]["per_seed"].items():
        for entity, path in per_entity.items():
            records.append({"block": "1_baselines_alone", "seed": int(seed), "entity": entity,
                            "draw_index": 0, "condition": "campaign_defaults", "path": path,
                            "prompt": prompt_of[entity]})

    for entity, paths in blocks["2_position_sweep"]["per_entity_by_position"].items():
        for position, path in enumerate(paths):
            records.append({"block": "2_position_sweep", "seed": 42, "entity": entity,
                            "draw_index": position, "condition": "campaign_defaults", "path": path,
                            "prompt": prompt_of[entity]})

    for entity, paths in blocks["3_reseeded_control"]["per_entity_by_call_index"].items():
        for call_index, path in enumerate(paths):
            records.append({"block": "3_reseeded_control", "seed": 42, "entity": entity,
                            "draw_index": 0, "condition": f"reseeded_call_{call_index:02d}",
                            "path": path, "prompt": prompt_of[entity]})

    for entity, per_setting in blocks["4_parameter_grid"]["per_entity"].items():
        for label, path in per_setting.items():
            records.append({"block": "4_parameter_grid", "seed": 42, "entity": entity,
                            "draw_index": 0, "condition": label, "path": path,
                            "prompt": prompt_of[entity]})
    return records


def _rescue_records(prompt_of: Dict[str, str]) -> List[Dict[str, Any]]:
    '''One record per image of the rescue grid, when that run has already produced its JSON.'''
    if not _RESCUE_JSON.is_file():
        return []
    payload = json.loads(_RESCUE_JSON.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for row in payload["images"]:
        records.append({"block": "rescue_grid", "seed": row["seed"], "entity": row["entity"],
                        "draw_index": 0, "condition": row["condition"], "path": row["path"],
                        "prompt": prompt_of[row["entity"]]})
    return records


def _sign_off_records(prompt_of: Dict[str, str]) -> List[Dict[str, Any]]:
    '''The 768-pixel sign-off run: off-baselines, on-images and the reproducibility repeats.'''
    result = _OUT / "validate_generation_768.json"
    if not result.is_file():
        return []
    payload = json.loads(result.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for row in payload.get("off_baselines", []):
        records.append({"block": "sign_off_768_off", "seed": row["seed"], "entity": row["entity"],
                        "draw_index": 0, "condition": "frozen_generation_hyperparameters",
                        "path": row["path"], "prompt": prompt_of[row["entity"]]})
    for row in payload.get("on_images", []):
        records.append({"block": "sign_off_768_on", "seed": row["seed"], "entity": row["entity"],
                        "draw_index": 0, "condition": f"adapter_epoch{row['epoch']}",
                        "path": row["path"], "prompt": prompt_of[row["entity"]]})
    for row in payload.get("reproducibility", []):
        records.append({"block": "sign_off_768_repeat", "seed": row["seed"], "entity": row["entity"],
                        "draw_index": 0, "condition": "regenerated_in_the_same_process",
                        "path": row["path"], "prompt": prompt_of[row["entity"]]})
    return records


def _verify_refactored_records(prompt_of: Dict[str, str]) -> List[Dict[str, Any]]:
    '''The images produced through the refactored library function, including four unused seeds.'''
    records: List[Dict[str, Any]] = []
    for stage in ("base", "adapter"):
        result = _OUT / f"verify_refactored_{stage}.json"
        if not result.is_file():
            continue
        payload = json.loads(result.read_text(encoding="utf-8"))
        entity = payload["entity"]
        for seed, path in zip(payload["seeds"], payload["images"]):
            records.append({"block": f"verify_refactored_{stage}", "seed": seed, "entity": entity,
                            "draw_index": 0, "condition": "frozen_generation_hyperparameters",
                            "path": path, "prompt": prompt_of[entity]})
    return records


def _campaign_off_records(prompt_of: Dict[str, str]) -> List[Dict[str, Any]]:
    '''The campaign's own off-baselines: ten entities in ONE call, one row per entity and seed.

    These are the images the campaign actually produced, and the ones every voided number was
    computed from. Scoring them with the same metric as everything else is what makes the
    call-shape comparison a measurement rather than an impression.
    '''
    records: List[Dict[str, Any]] = []
    for seed in (42, 43):
        manifest = _OUT / f"campaign_seed{seed}.json"
        if not manifest.is_file():
            continue
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            if row["epoch"] is not None:
                continue
            records.append({"block": "campaign_off_baseline", "seed": row["seed"],
                            "entity": row["entity"], "draw_index": None,
                            "condition": "ten_prompts_in_one_call", "path": row["path"],
                            "prompt": prompt_of[row["entity"]]})
    return records


def _group_range(values: List[float]) -> Tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def main() -> None:
    from PIL import Image

    from run_campaign import _generation_order
    from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity

    prompt_of = {entry["name"]: entry["prompt"] for entry in _generation_order()}

    records = (_overnight_records(prompt_of) + _rescue_records(prompt_of)
               + _campaign_off_records(prompt_of) + _sign_off_records(prompt_of)
               + _verify_refactored_records(prompt_of))
    missing = [r["path"] for r in records if not Path(r["path"]).is_file()]
    assert not missing, f"{len(missing)} listed images are not on disk, first: {missing[:3]}"

    metric = MetricImageTextSimilarity(metrics=["clip"])
    for index, record in enumerate(records):
        path = Path(record["path"])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            record["clip_own_prompt"] = round(
                float(metric.score_batch_same_text([image], record["prompt"])[0]["clip"]), 3)
        record["flat_colour_fraction"] = flat_colour_fraction(path)
        if (index + 1) % 25 == 0:
            print(f"scored {index + 1} of {len(records)} images", flush=True)

    good = [r for r in records
            if r["block"] == "1_baselines_alone" and r["seed"] == _CONTROL_GOOD_SEED]
    bad = [r for r in records
           if r["block"] == "1_baselines_alone" and r["seed"] == _CONTROL_BAD_SEED]
    control: Optional[Dict[str, Any]] = None
    if good and bad:
        good_flat = [r["flat_colour_fraction"] for r in good]
        bad_flat = [r["flat_colour_fraction"] for r in bad]
        good_clip = [r["clip_own_prompt"] for r in good]
        bad_clip = [r["clip_own_prompt"] for r in bad]
        control = {
            "known_good_group": f"block 1, seed {_CONTROL_GOOD_SEED}, n={len(good)}",
            "known_bad_group": f"block 1, seed {_CONTROL_BAD_SEED}, n={len(bad)}",
            "flat_colour_fraction_good_min_mean_max": _group_range(good_flat),
            "flat_colour_fraction_bad_min_mean_max": _group_range(bad_flat),
            "flat_colour_fraction_separates": max(good_flat) < min(bad_flat),
            "clip_own_prompt_good_min_mean_max": _group_range(good_clip),
            "clip_own_prompt_bad_min_mean_max": _group_range(bad_clip),
            "clip_own_prompt_separates": min(good_clip) > max(bad_clip),
        }
        print("POSITIVE CONTROL, ten known-good against ten known-bad images")
        print(f"  flat_colour_fraction good min/mean/max = {_group_range(good_flat)}")
        print(f"  flat_colour_fraction bad  min/mean/max = {_group_range(bad_flat)}")
        print(f"  good maximum {max(good_flat)} < bad minimum {min(bad_flat)}: "
              f"{max(good_flat) < min(bad_flat)}")
        print(f"  clip_own_prompt good min/mean/max = {_group_range(good_clip)}")
        print(f"  clip_own_prompt bad  min/mean/max = {_group_range(bad_clip)}")
        print(f"  good minimum {min(good_clip)} > bad maximum {max(bad_clip)}: "
              f"{min(good_clip) > max(bad_clip)}")

    _RESULT.write_text(json.dumps({"records": records, "positive_control": control,
                                   "quantisation_levels": _QUANTISATION_LEVELS,
                                   "top_bins": _TOP_BINS}, indent=2), encoding="utf-8")
    print(f"RENDER_QUALITY_METRICS_DONE images={len(records)} written={_RESULT}")


if __name__ == "__main__":
    main()
