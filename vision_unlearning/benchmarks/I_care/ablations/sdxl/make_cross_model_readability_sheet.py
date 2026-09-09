'''S1 of PLAN-TASK-2026-08-12-SDXL: can a reader compare the two models at all, at their own sizes?

The two models are deliberately read at different resolutions (D3): Stable Diffusion 1.4 at 512, its
native size, and Stable Diffusion XL at 768, the largest this card reaches. Stable Diffusion XL at
512 renders unusable images for two seeds in five, so a common-resolution comparison would compare a
working model against a broken one.

That decision has a consequence the deliverable depends on: the paper section claims subtle,
subjective differences are visible ACROSS the two models. If a reader cannot put a 512-pixel image
and a 768-pixel image side by side and read them as comparable, that claim needs rethinking -- and
this figure, built from images already on disk, is what says so before 15 to 23 hours of graphics
time are spent.

Three entities, off-baselines only, seed 42, both models, at their true pixel sizes -- NOT rescaled
to a common size, because rescaling would answer a different question than the one asked.

Inputs, all already generated:
  * Stable Diffusion 1.4: `every_epoch/assets/epoch_grid_campaign_people/off_s42_b<index>.png`, with
    the index read from `epoch_grid_campaign_people_seed42.json`'s `entities_by_interference` -- the
    generation order that file itself records, never retyped.
  * Stable Diffusion XL: `assets/campaign_seed42/off_<entity>_seed42.png`, written by the runner this
    stage re-pointed at the frozen configuration.

    PYTHONPATH=<repo root> python make_cross_model_readability_sheet.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_EVERY_EPOCH_ASSETS = _HERE.parent / "every_epoch" / "assets"
_SD_GRID_MANIFEST = _EVERY_EPOCH_ASSETS / "epoch_grid_campaign_people_seed42.json"
_SD_IMAGES = _EVERY_EPOCH_ASSETS / "epoch_grid_campaign_people"
_SD_NOISE_FLOOR = _EVERY_EPOCH_ASSETS / "noise_floor_people.json"
_XL_IMAGES = _OUT / "campaign_seed42"
_FIGURE = _OUT / "cross_model_readability_sheet.png"
_RESULT = _OUT / "cross_model_readability_sheet.json"

_SEED = 42
_ENTITIES = ["Mark_Philippoussis", "Andy_Roddick", "Megawati_Sukarnoputri"]
_SD_RESOLUTION = 512
_XL_RESOLUTION = 768


def _stable_diffusion_image(entity: str) -> Path:
    '''The off-baseline of one entity in the Stable Diffusion 1.4 grid, found by its own generation index.'''
    manifest = json.loads(_SD_GRID_MANIFEST.read_text(encoding="utf-8"))
    names = [record["name"] for record in manifest["entities_by_interference"]]
    assert entity in names, f"{entity} is not in {_SD_GRID_MANIFEST}: {names}"
    return _SD_IMAGES / f"off_s{_SEED}_b{names.index(entity)}.png"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    noise_floor = json.loads(_SD_NOISE_FLOOR.read_text(encoding="utf-8"))
    pairs: List[Dict[str, Any]] = []
    for entity in _ENTITIES:
        pairs.append({
            "entity": entity,
            "stable_diffusion_image": str(_stable_diffusion_image(entity)),
            "stable_diffusion_xl_image": str(_XL_IMAGES / f"off_{entity}_seed{_SEED}.png"),
        })
    for pair in pairs:
        for key in ("stable_diffusion_image", "stable_diffusion_xl_image"):
            assert Path(pair[key]).is_file(), f"missing input image: {pair[key]}"

    figure, axes = plt.subplots(len(pairs), 2, figsize=(8, 4 * len(pairs)))
    for row, pair in enumerate(pairs):
        for column, (key, model, resolution) in enumerate((
            ("stable_diffusion_image", "Stable Diffusion 1.4", _SD_RESOLUTION),
            ("stable_diffusion_xl_image", "Stable Diffusion XL", _XL_RESOLUTION),
        )):
            with Image.open(pair[key]) as image:
                axes[row][column].imshow(image)
                measured = image.size
            axes[row][column].set_title(
                f"{pair['entity'].replace('_', ' ')}\n{model}, {measured[0]}x{measured[1]} pixels, seed {_SEED}",
                fontsize=9)
            axes[row][column].axis("off")
            pair[f"{key}_pixel_size"] = list(measured)
            assert measured[0] == resolution, \
                f"{pair[key]} is {measured[0]} pixels wide, expected {resolution}"

    figure.suptitle(
        "Off-baseline renderings of the same three entities, each model at its own resolution "
        f"(no rescaling), seed {_SEED}", fontsize=10)
    figure.tight_layout()
    figure.savefig(_FIGURE, dpi=150)
    plt.close(figure)

    result = {
        "figure": str(_FIGURE),
        "seed": _SEED,
        "entities": _ENTITIES,
        "stable_diffusion_resolution": _SD_RESOLUTION,
        "stable_diffusion_xl_resolution": _XL_RESOLUTION,
        "stable_diffusion_noise_floor_per_entity": {
            entity: noise_floor["per_entity"][entity] for entity in _ENTITIES},
        "stable_diffusion_noise_floor_summary": noise_floor["summary"],
        "stable_diffusion_xl_noise_floor_at_768": "not measured yet -- plan stage S8",
        "pairs": pairs,
    }
    _RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"CROSS_MODEL_READABILITY_SHEET_DONE rows={len(pairs)} figure={_FIGURE}")


if __name__ == "__main__":
    main()
