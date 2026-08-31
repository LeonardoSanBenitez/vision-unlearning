'''The Stable Diffusion 1.4 every-epoch campaign, addressed from this ablation.

Every cross-model figure and measurement in the Stable Diffusion XL ablation needs the same three
things from the older campaign -- an entity's images by epoch, its `clip_diff` by epoch, and the
noise floor its numbers are judged against -- and each of them is easy to get subtly wrong. The image
files are named by an entity's POSITION in that run's generation order, so the only safe way to find
one is through that run's own manifest; pairing a position with the wrong entity produces a complete,
plausible, wrong figure. Putting the three lookups here means that is written once.

Nothing here regenerates or re-scores anything: the every-epoch campaign is finished, and these are
reads of its artifacts.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
ASSETS = _HERE.parent / "every_epoch" / "assets"
IMAGES = ASSETS / "epoch_grid_campaign_people"
_GRID_TEMPLATE = "epoch_grid_campaign_people_seed{seed}.json"
_MANIFEST_TEMPLATE = "manifest_s{seed}.json"
_FLOOR = ASSETS / "noise_floor_people.json"

MODEL_ID = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512


def generation_order(seed: int) -> List[str]:
    '''The entity order the run generated in, read from its own manifest. Never retyped.'''
    manifest = json.loads((IMAGES / _MANIFEST_TEMPLATE.format(seed=seed)).read_text(encoding="utf-8"))
    order: List[str] = manifest["generation_order"]
    return order


def entity_cells(seed: int, entity: str) -> Tuple[Dict[Any, Path], Dict[Any, float], List[int]]:
    '''(image path by epoch, `clip_diff` by epoch, epochs) for one entity of one seed.

    `None` is the off-baseline epoch, matching the manifest convention this ablation uses. The
    off-baseline has no `clip_diff` entry, because it is zero by construction.
    '''
    manifest = json.loads((IMAGES / _MANIFEST_TEMPLATE.format(seed=seed)).read_text(encoding="utf-8"))
    grid = json.loads((ASSETS / _GRID_TEMPLATE.format(seed=seed)).read_text(encoding="utf-8"))
    order: List[str] = manifest["generation_order"]
    assert entity in order, f"{entity} is not in the Stable Diffusion 1.4 generation order {order}"
    index = order.index(entity)
    epochs: List[int] = manifest["epochs"]
    assert grid["epochs"] == epochs, \
        f"the grid result and the image manifest disagree about epochs: {grid['epochs']} against {epochs}"

    paths: Dict[Any, Path] = {None: IMAGES / f"off_s{seed}_b{index}.png"}
    values: Dict[Any, float] = {}
    for row, epoch in enumerate(epochs, start=1):
        paths[epoch] = IMAGES / f"on_ep{epoch}_s{seed}_b{index}.png"
        values[epoch] = float(grid["clip_diff"][f"{row},{index}"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    assert not missing, f"Stable Diffusion 1.4 images missing: {missing[:5]}"
    return paths, values, epochs


def noise_floor_summary() -> Dict[str, float]:
    '''The ten-entity, two-seed `clip_diff` floor of the Stable Diffusion 1.4 campaign.

    A different construction from this ablation's own floor (one entity over six seeds), which is why
    the two are always printed with their construction beside them rather than as one number.
    '''
    payload = json.loads(_FLOOR.read_text(encoding="utf-8"))
    summary: Dict[str, float] = payload["summary"]
    return summary
