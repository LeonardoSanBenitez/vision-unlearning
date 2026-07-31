"""Cross-session reproducibility sanity check for the every-epoch grid.

In a FRESH process, regenerate one image per selected breed from the epoch-10 adapter (same 10-prompt list,
same order, seed 42, batch_size=1) and compare pixel-for-pixel against the epoch-10 row already written by
make_epoch_grid.py (grid_dir/on_ep10_b{bi}.png). If the seed handling is correct, every pair is identical
(max abs pixel diff = 0). Writes sanity_reproduce.json. Run with PYTHONPATH at the repo root, HF_TOKEN set,
HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TASK: Literal["breeds"] = "breeds"
_METHOD: Literal["distil"] = "distil"
_SEED = 42
_EPOCH = 10


def main() -> int:
    import numpy as np
    from PIL import Image

    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("sanity_reproduce")
    setup_loggers(modules_info=["unlearning"])

    sel = json.loads((_OUT / "selection_breeds.json").read_text(encoding="utf-8"))
    breeds: List[Tuple[str, float]] = [(sel["target"]["name"], sel["target"]["self_clip_diff"])]
    breeds += [(r["name"], r["clip_diff"]) for r in sel["receivers"]]
    breeds.sort(key=lambda x: x[1])
    prompts = [f"An image of {get_target_overwrite(_TASK, _METHOD, n)[0]}" for n, _ in breeds]

    grid_dir = _OUT / "epoch_grid"
    sanity_dir = _OUT / "sanity_reproduce"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    adapter: Any = str(_OUT / "models" / "breeds_bouvier_demo_distil_30" / f"epoch-{_EPOCH}")

    filenames = [f"repro_b{bi}.png" for bi in range(len(breeds))]
    logger.info("regenerating epoch-%d for %d breeds (fresh process) ...", _EPOCH, len(breeds))
    generate_dataset(
        model_base_name=_MODEL_ID, lora_name=adapter, prompts=prompts, output_path=str(sanity_dir),
        seeds=[_SEED], filenames=filenames, batch_size=1, lora_requires_inversion=False,
    )

    results: List[Dict[str, Any]] = []
    for bi, (name, _) in enumerate(breeds):
        grid_img = grid_dir / f"on_ep{_EPOCH}_b{bi}.png"
        repro_img = sanity_dir / f"repro_b{bi}.png"
        a = np.asarray(Image.open(grid_img).convert("RGB"), dtype=np.int64)
        b = np.asarray(Image.open(repro_img).convert("RGB"), dtype=np.int64)
        max_abs = int(np.max(np.abs(a - b))) if a.shape == b.shape else -1
        results.append({"breed": name, "max_abs_pixel_diff": max_abs, "identical": max_abs == 0})
        logger.info("  %-38s max_abs_pixel_diff=%d", name, max_abs)

    overall = max(int(r["max_abs_pixel_diff"]) for r in results)
    out = {
        "epoch": _EPOCH, "seed": _SEED, "n_breeds": len(breeds),
        "max_abs_pixel_diff_over_breeds": overall,
        "all_pixel_identical": all(r["identical"] for r in results),
        "per_breed": results,
    }
    (_OUT / "sanity_reproduce.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("SANITY_OK all_identical=%s max_abs=%d" % (out["all_pixel_identical"], overall))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
