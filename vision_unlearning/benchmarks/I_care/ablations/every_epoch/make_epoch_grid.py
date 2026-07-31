"""Every-epoch x every-entity grid (the W6 deliverable format), for the breeds selection.

Rows = [original base model, then each saved epoch]; columns = the 10 selected breeds ordered by canonical
interference (target at column 0). Each cell is the entity's own concept prompt generated from that epoch's
adapter, seed 42, batch_size=1 (dedicated-VRAM path). Reuses the base (off) and epoch-10 (on) images already
produced by seed_validation.py; generates the intermediate epochs it is missing. Writes epoch_grid.png and
epoch_grid.json (clip_diff per cell). Run with PYTHONPATH at the repo root, HF_TOKEN set, HF_HUB_DISABLE_XET=1.
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


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from vision_unlearning.utils.data_generation import generate_dataset
    from vision_unlearning.metrics import MetricImageTextSimilarity
    from vision_unlearning.datasets.testbed import get_target_overwrite
    from vision_unlearning.utils.logger import get_logger, setup_loggers

    logger = get_logger("epoch_grid")
    setup_loggers(modules_info=["unlearning"])

    model_dir = _OUT / "models" / "breeds_bouvier_demo_distil_30"
    epochs: List[int] = sorted(
        int(p.name.split("-")[1]) for p in model_dir.glob("epoch-*")
        if (p / "pytorch_lora_weights.safetensors").exists()
    )
    logger.info("epochs available: %s", epochs)

    sel = json.loads((_OUT / "selection_breeds.json").read_text(encoding="utf-8"))
    breeds: List[Tuple[str, float]] = [(sel["target"]["name"], sel["target"]["self_clip_diff"])]
    breeds += [(r["name"], r["clip_diff"]) for r in sel["receivers"]]
    breeds.sort(key=lambda x: x[1])  # most interfered (target) first
    breed_pre = {n: get_target_overwrite(_TASK, _METHOD, n)[0] for n, _ in breeds}
    prompt_of = {n: f"An image of {breed_pre[n]}" for n, _ in breeds}

    grid_dir = _OUT / "epoch_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    seedval = _OUT / "seed_validation"  # base_s42_b{bi}_t0.png (off), on_s42_b{bi}_t0.png (epoch-10)

    def off_path(bi: int) -> Path:
        reuse = seedval / f"base_s{_SEED}_b{bi}_t0.png"
        return reuse if reuse.exists() else grid_dir / f"off_b{bi}.png"

    def on_path(bi: int, n: int) -> Path:
        if n == max(epochs) and (seedval / f"on_s{_SEED}_b{bi}_t0.png").exists():
            return seedval / f"on_s{_SEED}_b{bi}_t0.png"
        return grid_dir / f"on_ep{n}_b{bi}.png"

    # generate any missing images, one generate_dataset call per (epoch or baseline), batch_size=1
    def gen(lora: Any, targets: List[Tuple[int, Path]]) -> None:
        todo = [(bi, p) for bi, p in targets if not p.exists()]
        if not todo:
            return
        prompts = [prompt_of[breeds[bi][0]] for bi, _ in todo]
        generate_dataset(
            model_base_name=_MODEL_ID, lora_name=lora, prompts=prompts, output_path=str(grid_dir),
            seeds=[_SEED], filenames=[p.name for _, p in todo], batch_size=1, lora_requires_inversion=False,
        )

    logger.info("ensuring baseline (off) images ...")
    gen(None, [(bi, off_path(bi)) for bi in range(len(breeds))])
    for n in epochs:
        logger.info("ensuring epoch-%d (on) images ...", n)
        gen(str(model_dir / f"epoch-{n}"), [(bi, on_path(bi, n)) for bi in range(len(breeds))])

    # clip_diff per cell
    m = MetricImageTextSimilarity(metrics=["clip"])
    rows = ["original"] + [f"epoch {n}" for n in epochs]
    cell_img: Dict[Tuple[int, int], Path] = {}
    clip_diff: Dict[Tuple[int, int], float] = {}
    for bi, (name, _) in enumerate(breeds):
        off = Image.open(off_path(bi)).convert("RGB")
        clip_off = m.score_batch_same_text([off], prompt_of[name])[0]["clip"]
        cell_img[(0, bi)] = off_path(bi)
        clip_diff[(0, bi)] = 0.0
        for ri, n in enumerate(epochs, start=1):
            p = on_path(bi, n)
            cell_img[(ri, bi)] = p
            on = Image.open(p).convert("RGB")
            clip_diff[(ri, bi)] = m.score_batch_same_text([on], prompt_of[name])[0]["clip"] - clip_off

    nrows, ncols = len(rows), len(breeds)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.9 * nrows))
    for ri in range(nrows):
        for bi in range(ncols):
            ax = axes[ri][bi]
            ax.imshow(np.asarray(Image.open(cell_img[(ri, bi)]).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            if ri > 0:
                ax.set_title(f"{clip_diff[(ri, bi)]:.0f}", fontsize=6)
            if bi == 0:
                ax.set_ylabel(rows[ri], fontsize=8, rotation=0, ha="right", va="center")
    for bi, (name, cd) in enumerate(breeds):
        axes[0][bi].set_title(f"{name.replace(' dog', '')}\ninterf {cd:.1f}", fontsize=6)
    fig.suptitle(
        "SPARE unlearning of 'a bouvier des flandres dog' -> 'a cat', per epoch (seed 42, distil, lr 6e-4)\n"
        "rows = original + saved epochs; columns = 10 selected breeds by canonical interference (target col 0); "
        "cell number = clip_diff vs original (more negative = more forgotten)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _OUT / "epoch_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    (_OUT / "epoch_grid.json").write_text(json.dumps({
        "epochs": epochs, "rows": rows,
        "breeds_by_interference": [{"name": n, "canonical_clip_diff": cd} for n, cd in breeds],
        "clip_diff": {f"{ri},{bi}": clip_diff[(ri, bi)] for ri in range(nrows) for bi in range(ncols)},
    }, indent=2), encoding="utf-8")
    logger.info("grid -> %s", out)
    print("GRID_OK", str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
