"""Generation-only every-epoch trajectory from already-saved LoRA adapters (report artifact).

Given a model dir already populated with ``epoch-{n}`` adapters (by the checkpoint hook), this generates
the TARGET concept at each saved epoch plus the base-model baseline (seed 42), computes ``clip_diff`` per
epoch, and writes ``demo_trajectory.json`` + a strip figure. It does NOT train, so it is fast and
low-memory (fp16 generation only) - used to build the trajectory when a long training run was interrupted
but its intermediate adapters survived. Paths resolve from __file__; run with PYTHONPATH at the repo root,
HF_TOKEN set, HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_ICARE_ASSETS = _ICARE_DIR / "assets"
_OUT = _THIS.parent / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TARGET = "bouvier des flandres dog"
_TASK: Literal["breeds"] = "breeds"
_METHOD: Literal["distil"] = "distil"


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

    logger = get_logger("every_epoch_gen_traj")
    setup_loggers(modules_info=["unlearning"])

    parser = argparse.ArgumentParser(description="Generation-only trajectory from saved adapters.")
    parser.add_argument("--model-dir", default=str(_OUT / "models" / "breeds_bouvier_demo_distil_30"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    epochs: List[int] = sorted(
        int(p.name.split("-")[1]) for p in model_dir.glob("epoch-*")
        if (p / "pytorch_lora_weights.safetensors").exists()
    )
    assert epochs, f"no epoch adapters in {model_dir}"

    target_pre, target_over = get_target_overwrite(_TASK, _METHOD, _TARGET)
    prompt = f"An image of {target_pre}"
    logger.info("target=%r prompt=%r epochs=%s", target_pre, prompt, epochs)

    gen_dir = _OUT / "demo_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    def gen(lora: Any, fname: str) -> Path:
        generate_dataset(
            model_base_name=_MODEL_ID, lora_name=lora, prompts=[prompt],
            output_path=str(gen_dir), seeds=[args.seed], filenames=[fname],
            batch_size=2, lora_requires_inversion=False,
        )
        return gen_dir / fname

    off_path = gen(None, f"off_seed{args.seed}.png")
    on_paths: Dict[int, Path] = {n: gen(str(model_dir / f"epoch-{n}"), f"on_epoch{n}_seed{args.seed}.png") for n in epochs}

    m = MetricImageTextSimilarity(metrics=["clip"])
    clip_off = m.score_batch_same_text([Image.open(off_path).convert("RGB")], prompt)[0]["clip"]
    traj: List[Dict[str, Any]] = []
    for n in epochs:
        clip_on = m.score_batch_same_text([Image.open(on_paths[n]).convert("RGB")], prompt)[0]["clip"]
        traj.append({"epoch": n, "clip_on": clip_on, "clip_off": clip_off, "clip_diff": clip_on - clip_off})
        logger.info("epoch %d: clip_on=%.3f clip_off=%.3f clip_diff=%.3f", n, clip_on, clip_off, clip_on - clip_off)

    result = {
        "task": _TASK, "method": _METHOD, "target": target_pre, "overwrite": target_over,
        "prompt": prompt, "seed": args.seed, "epochs": epochs, "trajectory": traj,
        "off_image": str(off_path), "on_images": {n: str(p) for n, p in on_paths.items()},
        "note": "generation-only from adapters saved during an interrupted 30-epoch training run (epochs 1-10 survived)",
    }
    (_OUT / "demo_trajectory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    cols = 1 + len(epochs)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.6))
    axes[0].imshow(np.asarray(Image.open(off_path).convert("RGB")))
    axes[0].set_title("base model (off)\nno unlearning", fontsize=9)
    axes[0].axis("off")
    for i, n in enumerate(epochs, start=1):
        cd = next(t["clip_diff"] for t in traj if t["epoch"] == n)
        axes[i].imshow(np.asarray(Image.open(on_paths[n]).convert("RGB")))
        axes[i].set_title(f"epoch {n}\nclip_diff {cd:.2f}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(
        f"SPARE unlearning of '{target_pre}' -> '{target_over}', per epoch (seed {args.seed})\n"
        f"prompt: {prompt}  |  clip_diff = clip_on - clip_off (more negative = more forgotten)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    strip = _OUT / "demo_trajectory_strip.png"
    fig.savefig(strip, dpi=110)
    plt.close(fig)
    logger.info("strip -> %s", strip)
    print("DEMO_OK", str(strip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
