"""
3b_compute_act_fingerprints.py — Compute UNet cross-attention activation fingerprints.

For each entity in a task, runs Stable Diffusion 1.4 (original, no LoRA) and
captures mean cross-attention (attn2) hidden states across denoising steps and
seeds.  The resulting L2-normalized fingerprint per entity is saved to:

    assets/datasets/act_fingerprints_{task}_sd1.4.json

These fingerprints are consumed by ResultTemplateSimilarityMatrix when
similarity_metric='act'.

Run from: unlearning/unlearning-analysis/
Interpreter: C:/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe

Usage:
    python 3b_compute_act_fingerprints.py --task scenes
    python 3b_compute_act_fingerprints.py --task scenes --n-steps 5 --device cpu
    python 3b_compute_act_fingerprints.py --task scenes --dry-run

Dependencies (all in sd-interpretability venv):
    torch, diffusers, numpy, pandas, vision_unlearning (from sibling ../vision-unlearning)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Ensure vision_unlearning is importable from the sibling repo directory
# ---------------------------------------------------------------------------
_VU_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision-unlearning")
)
if _VU_PATH not in sys.path:
    sys.path.insert(0, _VU_PATH)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("compute_act_fingerprints")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
MODEL_SHORT_NAME = "sd1.4"
BASE_FOLDER = "assets"
DEFAULT_N_STEPS = 10
DEFAULT_SEEDS = [42, 43, 44, 45]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute UNet cross-attention activation fingerprints for all entities in a task."
    )
    parser.add_argument(
        "--task",
        choices=["scenes", "breeds", "people"],
        default="scenes",
        help="Task name (default: scenes).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Number of denoising steps per inference (default: {DEFAULT_N_STEPS}). "
             "Fewer steps = faster; 10 is a good balance for stable fingerprints.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Generation seeds (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--base-folder",
        default=BASE_FOLDER,
        help=f"Local base folder for data storage (default: {BASE_FOLDER}).",
    )
    parser.add_argument(
        "--replace-if-exists",
        action="store_true",
        default=False,
        help="Re-compute even if fingerprints file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without loading the model or computing anything.",
    )
    args = parser.parse_args()

    task: Literal["scenes", "breeds", "people"] = args.task  # type: ignore[assignment]
    n_steps: int = args.n_steps
    seeds: List[int] = args.seeds
    base_folder: str = args.base_folder

    # Lazy imports after CLI parsing — avoids slow torch/diffusers load on --help
    from vision_unlearning.benchmarks.I_care.configuration import (
        unlearning_algorithm_to_epochs,
    )
    from vision_unlearning.datasets.testbed import (
        get_metadata_filtered,
        get_target_overwrite,
    )
    from vision_unlearning.utils.mechanistic_interpretability import (
        SD_MODEL_NAME as MI_SD_MODEL_NAME,
        compute_act_fingerprints_for_task,
        get_act_fingerprints_path,
        save_act_fingerprints,
    )

    # --- Metadata ---
    metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(
        task, base_folder=base_folder
    )
    entity_names: List[str] = [e["name"] for e in metadata_filtered]

    # Build prompts using the canonical get_target_overwrite convention
    # (same approach as 3_compute_embeddings.py and result_templates.py 'dino')
    entity_prompts: Dict[str, str] = {
        name: f"An image of {get_target_overwrite(task, 'distil', name)[0]}"
        for name in entity_names
    }

    output_path = get_act_fingerprints_path(task, MODEL_SHORT_NAME, base_folder)

    # --- Dry run ---
    if args.dry_run:
        logger.info("DRY RUN — no computation will be performed.")
        logger.info("Task:         %s", task)
        logger.info("Entities:     %d", len(entity_names))
        logger.info("n_steps:      %d", n_steps)
        logger.info("seeds:        %s", seeds)
        logger.info("Output path:  %s", output_path)
        logger.info("File exists:  %s", os.path.exists(output_path))
        logger.info("First 5 entities and prompts:")
        for name, prompt in list(entity_prompts.items())[:5]:
            logger.info("  %-30s -> %s", name, prompt)
        sys.exit(0)

    # --- Skip if exists ---
    if os.path.exists(output_path) and not args.replace_if_exists:
        logger.info("Fingerprints already exist at %s. Use --replace-if-exists to recompute.", output_path)
        sys.exit(0)

    # --- Load model ---
    import torch
    from diffusers import StableDiffusionPipeline

    device: str = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    # GPU memory check before loading
    if device == "cuda" and torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        used_vram = total_vram - free_vram
        logger.info(
            "VRAM before load: %.2fGB used / %.2fGB total (%.1f%% used, %.2fGB free)",
            used_vram / 1024**3, total_vram / 1024**3,
            100.0 * used_vram / total_vram, free_vram / 1024**3,
        )
        if free_vram < 4.0 * 1024**3:
            logger.warning(
                "Less than 4GB VRAM free (%.2fGB). SD 1.4 needs ~3.5GB. Proceeding anyway — "
                "reduce other GPU loads if OOM errors occur.",
                free_vram / 1024**3,
            )

    logger.info("Loading Stable Diffusion 1.4 (%s)...", SD_MODEL_NAME)
    pipeline: Any = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,        # skip NSFW checker for speed
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)  # suppress tqdm per-image

    if device == "cuda" and torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        used_vram = total_vram - free_vram
        logger.info(
            "VRAM after load: %.2fGB used / %.2fGB total (%.1f%%)",
            used_vram / 1024**3, total_vram / 1024**3, 100.0 * used_vram / total_vram,
        )

    # --- Compute fingerprints ---
    logger.info(
        "Computing activation fingerprints for %d entities, task='%s', n_steps=%d, seeds=%s",
        len(entity_names), task, n_steps, seeds,
    )

    fingerprints = compute_act_fingerprints_for_task(
        task=task,
        pipeline=pipeline,
        entity_prompts=entity_prompts,
        device=device,
        n_steps=n_steps,
        seeds=seeds,
    )

    # --- Final VRAM check ---
    if device == "cuda" and torch.cuda.is_available():
        import psutil
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        used_vram = total_vram - free_vram
        ram = psutil.virtual_memory()
        logger.info(
            "Post-run resources: VRAM %.2fGB/%.2fGB (%.1f%%) | RAM %.1f%% (%.1fGB free)",
            used_vram / 1024**3, total_vram / 1024**3, 100.0 * used_vram / total_vram,
            ram.percent, ram.available / 1024**3,
        )

    # --- Save ---
    saved_path = save_act_fingerprints(
        fingerprints=fingerprints,
        task=task,
        model=MODEL_SHORT_NAME,
        base_folder=base_folder,
        n_steps=n_steps,
        seeds=seeds,
        model_name=MI_SD_MODEL_NAME,
    )

    logger.info("Done. Fingerprints saved to: %s", saved_path)
    logger.info("Fingerprint dimensionality: %d", next(iter(fingerprints.values())).shape[0])
    logger.info(
        "Next step: run ResultTemplateSimilarityMatrix(task='%s', similarity_metric='act').compute()",
        task,
    )
