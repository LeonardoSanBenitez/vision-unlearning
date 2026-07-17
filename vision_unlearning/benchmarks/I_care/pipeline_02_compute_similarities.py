"""pipeline_02 — Compute pairwise entity similarity matrices.

Supports four similarity metrics (``--similarity`` flag):

    clip          CLIP-space cosine similarity.  The RT downloads the matrix from
                  HuggingFace when available; a from-scratch recompute requires a
                  GPU with a CLIP/SD model and is intentionally not wired here.
    dino          DINOv2-ViT-S/14 cosine similarity computed from the method-agnostic
                  baseline embedding file (``embeddings_{task}_original.json``).
                  CPU-only.
    jacc          Jaccard overlap of visual attributes from the task metadata.
                  CPU-only; writes a partial checkpoint after each row.
    act           Cosine similarity of UNet cross-attention activation fingerprints.
                  If the fingerprint file already exists the RT computes the matrix
                  from it (CPU).  If the file is missing, this script loads Stable
                  Diffusion 1.4 and generates it first (requires GPU).
    all           Runs all four in order: clip → dino → jacc → act.

For ``clip``, ``dino``, and ``jacc`` (and the RT step for ``act``) the work is
delegated to ``ResultTemplateSimilarityMatrix(...).compute()`` which handles the
local-cache / HuggingFace / compute-from-scratch chain automatically.

Usage
-----
    python pipeline_02_compute_similarities.py --task breeds --similarity dino
    python pipeline_02_compute_similarities.py --task people --similarity all
    python pipeline_02_compute_similarities.py --task scenes --similarity act --device cpu

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Literal

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("pipeline_02")

SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
MODEL_SHORT_NAME = "sd1.4"
DEFAULT_N_STEPS = 10
DEFAULT_SEEDS = [42, 43, 44, 45]


# ---------------------------------------------------------------------------
# Similarity runners
# ---------------------------------------------------------------------------

def run_clip(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
) -> None:
    """Download or fall through to the CLIP similarity matrix for a task.

    The RT downloads from HuggingFace when available; if the matrix is not
    present locally or on HF it raises ArtifactNotAvailableError with instructions
    for the GPU path.
    """
    import vision_unlearning.benchmarks.I_care as vb
    rt = vb.ResultTemplateSimilarityMatrix(
        task=task,
        similarity_metric="clip",
        base_folder=base_folder,
    )
    logger.info("CLIP similarity matrix for task=%s — calling RT.compute()...", task)
    try:
        rt.compute()
        logger.info("CLIP similarity matrix for task=%s OK.", task)
    except vb.ArtifactNotAvailableError as exc:
        logger.error(
            "CLIP similarity matrix for task=%s: %s\n"
            "A from-scratch CLIP recompute requires a GPU with CLIP/SD 1.4 loaded, "
            "which is not supported in this pipeline.  Either:\n"
            "  1. Run the original calculate_similarity_clip on the GPU machine, or\n"
            "  2. Make sure the matrix is uploaded to HuggingFace so the RT can download it.",
            task, exc,
        )


def run_dino(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
) -> None:
    """Compute or download the DINOv2 similarity matrix for a task (CPU-only)."""
    import vision_unlearning.benchmarks.I_care as vb
    rt = vb.ResultTemplateSimilarityMatrix(
        task=task,
        similarity_metric="dino",
        base_folder=base_folder,
    )
    logger.info("DINOv2 similarity matrix for task=%s — calling RT.compute()...", task)
    rt.compute()
    logger.info("DINOv2 similarity matrix for task=%s OK.", task)


def run_jacc(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
) -> None:
    """Compute or download the Jaccard similarity matrix for a task (CPU-only)."""
    import vision_unlearning.benchmarks.I_care as vb
    rt = vb.ResultTemplateSimilarityMatrix(
        task=task,
        similarity_metric="jacc",
        base_folder=base_folder,
    )
    logger.info("Jaccard similarity matrix for task=%s — calling RT.compute()...", task)
    rt.compute()
    logger.info("Jaccard similarity matrix for task=%s OK.", task)


def run_act(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
    device: str | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    seeds: List[int] | None = None,
    replace_if_exists: bool = False,
) -> None:
    """Compute or download the activation-fingerprint similarity matrix for a task.

    If the fingerprint file is already present (or available on HF via the RT),
    no SD model is loaded.  Otherwise Stable Diffusion 1.4 is loaded (GPU required).
    """
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    from vision_unlearning.utils.mechanistic_interpretability import (
        SD_MODEL_NAME as MI_SD_MODEL_NAME,
        compute_act_fingerprints_for_task,
        get_act_fingerprints_path,
        save_act_fingerprints,
    )
    from vision_unlearning.datasets.testbed import (
        get_metadata_filtered,
        get_target_overwrite,
    )
    from typing import Any, Dict

    fingerprint_path = get_act_fingerprints_path(task, MODEL_SHORT_NAME, base_folder)

    if not os.path.exists(fingerprint_path) or replace_if_exists:
        logger.info(
            "Activation fingerprints for task=%s not found at %s — computing (GPU required).",
            task, fingerprint_path,
        )
        import torch
        from diffusers import StableDiffusionPipeline

        resolved_device: str = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", resolved_device)
        if resolved_device == "cuda" and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            used_vram = total_vram - free_vram
            logger.info(
                "VRAM before SD load: %.2fGB used / %.2fGB total (%.1f%% used, %.2fGB free)",
                used_vram / 1024**3, total_vram / 1024**3,
                100.0 * used_vram / total_vram, free_vram / 1024**3,
            )

        logger.info("Loading Stable Diffusion 1.4 (%s)...", SD_MODEL_NAME)
        pipeline: Any = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME,
            torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(resolved_device)
        pipeline.set_progress_bar_config(disable=True)

        metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(
            task, base_folder=base_folder
        )
        entity_names = [e["name"] for e in metadata_filtered]
        entity_prompts: Dict[str, str] = {
            name: f"An image of {get_target_overwrite(task, 'distil', name)[0]}"
            for name in entity_names
        }

        fingerprints = compute_act_fingerprints_for_task(
            task=task,
            pipeline=pipeline,
            entity_prompts=entity_prompts,
            device=resolved_device,
            n_steps=n_steps,
            seeds=seeds,
        )
        save_act_fingerprints(
            fingerprints=fingerprints,
            task=task,
            model=MODEL_SHORT_NAME,
            base_folder=base_folder,
            n_steps=n_steps,
            seeds=seeds,
            model_name=MI_SD_MODEL_NAME,
        )
        logger.info("Activation fingerprints saved to %s.", fingerprint_path)
    else:
        logger.info(
            "Activation fingerprints for task=%s found at %s — skipping computation.",
            task, fingerprint_path,
        )

    import vision_unlearning.benchmarks.I_care as vb
    rt = vb.ResultTemplateSimilarityMatrix(
        task=task,
        similarity_metric="act",
        base_folder=base_folder,
    )
    logger.info("Activation similarity matrix for task=%s — calling RT.compute()...", task)
    rt.compute()
    logger.info("Activation similarity matrix for task=%s OK.", task)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise entity similarity matrices for a given task and similarity metric. "
            "Delegates to ResultTemplateSimilarityMatrix.compute() for all metrics except 'act' "
            "(which may require generating activation fingerprints on a GPU first)."
        )
    )
    parser.add_argument(
        "--task",
        choices=["scenes", "breeds", "people"],
        required=True,
        help="Task name.",
    )
    parser.add_argument(
        "--similarity",
        choices=["clip", "dino", "jacc", "act", "all"],
        default="all",
        help=(
            "Similarity metric to compute.  Use 'all' to run all four in order "
            "(clip → dino → jacc → act).  Default: all."
        ),
    )
    parser.add_argument(
        "--base-folder",
        default="assets",
        help="Local base folder for data storage (default: assets).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device for activation fingerprint computation "
            "(e.g. 'cuda', 'cpu'). Defaults to CUDA if available."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Denoising steps for act fingerprint generation (default: {DEFAULT_N_STEPS}).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Seeds for act fingerprint generation (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--replace-if-exists",
        action="store_true",
        default=False,
        help="Recompute fingerprints even if the file already exists (act only).",
    )
    args = parser.parse_args()

    task: Literal["scenes", "breeds", "people"] = args.task  # type: ignore[assignment]
    similarities: List[str] = (
        ["clip", "dino", "jacc", "act"] if args.similarity == "all" else [args.similarity]
    )

    for similarity in similarities:
        logger.info("=== similarity=%s, task=%s ===", similarity, task)
        if similarity == "clip":
            run_clip(task, args.base_folder)
        elif similarity == "dino":
            run_dino(task, args.base_folder)
        elif similarity == "jacc":
            run_jacc(task, args.base_folder)
        elif similarity == "act":
            run_act(
                task=task,
                base_folder=args.base_folder,
                device=args.device,
                n_steps=args.n_steps,
                seeds=args.seeds,
                replace_if_exists=args.replace_if_exists,
            )

    logger.info("pipeline_02 done.")


if __name__ == "__main__":
    main()
