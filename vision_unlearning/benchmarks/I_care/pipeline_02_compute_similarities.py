"""pipeline_02 — Compute pairwise entity similarity matrices.

Supports five similarity metrics (``--similarity`` flag):

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
    unet_latent   Cosine similarity of the final denoised latents (the 4x64x64 tensors
                  the VAE decoder consumes).  If the latent cache already exists the
                  matrix is computed from it (CPU).  If the cache is missing, this
                  script loads Stable Diffusion 1.4 and captures it first (requires
                  GPU): 100 entities x 4 seeds x 50 denoising steps per task.  The
                  capture reproduces the canonical image generation's randomness (one
                  generator per seed, advanced across the prompt list), so every
                  latent is decoded and checked against the baseline image the
                  benchmark already generated for that (entity, seed).
    all           Runs all five in order: clip → dino → jacc → act → unet_latent.

For ``clip``, ``dino``, and ``jacc`` (and the RT step for ``act`` and ``unet_latent``)
the work is delegated to ``ResultTemplateSimilarityMatrix(...).compute()`` which handles
the local-cache / HuggingFace / compute-from-scratch chain automatically.

``--n-steps`` applies to ``act`` only.  ``unet_latent`` uses 50 denoising steps, the
value the canonical image generation uses, and is not configurable here.

``--upload`` uploads a matrix that this run computed.  It is ``upload_if_recomputed``, so a
matrix already present locally is read rather than recomputed and nothing is uploaded; to
publish an existing matrix, move it aside first so the resolve has to recompute it (the
matrix is a deterministic function of its source data, so this is safe).  ``--validate-capture``
runs the ``unet_latent`` correctness gate instead of a bulk capture: determinism under
replay, agreement with the baseline images for the first three entities across all four
seeds, the cache round trip, agreement of the pipeline's own image output, sensitivity to
seed and prompt, and the measured budget.  About eighteen generations.

Usage
-----
    python pipeline_02_compute_similarities.py --task breeds --similarity dino
    python pipeline_02_compute_similarities.py --task people --similarity all
    python pipeline_02_compute_similarities.py --task scenes --similarity act --device cpu
    python pipeline_02_compute_similarities.py --task breeds --similarity unet_latent --validate-capture

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

def resolve_matrix(
    task: Literal["scenes", "breeds", "people"],
    similarity_metric: str,
    base_folder: str,
    upload: bool = False,
) -> None:
    """Resolve one similarity matrix through the RT (local -> HuggingFace -> compute).

    ``ResultTemplateSimilarityMatrix`` does not forward ``upload_if_recomputed`` to the
    ``Similarity`` artifact it reads, so an upload run resolves the artifact itself first —
    that is what puts a freshly computed ``similarity_{s}_{task}.json`` on HuggingFace.
    """
    import vision_unlearning.benchmarks.I_care as vb
    if upload:
        vb.Similarity(
            task=task,
            similarity_metric=similarity_metric,  # type: ignore[arg-type]
            base_folder=base_folder,
            upload_if_recomputed=True,
        ).compute()
    vb.ResultTemplateSimilarityMatrix(
        task=task,
        similarity_metric=similarity_metric,  # type: ignore[arg-type]
        base_folder=base_folder,
        upload_if_recomputed=upload,
    ).compute()


def run_clip(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
    upload: bool = False,
) -> None:
    """Download or fall through to the CLIP similarity matrix for a task.

    The RT downloads from HuggingFace when available; if the matrix is not
    present locally or on HF it raises ArtifactNotAvailableError with instructions
    for the GPU path.
    """
    import vision_unlearning.benchmarks.I_care as vb
    logger.info("CLIP similarity matrix for task=%s — calling RT.compute()...", task)
    try:
        resolve_matrix(task, "clip", base_folder, upload)
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
    upload: bool = False,
) -> None:
    """Compute or download the DINOv2 similarity matrix for a task (CPU-only)."""
    logger.info("DINOv2 similarity matrix for task=%s — calling RT.compute()...", task)
    resolve_matrix(task, "dino", base_folder, upload)
    logger.info("DINOv2 similarity matrix for task=%s OK.", task)


def run_jacc(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
    upload: bool = False,
) -> None:
    """Compute or download the Jaccard similarity matrix for a task (CPU-only)."""
    logger.info("Jaccard similarity matrix for task=%s — calling RT.compute()...", task)
    resolve_matrix(task, "jacc", base_folder, upload)
    logger.info("Jaccard similarity matrix for task=%s OK.", task)


def run_act(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
    device: str | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    seeds: List[int] | None = None,
    replace_if_exists: bool = False,
    upload: bool = False,
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

    logger.info("Activation similarity matrix for task=%s — calling RT.compute()...", task)
    resolve_matrix(task, "act", base_folder, upload)
    logger.info("Activation similarity matrix for task=%s OK.", task)


def run_unet_latent(
    task: Literal["scenes", "breeds", "people"],
    base_folder: str,
    device: str | None = None,
    upload: bool = False,
    validate_capture: bool = False,
    report_dir: str = "reports/similarity_unet_latent",
) -> None:
    """Compute or download the final-denoised-latent similarity matrix for a task.

    If the latent cache is already present (or the matrix is available on HF via the RT),
    no SD model is loaded.  Otherwise Stable Diffusion 1.4 is loaded and 100 x 4 latents
    are captured (GPU required), each one checked against its baseline image as it is
    captured.  With ``validate_capture`` the correctness gate runs instead: about eighteen
    generations, no bulk pass, images and measurements written to ``report_dir``.
    """
    import vision_unlearning.benchmarks.I_care as vb

    latents = vb.UnetLatentSimilarity(task=task, base_folder=base_folder)

    if validate_capture:
        import torch
        resolved_device: str = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Validating the unet_latent capture for task=%s on device=%s...", task, resolved_device,
        )
        measurements = latents.validate_capture(resolved_device, report_dir)
        for name, value in measurements.items():
            logger.info("  %s = %s", name, value)
        logger.info("unet_latent capture validated; measurements written to %s.", report_dir)
        return

    if not os.path.exists(latents.cache_path()):
        import torch
        resolved_device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Final denoised latents for task=%s not found at %s — capturing on device=%s "
            "(GPU required).", task, latents.cache_path(), resolved_device,
        )
        if resolved_device == "cuda" and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            used_vram = total_vram - free_vram
            logger.info(
                "VRAM before SD load: %.2fGB used / %.2fGB total (%.1f%% used, %.2fGB free)",
                used_vram / 1024**3, total_vram / 1024**3,
                100.0 * used_vram / total_vram, free_vram / 1024**3,
            )
        latents.capture(device=resolved_device)
        logger.info("Final denoised latents saved to %s.", latents.cache_path())
    else:
        logger.info(
            "Final denoised latents for task=%s found at %s — skipping capture.",
            task, latents.cache_path(),
        )

    logger.info("Final-latent similarity matrix for task=%s — calling RT.compute()...", task)
    resolve_matrix(task, "unet_latent", base_folder, upload)
    logger.info("Final-latent similarity matrix for task=%s OK.", task)


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
        choices=["clip", "dino", "jacc", "act", "unet_latent", "all"],
        default="all",
        help=(
            "Similarity metric to compute.  Use 'all' to run all five in order "
            "(clip → dino → jacc → act → unet_latent).  Default: all."
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
    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload a freshly computed similarity matrix to HuggingFace.",
    )
    parser.add_argument(
        "--validate-capture",
        action="store_true",
        default=False,
        help=(
            "Run the unet_latent capture correctness gate (about eighteen generations) instead of "
            "a bulk capture, writing its measurements and images to --report-dir."
        ),
    )
    parser.add_argument(
        "--report-dir",
        default="reports/similarity_unet_latent",
        help=(
            "Where --validate-capture writes its measurements and images "
            "(default: reports/similarity_unet_latent)."
        ),
    )
    args = parser.parse_args()

    task: Literal["scenes", "breeds", "people"] = args.task  # type: ignore[assignment]
    similarities: List[str] = (
        ["clip", "dino", "jacc", "act", "unet_latent"]
        if args.similarity == "all" else [args.similarity]
    )

    for similarity in similarities:
        logger.info("=== similarity=%s, task=%s ===", similarity, task)
        if similarity == "clip":
            run_clip(task, args.base_folder, upload=args.upload)
        elif similarity == "dino":
            run_dino(task, args.base_folder, upload=args.upload)
        elif similarity == "jacc":
            run_jacc(task, args.base_folder, upload=args.upload)
        elif similarity == "act":
            run_act(
                task=task,
                base_folder=args.base_folder,
                device=args.device,
                n_steps=args.n_steps,
                seeds=args.seeds,
                replace_if_exists=args.replace_if_exists,
                upload=args.upload,
            )
        elif similarity == "unet_latent":
            run_unet_latent(
                task=task,
                base_folder=args.base_folder,
                device=args.device,
                upload=args.upload,
                validate_capture=args.validate_capture,
                report_dir=args.report_dir,
            )

    logger.info("pipeline_02 done.")


if __name__ == "__main__":
    main()
