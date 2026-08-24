"""pipeline_04 — Unlearn a model and generate the evaluation dataset.

Two modes:

  Normal mode (default): for a given task/method/epochs, trains the unlearned
  model for each entity and generates images under both LoRA-ON and LoRA-OFF
  conditions.  Requires a GPU.

  Baseline mode (--baseline): generates the method-agnostic baseline dataset
  (original Stable Diffusion, no LoRA) into
  ``assets/datasets/generated_{task}_baseline/``.  This run is task-scoped
  (not per-method) and is CPU-optional (GPU makes it faster).  Uses
  ``GeneratedDataset.compute()`` which handles local-cache / HF / generate.

Usage
-----
    # Normal mode
    python pipeline_04_generate_dataset.py --task breeds --method distil --num-train-epochs 100
    python pipeline_04_generate_dataset.py --task people --method munba --num-train-epochs 200

    # Baseline mode (run once per task, method-agnostic)
    python pipeline_04_generate_dataset.py --task breeds --baseline
    python pipeline_04_generate_dataset.py --task people --baseline

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
Requires: CUDA GPU for normal mode; GPU recommended but not required for baseline mode.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Literal, Optional

import dotenv
import torch

from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unlearn a Stable Diffusion model for one task/method/epoch combination "
            "and generate the evaluation image dataset, or (with --baseline) generate "
            "the method-agnostic baseline dataset from the original SD."
        )
    )
    parser.add_argument(
        "--task",
        choices=["scenes", "breeds", "people"],
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=["uce", "munba", "distil"],
        default=None,
        help="Unlearning method.  Required unless --baseline is set.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Training epochs for the unlearning run.  Required unless --baseline is set.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help=(
            "Generate the method-agnostic baseline dataset (original SD, no LoRA). "
            "Ignores --method and --num-train-epochs."
        ),
    )
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--max-identities", type=int, default=100)
    parser.add_argument(
        "--replace-if-exists",
        action="store_true",
        default=False,
        help="Recompute even if outputs already exist.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45],
        help="Seeds for image generation (default: 42 43 44 45).",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Cap the number of prompts per entity (default: no cap).",
    )
    parser.add_argument(
        "--upload-if-recomputed",
        action="store_true",
        default=False,
        help="Upload generated baseline to HuggingFace after computation (baseline mode only).",
    )
    parser.add_argument(
        "--base-folder",
        default="assets",
        help="Local assets folder (default: assets).",
    )
    parser.add_argument(
        "--ledger-path",
        default="",
        metavar="PATH",
        help=(
            "Where to append the per-entity run ledger (debugging/traceability only, "
            "not a paper artifact). Default (empty): '{base_folder}/logs/pipeline_04_ledger.jsonl'."
        ),
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        default=False,
        help="Disable writing the run ledger for this invocation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Baseline mode
# ---------------------------------------------------------------------------

def run_baseline(
    task: Literal["scenes", "breeds", "people"],
    seeds: List[int],
    replace_if_exists: bool,
    upload_if_recomputed: bool,
    base_folder: str,
    ledger: Optional[RunLedger] = None,
) -> None:
    """Generate method-agnostic baseline images (original SD, no LoRA)."""
    from vision_unlearning.utils.logger import get_logger, setup_loggers
    from vision_unlearning.datasets.testbed import (
        GeneratedDataset,
        get_metadata_filtered,
        get_target_overwrite,
    )
    logger = get_logger("pipeline_04_baseline")
    setup_loggers(modules_info=["unlearning"])

    metadata_filtered = get_metadata_filtered(task, base_folder=base_folder)
    prompts: List[str] = [
        f"An image of {get_target_overwrite(task, 'distil', m['name'])[0]}"
        for m in metadata_filtered
    ]

    logger.info(
        "Baseline mode: task=%s | %d entities | %d seeds | %d images total",
        task, len(prompts), len(seeds), len(prompts) * len(seeds),
    )

    ds = GeneratedDataset(
        task=task,
        base_folder=base_folder,
        recompute_if_exists=replace_if_exists,
        upload_if_recomputed=upload_if_recomputed,
    )
    logger.info("Target folder: %s", ds.folder_path)

    import time
    t0 = time.time()
    result_folder = ds.compute(seeds=seeds, prompts=prompts, batch_size=1)
    elapsed = time.time() - t0
    n_images = len(seeds) * len(prompts)
    logger.info(
        "ALL DONE. folder=%s  elapsed=%.1f s  (avg %.2f s/image)",
        result_folder, elapsed, elapsed / n_images if n_images else 0.0,
    )
    if ledger is not None:
        ledger.record(f"GeneratedDatasetBaseline for {task}", status="ok")


# ---------------------------------------------------------------------------
# Normal mode
# ---------------------------------------------------------------------------

def run_normal(
    task: Literal["scenes", "breeds", "people"],
    method: Literal["uce", "munba", "distil"],
    num_train_epochs: int,
    index_start: int,
    max_identities: int,
    replace_if_exists: bool,
    seeds: List[int],
    limit_prompts: Optional[int],
    base_folder: str,
    ledger: Optional[RunLedger] = None,
) -> None:
    """Unlearn the model and generate the per-entity evaluation dataset."""
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa: F401 (side-effect: GUI backend not needed; import kept for compatibility)
    from diffusers import AutoPipelineForText2Image  # noqa: F401

    from vision_unlearning.utils.logger import get_logger, setup_loggers
    from vision_unlearning.datasets import UnlearnDatasetImagenette
    from vision_unlearning.utils.gradient_weighting import (
        GradientWeightingMethodSimple,
        GradientWeightingMethodMunba,
    )
    from vision_unlearning.benchmarks.I_care import check_eval_results
    from vision_unlearning.datasets.testbed import (
        get_target_overwrite,
        get_generated_dataset_folder,
        get_unlearned_model_folder,
        exists_unlearned_model,
        exists_unlearned_dataset,
        get_metadata_filtered,
    )
    from vision_unlearning.utils.data_generation import generate_dataset

    logger = get_logger("pipeline_04")
    setup_loggers(modules_info=["unlearning"])

    with open(
        os.path.join(base_folder, f"metadata_{task}_2_enriched_filtered.json"),
        "r",
        encoding="utf-8",
    ) as f:
        metadata_filtered = json.load(f)

    model_base_name = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for index in range(index_start, index_start + max_identities):
        target = metadata_filtered[index]["name"]
        output_dir = get_unlearned_model_folder(task, method, num_train_epochs, target)

        if task == "people":
            dataset_base_path = os.path.join(base_folder, "datasets/lfw_splits_filtered")
        elif task == "breeds":
            dataset_base_path = os.path.join(base_folder, "datasets/taras_breeds_splits_filtered")
        elif task == "scenes":
            dataset_base_path = os.path.join(base_folder, "datasets/SUN_splits_filtered")
        else:
            raise NotImplementedError(f"Unknown task: {task}")

        dataset_forget_name = f"{dataset_base_path}/{target}/train_forget"
        dataset_retain_name = f"{dataset_base_path}/{target}/train_retain"

        target_preprocessed, target_overwrite = get_target_overwrite(task, method, target)

        # Evaluation prompts for optional progress checks
        if task in ("people", "breeds"):
            validation_prompt = f"An image of {target_preprocessed}"
            example_prompts_forget = [
                f"An image of {target_preprocessed}",
                f"Photograph of {target_preprocessed.replace('_', ' ')}; high definition",
                f"An picture of {target_preprocessed} in the rain",
                f"An picture of {target_preprocessed} running",
            ]
            example_prompts_retain = [
                f"An image of {target_overwrite}",
                f"Photograph of {target_overwrite}; high definition",
                f"An picture of {target_overwrite} in the rain",
                f"An picture of {target_overwrite} running",
            ]
        elif task == "scenes":
            validation_prompt = f"An image of {target_preprocessed}"
            example_prompts_forget = [
                f"An image of {target_preprocessed}",
                f"Photograph of {target_preprocessed.replace('_', ' ')}; high definition",
                f"An picture of {target_preprocessed} full of people",
                f"An picture of {target_preprocessed} during the night",
            ]
            example_prompts_retain = [
                f"An image of {target_overwrite}",
                f"Photograph of {target_overwrite}; high definition",
                f"An picture of {target_overwrite} full of people",
                f"An picture of {target_overwrite} during the night",
            ]
        else:
            raise NotImplementedError(f"Unknown task: {task}")

        # GPU memory info
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_memory, total_memory = torch.cuda.mem_get_info()
        logger.info("Free Memory: %.2f GB", free_memory / 1e9)
        logger.info("Total Memory: %.2f GB", total_memory / 1e9)

        hyperparameters: Dict[str, Any] = {
            "output_dir": output_dir,
            "hub_model_id": None,
            "final_eval_prompts_forget": example_prompts_forget,
            "final_eval_prompts_retain": example_prompts_retain,
        }
        if method == "uce":
            hyperparameters.update({
                "pretrained_model_name_or_path": model_base_name,
                "erase_scale": 30,
                "preserve_scale": 0.01,
                "lamb": 0.01,
                "edit_concepts": target_preprocessed,
                "guide_concepts": task,
                "preserve_concepts": task,
                "expand_prompts": False,
                "device": device,
                "save_entire_model": False,
            })
        else:
            hyperparameters.update({
                "model_name_or_path": model_base_name,
                "dataset_forget_name": dataset_forget_name,
                "dataset_retain_name": dataset_retain_name,
                "validation_prompt": validation_prompt,
                "dataloader_num_workers": 2,
                "resolution": 512,
                "num_validation_images": 1,
                "mixed_precision": "no",
                "learning_rate": 6e-4,
                "max_grad_norm": 5.0,
                "num_train_epochs": num_train_epochs,
                "validation_epochs": num_train_epochs + 1,
                "checkpointing_steps": 10000,
                "lr_scheduler_type": "constant",
                "lr_warmup_steps": 0,
                "save_strategy": "epoch",
                "save_total_limit": 2,
                "random_flip": True,
                "lora_r": 16,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "lora_alpha": 4,
                "lora_dropout": 0.2,
                "seed": 42,
            })
            if free_memory > 20e9:
                hyperparameters.update({
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 1,
                })
            elif free_memory > 14e9:
                hyperparameters.update({
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                })
            else:
                hyperparameters.update({
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                })

        if free_memory > 20e9:
            batch_size_inference = 50
        elif free_memory > 14e9:
            batch_size_inference = 25
        else:
            batch_size_inference = 25

        if method == "distil":
            from unlearner_lora_distillation import UnlearnerLoraDistillation
            hyperparameters["overwritting_concept"] = target_overwrite
            hyperparameters["gradient_weighting_method"] = GradientWeightingMethodSimple(
                forget_weight=0.3, retain_weight=1.0
            )
            unlearner = UnlearnerLoraDistillation(**hyperparameters)
        elif method == "munba":
            from vision_unlearning.unlearner import UnlearnerLoraDirect
            hyperparameters["gradient_weighting_method"] = GradientWeightingMethodMunba()
            unlearner = UnlearnerLoraDirect(**hyperparameters)
        elif method == "uce":
            from vision_unlearning.unlearner import UCE, ConceptType  # noqa: F401
            unlearner = UCE(**hyperparameters)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

        if replace_if_exists or not exists_unlearned_model(task, method, num_train_epochs, target):
            logger.info("Overwriting the entity '%s' by '%s'", target, target_overwrite)
            logger.info("%s", hyperparameters)
            eval_results = unlearner.train()
            logger.info(
                "Evaluation metrics:\n%s",
                pd.DataFrame([
                    {"Name": r.metric_name, "Value": r.metric_value} for r in eval_results
                ]).to_markdown(),
            )

            check_eval_results(eval_results, "Runtime training seconds", 10 * 3600, "lt")
            check_eval_results(eval_results, "Runtime training seconds", 60, "gt")

            if task == "breeds" and method == "distil":
                check_eval_results(
                    eval_results,
                    "ForgetSet clip score difference between original and unlearned mean", 3, "gt",
                )
                check_eval_results(
                    eval_results,
                    "RetainSet clip score difference between original and unlearned mean", 0.5, "lt",
                )
            elif task == "people" and method == "munba":
                check_eval_results(
                    eval_results,
                    "ForgetSet clip score difference between original and unlearned mean", 1.5, "gt",
                )
                check_eval_results(
                    eval_results,
                    "RetainSet clip score difference between original and unlearned mean", 1, "lt",
                )
            elif task == "people" and method == "distil":
                check_eval_results(
                    eval_results,
                    "ForgetSet clip score difference between original and unlearned mean", 5, "gt",
                )
                check_eval_results(
                    eval_results,
                    "RetainSet clip score difference between original and unlearned mean", 1, "lt",
                )
            elif task == "scenes" and method == "distil":
                check_eval_results(
                    eval_results,
                    "ForgetSet clip score difference between original and unlearned mean", 5, "gt",
                )
                check_eval_results(
                    eval_results,
                    "RetainSet clip score difference between original and unlearned mean", 1, "lt",
                )

            del unlearner
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            free_memory, total_memory = torch.cuda.mem_get_info()
            logger.info(
                "After training — Free: %.2f GB | Total: %.2f GB",
                free_memory / 1e9, total_memory / 1e9,
            )
        else:
            logger.info(
                "Skipping unlearning for '%s' — already exists at '%s'", target, output_dir
            )

        # Generate dataset
        prompts_gen = [
            f"An image of {get_target_overwrite(task, method, m['name'])[0]}"
            for m in metadata_filtered
        ]
        if limit_prompts is not None:
            prompts_gen = prompts_gen[:limit_prompts]

        generated_dataset_output_path = get_generated_dataset_folder(
            task, method, num_train_epochs, target_preprocessed
        )

        if replace_if_exists or not exists_unlearned_dataset(
            generated_dataset_output_path, seeds, prompts_gen
        ):
            logger.info(
                "Generating images for '%s' → '%s'",
                target_preprocessed, generated_dataset_output_path,
            )
            for seed in seeds:
                for lora_state in ["on", "off"]:
                    filenames = [f"{lora_state}_{seed}_{p}.png" for p in prompts_gen]
                    if method == "uce":
                        from vision_unlearning.unlearner import UCE as _UCE
                        model_pipeline = _UCE.get_pipeline_from_modified_weights(
                            pretrained_model_name_or_path=model_base_name,
                            device=device,
                            output_dir=output_dir,
                        )
                        generate_dataset(
                            model_base_name=None,
                            lora_name=None,
                            model_pipeline=model_pipeline,  # type: ignore[arg-type]
                            prompts=prompts_gen,
                            output_path=generated_dataset_output_path,
                            filenames=filenames,
                            batch_size=batch_size_inference,
                            lora_requires_inversion=False,
                        )
                    else:
                        lora_name = output_dir if (lora_state == "on") else None
                        generate_dataset(
                            model_base_name=model_base_name,
                            lora_name=lora_name,
                            model_pipeline=None,
                            prompts=prompts_gen,
                            output_path=generated_dataset_output_path,
                            filenames=filenames,
                            batch_size=batch_size_inference,
                            lora_requires_inversion=(method == "munba"),
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        else:
            logger.info(
                "Skipping dataset generation for '%s' — already exists at '%s'",
                target_preprocessed, generated_dataset_output_path,
            )

        assert exists_unlearned_dataset(generated_dataset_output_path, seeds, prompts_gen)
        logger.info(
            "-" * 100 + "\nFinished: target='%s' (%d/%d)\n" + "-" * 100,
            target_preprocessed, index, index_start + max_identities - 1,
        )
        if ledger is not None:
            # A single "ok" per entity, not distinguishing unlearning-skipped from
            # dataset-generation-skipped -- the two stages above already log their own
            # skip/recompute decision; this is the entity-level completion marker (the
            # assert above is the actual, existing correctness check for it).
            ledger.record(
                f"GeneratedDataset for {task}/{method}/{num_train_epochs}/index={index}",
                status="ok",
            )

    logger.info("ALL DONE.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env for HF_TOKEN and WANDB settings
    for _ep in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".env"),
    ]:
        if os.path.exists(_ep):
            dotenv.load_dotenv(_ep)
            break
    os.environ.setdefault("WANDB_DISABLED", "true")

    args = _parse_args()
    task: Literal["scenes", "breeds", "people"] = args.task  # type: ignore[assignment]

    ledger: Optional[RunLedger] = None
    if not args.no_ledger:
        ledger_path = args.ledger_path or os.path.join(args.base_folder, "logs", "pipeline_04_ledger.jsonl")
        ledger = RunLedger(ledger_path)

    try:
        if args.baseline:
            run_baseline(
                task=task,
                seeds=args.seeds,
                replace_if_exists=args.replace_if_exists,
                upload_if_recomputed=args.upload_if_recomputed,
                base_folder=args.base_folder,
                ledger=ledger,
            )
        else:
            if args.method is None:
                raise SystemExit("--method is required in normal mode (or use --baseline).")
            if args.num_train_epochs is None:
                raise SystemExit("--num-train-epochs is required in normal mode (or use --baseline).")
            method: Literal["uce", "munba", "distil"] = args.method  # type: ignore[assignment]
            run_normal(
                task=task,
                method=method,
                num_train_epochs=args.num_train_epochs,
                index_start=args.index_start,
                max_identities=args.max_identities,
                replace_if_exists=args.replace_if_exists,
                seeds=args.seeds,
                limit_prompts=args.limit_prompts,
                base_folder=args.base_folder,
                ledger=ledger,
            )
    finally:
        if ledger is not None:
            print(f"Run ledger: {ledger.count} records written to {ledger.path}")
            ledger.close()


if __name__ == "__main__":
    main()
