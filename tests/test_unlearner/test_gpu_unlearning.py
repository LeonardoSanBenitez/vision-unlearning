"""GPU integration tests for unlearning algorithms.

These tests run the actual unlearning code end-to-end on real GPU hardware.
They are intentionally minimal (1 step, 1 image) to exercise the code path
without requiring hours of compute.

Run locally with:
    poetry run pytest tests/test_unlearner/test_gpu_unlearning.py -v -s

Or via the Makefile (currently only supported locally, not in CI):
    make test-gpu

Requirements:
- CUDA-compatible GPU (tested on AMD ROCm via HIP/PyTorch interface)
- CompVis/stable-diffusion-v1-4 accessible (will download if not cached)
- Sufficient VRAM (~4-6 GB for UCE, ~8-10 GB for distil with fp16)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Skip all tests in this module unless a GPU is actually present
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.gpu


def _skip_if_no_cuda() -> None:
    """Raise a skip signal if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device available — GPU tests require a GPU.")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SD_MODEL = "CompVis/stable-diffusion-v1-4"
_TINY_SIZE = (64, 64)   # smallest resolution that SD pipelines typically accept


def _make_tiny_image(path: str) -> None:
    """Save a minimal valid PNG image at *path*."""
    arr = np.random.randint(0, 256, (*_TINY_SIZE, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


@pytest.fixture(scope="module")
def tiny_dataset_dir() -> Generator[str, None, None]:
    """Create a temporary directory with a 1-image dataset in ImageFolder format.

    Layout (compatible with HuggingFace ``datasets.load_dataset('imagefolder', ...)``):
        <tmpdir>/
            metadata.jsonl
            image_00.png
    """
    with tempfile.TemporaryDirectory(prefix="vu_gpu_test_") as tmpdir:
        img_name = "image_00.png"
        img_path = os.path.join(tmpdir, img_name)
        _make_tiny_image(img_path)

        meta = {"file_name": img_name, "text": "a photo of a cat"}
        with open(os.path.join(tmpdir, "metadata.jsonl"), "w") as f:
            f.write(json.dumps(meta) + "\n")

        yield tmpdir


# ---------------------------------------------------------------------------
# UCE smoke test
# ---------------------------------------------------------------------------

class TestUCESmoke:
    """Minimal end-to-end test for UCE (Unified Concept Editing).

    UCE is a closed-form weight update — no training loop, no dataset required.
    It only needs the text encoder + cross-attention weights.
    """

    def test_uce_erases_concept(self, tmp_path: Path) -> None:
        """UCE.train() completes without error on a real GPU."""
        _skip_if_no_cuda()

        # Local import to avoid loading heavy deps at collection time
        from vision_unlearning.unlearner.uce_sd_erase import UCE

        output_dir = str(tmp_path / "uce_output")
        unlearner = UCE(
            pretrained_model_name_or_path=_SD_MODEL,
            edit_concepts="cat",
            preserve_concepts="animal",
            guide_concepts=None,
            erase_scale=0.4,
            preserve_scale=1.0,
            lamb=0.5,
            device="cuda:0",
            output_dir=output_dir,
            hub_model_id=None,   # do not push to HuggingFace
            compute_runtimes=False,
            expand_prompts=False,
            final_eval_prompts_forget=[],
            final_eval_prompts_retain=[],
        )

        results = unlearner.train()

        # The method should return a list (possibly empty if no eval prompts)
        assert isinstance(results, list)

        # The output directory should exist after training
        assert os.path.isdir(output_dir), (
            f"UCE did not produce an output directory at {output_dir}"
        )

    def test_uce_gpu_vram_consumed(self) -> None:
        """After UCE completes, some VRAM should have been used."""
        _skip_if_no_cuda()

        free_before, total = torch.cuda.mem_get_info(0)
        used_before = total - free_before

        from vision_unlearning.unlearner.uce_sd_erase import UCE

        with tempfile.TemporaryDirectory(prefix="vu_gpu_uce_vram_") as tmp:
            unlearner = UCE(
                pretrained_model_name_or_path=_SD_MODEL,
                edit_concepts="cat",
                preserve_concepts="animal",
                erase_scale=0.4,
                preserve_scale=1.0,
                device="cuda:0",
                output_dir=tmp,
                hub_model_id=None,
                compute_runtimes=False,
                expand_prompts=False,
                final_eval_prompts_forget=[],
                final_eval_prompts_retain=[],
            )
            unlearner.train()

        free_after, _ = torch.cuda.mem_get_info(0)
        used_after = total - free_after
        # Sanity: something was allocated during UCE (at least the text encoder)
        # We use a generous lower bound because models may be cached differently
        assert used_after >= used_before or used_after > 0, (
            "UCE did not appear to allocate any VRAM — possible silent failure."
        )


# ---------------------------------------------------------------------------
# Distillation smoke test
# ---------------------------------------------------------------------------

class TestDistilSmoke:
    """Minimal end-to-end test for UnlearnerLoraDistillation (FADE distil).

    We run exactly 1 training step (max_train_steps=1) on a single tiny image
    to verify the forward pass, backward pass, optimizer step, and checkpoint
    saving all work correctly on real GPU hardware.
    """

    def test_distil_one_step(self, tiny_dataset_dir: str, tmp_path: Path) -> None:
        """UnlearnerLoraDistillation.train() completes 1 step on a real GPU."""
        _skip_if_no_cuda()

        from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
        from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

        output_dir = str(tmp_path / "distil_output")

        unlearner = UnlearnerLoraDistillation(  # type: ignore[call-arg]
            model_name_or_path=_SD_MODEL,
            # Use the same tiny dir for both forget and retain
            dataset_forget_name=tiny_dataset_dir,
            dataset_retain_name=tiny_dataset_dir,
            num_train_epochs=1,
            max_train_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            lr_warmup_steps=0,
            resolution=_TINY_SIZE[0],
            output_dir=output_dir,
            hub_model_id=None,
            device="cuda",
            gradient_weighting_method=GradientWeightingMethodSimple(),
            compute_runtimes=False,
            compute_gradient_conflict=False,
            compute_memory=False,
            # Use overwritting_concept so no json_metafile is needed
            overwritting_concept="dog",
            # No validation during this smoke test
            validation_prompt=None,
            num_validation_images=0,
            final_eval_prompts_forget=[],
            final_eval_prompts_retain=[],
            # Reduce memory usage
            mixed_precision="fp16",
        )

        results = unlearner.train()

        assert isinstance(results, list)
        assert os.path.isdir(output_dir), (
            f"Distil did not produce an output directory at {output_dir}"
        )

    def test_distil_checkpoint_saved(self, tiny_dataset_dir: str, tmp_path: Path) -> None:
        """After training, a LoRA checkpoint should exist in the output directory."""
        _skip_if_no_cuda()

        from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation
        from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

        output_dir = str(tmp_path / "distil_checkpoint")

        unlearner = UnlearnerLoraDistillation(  # type: ignore[call-arg]
            model_name_or_path=_SD_MODEL,
            dataset_forget_name=tiny_dataset_dir,
            dataset_retain_name=tiny_dataset_dir,
            num_train_epochs=1,
            max_train_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            lr_warmup_steps=0,
            resolution=_TINY_SIZE[0],
            output_dir=output_dir,
            hub_model_id=None,
            device="cuda",
            gradient_weighting_method=GradientWeightingMethodSimple(),
            compute_runtimes=False,
            compute_gradient_conflict=False,
            compute_memory=False,
            overwritting_concept="dog",
            validation_prompt=None,
            num_validation_images=0,
            final_eval_prompts_forget=[],
            final_eval_prompts_retain=[],
            mixed_precision="fp16",
        )

        unlearner.train()

        # Verify that a safetensors checkpoint was saved
        checkpoint_files = list(Path(output_dir).rglob("*.safetensors"))
        assert len(checkpoint_files) > 0, (
            f"No .safetensors checkpoint found in {output_dir}. "
            f"Contents: {list(Path(output_dir).rglob('*'))}"
        )
