"""Tests for vision_unlearning.utils.data_generation.generate_dataset.

All tests are CPU-only and GPU-free.  The pipeline and image-saving are stubbed
so no model weights are needed.

Coverage:
- Legacy mode (no seeds): filenames, metadata structure, no-seeding path.
- Seeded mode: determinism guarantee (same seed → same generator calls);
  different seeds → different images; auto-generated filenames match convention.
- Parameter validation: seeds + filenames mutually exclusive; seeds without
  lora_state raises.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: ensure vision_unlearning is importable from the sibling repo
# ---------------------------------------------------------------------------
_VU_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _VU_PATH not in sys.path:
    sys.path.insert(0, _VU_PATH)


# ---------------------------------------------------------------------------
# Stub out heavy dependencies before importing the module under test.
# The real imports (diffusers, unlearn_lora) would require GPU / model weights.
# ---------------------------------------------------------------------------

def _make_stub_image(pixel_value: int = 128) -> Any:
    """Return a fake PIL image object that supports .save()."""
    img = MagicMock()
    # Store a pixel value so we can distinguish images generated with different seeds.
    img._pixel_value = pixel_value
    return img


def _make_pipeline_stub(pixel_value: int = 128) -> MagicMock:
    """Return a callable pipeline stub that returns fake images."""
    def _call(prompts: List[str], **kwargs: Any) -> Any:
        result = MagicMock()
        result.images = [_make_stub_image(pixel_value) for _ in prompts]
        return result

    stub = MagicMock(side_effect=_call)
    # Make it behave like a pipeline (supports .to())
    stub.to = MagicMock(return_value=stub)
    return stub


# Patch the heavy imports so we can import data_generation without GPU
_diffusers_stub = types.ModuleType("diffusers")
_diffusers_stub.AutoPipelineForText2Image = MagicMock()  # type: ignore[attr-defined]
_vu_stub = types.ModuleType("vision_unlearning")
_vu_datasets_stub = types.ModuleType("vision_unlearning.datasets")
_vu_datasets_others_stub = types.ModuleType("vision_unlearning.datasets.others")
_vu_datasets_others_stub.jsonl_dump = MagicMock()  # type: ignore[attr-defined]
_vu_unlearner_stub = types.ModuleType("vision_unlearning.unlearner")
_vu_unlearner_lora_stub = types.ModuleType("vision_unlearning.unlearner.lora")
_vu_unlearner_lora_stub.unlearn_lora = MagicMock()  # type: ignore[attr-defined]

_patches: dict = {}  # populated in setUpModule


def setUpModule() -> None:
    """Install stubs before any test in this module runs."""
    _patches["diffusers"] = patch.dict(
        "sys.modules",
        {
            "diffusers": _diffusers_stub,
            "vision_unlearning": _vu_stub,
            "vision_unlearning.datasets": _vu_datasets_stub,
            "vision_unlearning.datasets.others": _vu_datasets_others_stub,
            "vision_unlearning.unlearner": _vu_unlearner_stub,
            "vision_unlearning.unlearner.lora": _vu_unlearner_lora_stub,
        },
    )
    _patches["diffusers"].start()


def tearDownModule() -> None:
    _patches["diffusers"].stop()


# Import the module under test AFTER patching
from vision_unlearning.utils import data_generation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_seeded_generation(
    seeds: List[int],
    prompts: List[str],
    lora_state: str,
    pipeline_stub: MagicMock,
    output_path: str,
    batch_size: int = 10,
) -> List[Dict[str, str]]:
    """Convenience wrapper for the seeded path."""
    with patch.object(
        data_generation,
        "AutoPipelineForText2Image",
        pipeline_stub,
    ):
        # Patch 'to' so pipeline.to(device) returns itself
        pipeline_stub.from_pretrained = MagicMock(return_value=pipeline_stub)
        return data_generation.generate_dataset(
            model_base_name="stub-model",
            lora_name=None,
            prompts=prompts,
            output_path=output_path,
            seeds=seeds,
            lora_state=lora_state,  # type: ignore[arg-type]
            batch_size=batch_size,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParameterValidation(unittest.TestCase):
    """Validate that bad parameter combinations raise early."""

    def test_seeds_and_filenames_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                data_generation.generate_dataset(
                    model_base_name="dummy",
                    lora_name=None,
                    prompts=["prompt A"],
                    output_path=tmp,
                    filenames=["file.png"],
                    seeds=[42],
                    lora_state="off",
                )
            self.assertIn("mutually exclusive", str(ctx.exception))

    def test_seeds_without_lora_state_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                data_generation.generate_dataset(
                    model_base_name="dummy",
                    lora_name=None,
                    prompts=["prompt A"],
                    output_path=tmp,
                    seeds=[42],
                    lora_state=None,
                )
            self.assertIn("lora_state", str(ctx.exception))

    def test_no_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                data_generation.generate_dataset(
                    model_base_name=None,
                    lora_name=None,
                    prompts=["prompt A"],
                    output_path=tmp,
                )


def _make_torch_stub() -> MagicMock:
    """Return a minimal torch mock that satisfies seeded generation without CUDA."""
    mock_torch = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.cuda.manual_seed_all = MagicMock()
    mock_torch.manual_seed = MagicMock()
    # Generator(device).manual_seed(seed) → return something pipeline accepts
    mock_generator_instance = MagicMock()
    mock_generator_instance.manual_seed = MagicMock(return_value=mock_generator_instance)
    mock_torch.Generator = MagicMock(return_value=mock_generator_instance)
    return mock_torch


class TestSeededGenerationFilenames(unittest.TestCase):
    """Seeded mode produces correctly-named files matching testbed convention."""

    def test_filenames_match_convention(self) -> None:
        """Files must be named {lora_state}_{seed:02d}_{prompt}.png."""
        prompts = ["An image of Alice", "An image of Bob"]
        seeds = [42, 43]
        lora_state = "off"

        pipeline_stub = _make_pipeline_stub()
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch", _make_torch_stub()):
                    meta = data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=prompts,
                        output_path=tmp,
                        seeds=seeds,
                        lora_state=lora_state,  # type: ignore[arg-type]
                        batch_size=10,
                    )

        # Check metadata records
        file_names_in_meta = [r["file_name"] for r in meta]
        expected = [
            f"{lora_state}_{s:02d}_{p}.png"
            for s in seeds
            for p in prompts
        ]
        self.assertEqual(sorted(file_names_in_meta), sorted(expected))

    def test_metadata_text_matches_prompt(self) -> None:
        """Each metadata record's 'text' field must equal the original prompt."""
        prompts = ["An image of Colin Powell", "An image of George W. Bush"]
        seeds = [7]
        lora_state = "on"

        pipeline_stub = _make_pipeline_stub()
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch", _make_torch_stub()):
                    meta = data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=prompts,
                        output_path=tmp,
                        seeds=seeds,
                        lora_state=lora_state,  # type: ignore[arg-type]
                    )

        texts_in_meta = {r["text"] for r in meta}
        self.assertEqual(texts_in_meta, set(prompts))


class TestSeededDeterminism(unittest.TestCase):
    """
    Determinism guarantee: same seed → same generator call sequence.

    Because the pipeline is stubbed, we verify determinism at the
    torch.Generator level: the same seed must produce the same generator
    object state for each pipeline call.

    We also verify that different seeds produce different generator states.
    """

    def _capture_generator_seeds(
        self,
        seeds: List[int],
        prompts: List[str],
        lora_state: str = "off",
    ) -> List[int]:
        """
        Run generate_dataset in seeded mode and capture the manual_seed values
        passed to torch.Generator for each pipeline() call.
        """
        captured_seeds: List[int] = []

        class _TrackedGenerator:
            def __init__(self, device: Any) -> None:
                self._device = device

            def manual_seed(self, seed: int) -> "_TrackedGenerator":
                captured_seeds.append(seed)
                return self

        def _fake_pipeline(prompts_batch: List[str], generator: Any = None, **kw: Any) -> Any:
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch") as mock_torch:
                    mock_torch.cuda = MagicMock()
                    mock_torch.cuda.is_available = MagicMock(return_value=False)
                    mock_torch.manual_seed = MagicMock()
                    mock_torch.cuda.manual_seed_all = MagicMock()
                    mock_torch.float16 = MagicMock()
                    mock_torch.Generator = MagicMock(return_value=MagicMock(
                        manual_seed=MagicMock(return_value=MagicMock())
                    ))
                    # Use real torch.Generator tracking via captured_seeds list
                    mock_torch.Generator.side_effect = _TrackedGenerator
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=prompts,
                        output_path=tmp,
                        seeds=seeds,
                        lora_state=lora_state,  # type: ignore[arg-type]
                    )
        return captured_seeds

    def test_same_seeds_produce_same_generator_call_sequence(self) -> None:
        """Two runs with the same seeds list must call Generator.manual_seed
        in the same order with the same values."""
        seeds = [42, 43]
        prompts = ["An image of Alice", "An image of Bob"]
        seq1 = self._capture_generator_seeds(seeds, prompts)
        seq2 = self._capture_generator_seeds(seeds, prompts)
        self.assertEqual(seq1, seq2)
        # Must have one Generator per seed
        self.assertEqual(seq1, seeds)

    def test_different_seeds_produce_different_generator_call_sequence(self) -> None:
        """Different seeds must produce different Generator.manual_seed sequences."""
        prompts = ["An image of Alice"]
        seq_a = self._capture_generator_seeds([10], prompts)
        seq_b = self._capture_generator_seeds([99], prompts)
        self.assertNotEqual(seq_a, seq_b)

    def test_generator_is_passed_to_pipeline(self) -> None:
        """The seeded generator must be passed to the pipeline call."""
        generator_passed: List[Any] = []

        def _fake_pipeline(prompts_batch: List[str], generator: Any = None, **kw: Any) -> Any:
            generator_passed.append(generator)
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        class _FakeGenerator:
            def __init__(self, device: Any) -> None:
                self._device = device
                self._seed: Optional[int] = None

            def manual_seed(self, seed: int) -> "_FakeGenerator":
                self._seed = seed
                return self

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        seeds = [7, 8]
        prompts = ["prompt A", "prompt B"]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch") as mock_torch:
                    mock_torch.cuda = MagicMock()
                    mock_torch.cuda.is_available = MagicMock(return_value=False)
                    mock_torch.manual_seed = MagicMock()
                    mock_torch.cuda.manual_seed_all = MagicMock()
                    mock_torch.float16 = MagicMock()
                    mock_torch.Generator = MagicMock(side_effect=_FakeGenerator)
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=prompts,
                        output_path=tmp,
                        seeds=seeds,
                        lora_state="off",  # type: ignore[arg-type]
                    )

        # One pipeline call per seed (since batch_size >= len(prompts))
        self.assertEqual(len(generator_passed), 2)
        # Each generator should have been seeded with the correct seed
        for gen, expected_seed in zip(generator_passed, seeds):
            self.assertIsInstance(gen, _FakeGenerator)
            self.assertEqual(gen._seed, expected_seed)


class TestLegacyMode(unittest.TestCase):
    """Legacy mode (no seeds): filenames and metadata behavior unchanged."""

    def test_legacy_with_explicit_filenames(self) -> None:
        """Legacy mode uses provided filenames verbatim."""
        prompts = ["prompt A", "prompt B"]
        filenames = ["off_42_prompt A.png", "off_42_prompt B.png"]

        def _fake_pipeline(prompts_batch: List[str], **kw: Any) -> Any:
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                meta = data_generation.generate_dataset(
                    model_base_name="stub-model",
                    lora_name=None,
                    prompts=prompts,
                    output_path=tmp,
                    filenames=filenames,
                )

        self.assertEqual([r["file_name"] for r in meta], filenames)
        self.assertEqual([r["text"] for r in meta], prompts)

    def test_legacy_without_filenames_uses_index(self) -> None:
        """Legacy mode without filenames uses index-based names."""
        prompts = ["prompt A", "prompt B", "prompt C"]

        def _fake_pipeline(prompts_batch: List[str], **kw: Any) -> Any:
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                meta = data_generation.generate_dataset(
                    model_base_name="stub-model",
                    lora_name=None,
                    prompts=prompts,
                    output_path=tmp,
                )

        self.assertEqual([r["file_name"] for r in meta], ["0.png", "1.png", "2.png"])


class TestGlobalRNGSeeding(unittest.TestCase):
    """Verify that torch/numpy/random are seeded at the start of each seed iteration."""

    def test_torch_manual_seed_called_per_seed(self) -> None:
        seeds = [1, 2, 3]
        prompts = ["prompt A"]

        def _fake_pipeline(prompts_batch: List[str], generator: Any = None, **kw: Any) -> Any:
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        torch_manual_seed_calls: List[int] = []

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch") as mock_torch:
                    mock_torch.float16 = MagicMock()
                    mock_torch.cuda = MagicMock()
                    mock_torch.cuda.is_available = MagicMock(return_value=False)
                    mock_torch.cuda.manual_seed_all = MagicMock()
                    mock_torch.manual_seed = MagicMock(
                        side_effect=lambda s: torch_manual_seed_calls.append(s)
                    )
                    mock_torch.Generator = MagicMock(
                        return_value=MagicMock(
                            manual_seed=MagicMock(return_value=MagicMock())
                        )
                    )
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=prompts,
                        output_path=tmp,
                        seeds=seeds,
                        lora_state="off",  # type: ignore[arg-type]
                    )

        self.assertEqual(torch_manual_seed_calls, seeds)


if __name__ == "__main__":
    unittest.main()
