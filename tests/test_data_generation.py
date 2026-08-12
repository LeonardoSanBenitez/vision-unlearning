"""Tests for vision_unlearning.utils.data_generation.generate_dataset.

All tests are CPU-only and GPU-free.  The pipeline and image-saving are stubbed
so no model weights are needed.

Coverage:
- Legacy mode (no seeds): filenames, metadata structure, no-seeding path.
- Seeded mode: determinism guarantee (same seed → same generator calls);
  different seeds → different images; auto-generated filenames match convention.
- Seeded mode with caller-supplied filenames: seeds and filenames may coexist.
- Parameter validation: seeds + filenames length mismatch raises; no model raises.
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
    pipeline_stub: MagicMock,
    output_path: str,
    filenames: Optional[List[str]] = None,
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
            filenames=filenames,
            batch_size=batch_size,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParameterValidation(unittest.TestCase):
    """Validate that bad parameter combinations raise early."""

    def test_seeds_and_filenames_wrong_count_raises(self) -> None:
        """seeds + filenames is allowed, but count must be len(seeds)*len(prompts)."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                data_generation.generate_dataset(
                    model_base_name="dummy",
                    lora_name=None,
                    prompts=["prompt A", "prompt B"],
                    output_path=tmp,
                    filenames=["only_one.png"],  # wrong: need 1 seed * 2 prompts = 2
                    seeds=[42],
                )
            self.assertIn("seed-major order", str(ctx.exception))

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
    """Seeded mode produces correctly-named files."""

    def test_auto_filenames_no_prefix(self) -> None:
        """When seeds provided and filenames=None, files are named {seed}_{prompt}.png."""
        prompts = ["An image of Alice", "An image of Bob"]
        seeds = [42, 43]

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
                        batch_size=10,
                    )

        file_names_in_meta = [r["file_name"] for r in meta]
        expected = [
            f"{s}_{p}.png"
            for s in seeds
            for p in prompts
        ]
        self.assertEqual(sorted(file_names_in_meta), sorted(expected))

    def test_caller_supplied_filenames_used_verbatim(self) -> None:
        """When seeds and filenames are both provided, filenames are used as-is."""
        prompts = ["An image of Alice", "An image of Bob"]
        seeds = [42, 43]
        filenames = [
            f'off_{seed}_{prompt}.png'
            for seed in seeds
            for prompt in prompts
        ]

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
                        filenames=filenames,
                        batch_size=10,
                    )

        file_names_in_meta = [r["file_name"] for r in meta]
        self.assertEqual(sorted(file_names_in_meta), sorted(filenames))

    def test_metadata_text_matches_prompt(self) -> None:
        """Each metadata record's 'text' field must equal the original prompt."""
        prompts = ["An image of Colin Powell", "An image of George W. Bush"]
        seeds = [7]

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


class TestResolutionPassthrough(unittest.TestCase):
    """height/width reach the pipeline when asked for, and are absent otherwise.

    The absence case is the load-bearing one: every existing Stable Diffusion 1.4 artifact was
    produced by a call that passed neither, so a default that silently started passing 512x512
    would change the arguments of runs that are compared against those artifacts.
    """

    def _capture_pipeline_kwargs(self, **generate_kwargs: Any) -> List[Dict[str, Any]]:
        captured: List[Dict[str, Any]] = []

        def _fake_pipeline(prompts_batch: List[str], **kw: Any) -> Any:
            captured.append(kw)
            result = MagicMock()
            result.images = [_make_stub_image() for _ in prompts_batch]
            return result

        pipeline_stub = MagicMock(side_effect=_fake_pipeline)
        pipeline_stub.to = MagicMock(return_value=pipeline_stub)
        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(return_value=pipeline_stub)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch", _make_torch_stub()):
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=["An image of Alice"],
                        output_path=tmp,
                        **generate_kwargs,
                    )
        return captured

    def test_absent_by_default_seeded(self) -> None:
        captured = self._capture_pipeline_kwargs(seeds=[42])
        self.assertEqual(len(captured), 1)
        self.assertNotIn("height", captured[0])
        self.assertNotIn("width", captured[0])

    def test_absent_by_default_legacy(self) -> None:
        captured = self._capture_pipeline_kwargs()
        self.assertEqual(len(captured), 1)
        self.assertNotIn("height", captured[0])
        self.assertNotIn("width", captured[0])

    def test_passed_through_when_given(self) -> None:
        captured = self._capture_pipeline_kwargs(seeds=[42], height=512, width=512)
        self.assertEqual(captured[0]["height"], 512)
        self.assertEqual(captured[0]["width"], 512)

    def test_one_without_the_other_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                data_generation.generate_dataset(
                    model_base_name="stub-model",
                    lora_name=None,
                    prompts=["An image of Alice"],
                    output_path=tmp,
                    seeds=[42],
                    height=512,
                )
            self.assertIn("together", str(ctx.exception))


class TestVariantPassthrough(unittest.TestCase):
    """variant reaches from_pretrained when asked for, and is absent otherwise.

    Absent is again the load-bearing case: every existing artifact was produced by a call that read
    the full-precision weights and cast them, and a default of "fp16" would silently change which
    weight files a Stable Diffusion 1.4 run reads.
    """

    def _capture_from_pretrained_kwargs(self, **generate_kwargs: Any) -> Dict[str, Any]:
        captured: Dict[str, Any] = {}

        def _fake_from_pretrained(model: str, **kw: Any) -> Any:
            captured.update(kw)
            pipeline_stub = _make_pipeline_stub()
            return pipeline_stub

        _apit = MagicMock()
        _apit.from_pretrained = MagicMock(side_effect=_fake_from_pretrained)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "AutoPipelineForText2Image", _apit):
                with patch.object(data_generation, "torch", _make_torch_stub()):
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name=None,
                        prompts=["An image of Alice"],
                        output_path=tmp,
                        seeds=[42],
                        **generate_kwargs,
                    )
        return captured

    def test_absent_by_default(self) -> None:
        captured = self._capture_from_pretrained_kwargs()
        self.assertNotIn("variant", captured)

    def test_passed_through_when_given(self) -> None:
        captured = self._capture_from_pretrained_kwargs(variant="fp16")
        self.assertEqual(captured["variant"], "fp16")

    def test_forwarded_to_unlearn_lora(self) -> None:
        """The adapter path builds its pipeline inside unlearn_lora, so variant must reach it too."""
        captured: Dict[str, Any] = {}

        def _fake_unlearn_lora(*args: Any, **kw: Any) -> Any:
            captured.update(kw)
            return None, None, _make_pipeline_stub()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data_generation, "unlearn_lora", MagicMock(side_effect=_fake_unlearn_lora)):
                with patch.object(data_generation, "torch", _make_torch_stub()):
                    data_generation.generate_dataset(
                        model_base_name="stub-model",
                        lora_name="stub-adapter",
                        prompts=["An image of Alice"],
                        output_path=tmp,
                        seeds=[42],
                        variant="fp16",
                    )
        self.assertEqual(captured["variant"], "fp16")


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
                    )

        self.assertEqual(torch_manual_seed_calls, seeds)


if __name__ == "__main__":
    unittest.main()
