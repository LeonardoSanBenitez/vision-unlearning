"""Library conformance, for every unlearner this package ships.

Heavy tier: torch only, and deliberately **no checkpoint download, no network and no graphics
card**. Nothing here knows that a benchmark exists; a method that is useful to somebody outside this
repository has to satisfy exactly what is asserted below and nothing more. The benchmark's own
requirements -- a registry entry, an artifact strategy, two exact metric names -- are a different
contract and live in their own file.

The determinism and artifact cases are driven through the weight-editing method's fit seam with a
stand-in pipeline: a denoiser small enough to run on the processor in a second, a text encoder that
is a fixed embedding, and the real scheduler. That is the whole point of the seam existing.
"""
import inspect
import pathlib
from typing import Any, Dict, List, Tuple

import pytest
import torch

from huggingface_hub.repocard_data import EvalResult


def _unlearner_classes() -> List[Tuple[str, Any]]:
    """Every concrete unlearner the package exports, found by walking the base class."""
    import vision_unlearning.unlearner as package
    from vision_unlearning.unlearner.base import Unlearner

    found: List[Tuple[str, Any]] = []
    for name in dir(package):
        candidate = getattr(package, name)
        if not isinstance(candidate, type) or not issubclass(candidate, Unlearner):
            continue
        if candidate is Unlearner or inspect.isabstract(candidate):
            continue
        found.append((name, candidate))
    return sorted(found)


def _minimal_fields(name: str) -> Dict[str, Any]:
    """The smallest set of documented fields each class needs, and nothing else."""
    from vision_unlearning.utils.gradient_weighting import (
        GradientWeightingMethodMunba,
        GradientWeightingMethodSimple,
    )

    if name == "UCE":
        return {"pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4", "edit_concepts": "a cat"}
    if name == "ESD":
        return {"pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4", "erase_concept": "a cat"}
    weighting: Any = GradientWeightingMethodMunba() if name == "UnlearnerLoraDirect" \
        else GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0)
    fields: Dict[str, Any] = {
        "model_name_or_path": "CompVis/stable-diffusion-v1-4",
        "dataset_forget_name": "unused",
        "dataset_retain_name": "unused",
        "gradient_weighting_method": weighting,
    }
    if "Sparse" in name:
        from vision_unlearning.utils.parameter_attribution import ParameterAttributionMethodSaliency

        fields["parameter_attribution_method"] = ParameterAttributionMethodSaliency()
    return fields


@pytest.mark.parametrize("name,klass", _unlearner_classes())
def test_it_constructs_from_its_documented_fields(name: str, klass: Any) -> None:
    """Construction must need only fields the class documents, with no checkpoint touched."""
    instance = klass(**_minimal_fields(name))
    assert instance.output_dir


@pytest.mark.parametrize("name,klass", _unlearner_classes())
def test_an_unrecognised_field_raises_instead_of_vanishing(name: str, klass: Any) -> None:
    """Two hyperparameters were silently dropped for the whole life of this package.

    The mutation that must fail this test is removing the strict extra policy from the base class.
    """
    from pydantic import ValidationError

    fields = _minimal_fields(name)
    fields["a_field_that_does_not_exist"] = 1
    with pytest.raises(ValidationError):
        klass(**fields)


@pytest.mark.parametrize("name,klass", _unlearner_classes())
def test_training_promises_evaluation_records(name: str, klass: Any) -> None:
    """The base class promises one thing, and it promises this return type."""
    annotation = inspect.signature(klass.train).return_annotation
    assert annotation in (List[EvalResult], "List[EvalResult]"), f"{name}.train returns {annotation!r}"


@pytest.mark.parametrize("name,klass", _unlearner_classes())
def test_a_weight_editing_method_can_rebuild_its_own_pipeline(name: str, klass: Any) -> None:
    """A method that writes a partial weight file must ship the loader that applies it.

    Nothing else in the package can know how to apply those tensors, so a method without this is a
    method whose artifact cannot be used. Low-rank adapter methods are exempt: their artifact is
    loaded by the pipeline class itself.
    """
    if not hasattr(klass, "get_pipeline_from_modified_weights"):
        pytest.skip(f"{name} produces a low-rank adapter, which the pipeline class loads itself")
    loader = klass.get_pipeline_from_modified_weights
    parameters = list(inspect.signature(loader).parameters)
    assert parameters == ["pretrained_model_name_or_path", "device", "output_dir"], parameters


##########################################
# The fit seam, driven with a stand-in
##########################################

class _StandInPipeline:
    """The members a weight-editing unlearner touches, and nothing else.

    The denoiser is a real cross-attention denoiser at a size that runs on the processor, so the
    optimizer step under test is a real one. The text encoder is replaced by a fixed embedding
    because what is being asserted is the training loop, not the encoder.
    """

    def __init__(self) -> None:
        from diffusers import DDIMScheduler, UNet2DConditionModel

        torch.manual_seed(0)
        self.unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 8),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=8,
            attention_head_dim=2,
            norm_num_groups=4,
        )
        self.unet.requires_grad_(False)
        self.scheduler = DDIMScheduler(num_train_timesteps=20)
        self.vae: Any = None
        self.text_encoder: Any = None
        self.vae_scale_factor = 8
        self._embeddings: Dict[str, torch.Tensor] = {}

    def set_progress_bar_config(self, *arguments: Any, **keywords: Any) -> None:
        return None

    def encode_prompt(self, prompt: str, **keywords: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """A deterministic embedding per prompt string, so two runs see identical conditioning."""
        for text in (prompt, ""):
            if text not in self._embeddings:
                seed = sum(ord(character) for character in text) + len(text)
                generator = torch.Generator().manual_seed(seed)
                self._embeddings[text] = torch.randn((1, 3, 8), generator=generator)
        return self._embeddings[prompt], self._embeddings[""]


def _esd_on_a_stand_in(output_dir: pathlib.Path, **overrides: Any) -> Any:
    from vision_unlearning.unlearner import ESD

    fields: Dict[str, Any] = {
        "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
        "erase_concept": "a cat",
        "train_method": "esd-x",
        "num_train_epochs": 2,
        "num_inference_steps": 3,
        "resolution": 64,
        "device": "cpu",
        "seed": 7,
        "output_dir": str(output_dir),
    }
    fields.update(overrides)
    return ESD(**fields)


def test_the_fit_seam_runs_with_no_checkpoint_no_network_and_no_card(tmp_path: pathlib.Path) -> None:
    """The claim the seam exists for. Before it, no test could observe an optimizer step at all."""
    unlearner = _esd_on_a_stand_in(tmp_path)
    trained = unlearner._fit(_StandInPipeline())

    assert trained, "the fit seam trained no tensor"
    assert all(torch.isfinite(tensor).all() for tensor in trained.values())


def test_the_same_seed_produces_the_same_tensors_twice(tmp_path: pathlib.Path) -> None:
    """Reproducibility is asserted, not assumed.

    The reference implementation draws its denoising depth and its sampling noise from the global
    random state, which nothing seeds; this port owns that stream. The mutation that must fail this
    test is seeding it from the clock.
    """
    first = _esd_on_a_stand_in(tmp_path)._fit(_StandInPipeline())
    second = _esd_on_a_stand_in(tmp_path)._fit(_StandInPipeline())

    assert sorted(first) == sorted(second)
    for name in first:
        assert torch.equal(first[name], second[name]), f"{name} differs between two seeded runs"


def test_a_different_seed_produces_different_tensors(tmp_path: pathlib.Path) -> None:
    """A run that ignored its seed entirely would satisfy the test above."""
    first = _esd_on_a_stand_in(tmp_path, seed=7)._fit(_StandInPipeline())
    second = _esd_on_a_stand_in(tmp_path, seed=8)._fit(_StandInPipeline())

    assert any(not torch.equal(first[name], second[name]) for name in first)


def test_only_the_selected_slice_of_the_denoiser_is_trained(tmp_path: pathlib.Path) -> None:
    """The cross-attention variant trains cross-attention and nothing else.

    It also asserts the denoiser handed in is left as it was found: the trained tensors live beside
    the frozen ones over one module, and a leak there would silently corrupt any caller that reuses
    the pipeline.
    """
    pipeline = _StandInPipeline()
    before = {name: parameter.detach().clone() for name, parameter in pipeline.unet.named_parameters()}
    unlearner = _esd_on_a_stand_in(tmp_path)

    trained = unlearner._fit(pipeline)

    assert all("attn2" in name for name in trained), sorted(trained)[:3]
    after = dict(pipeline.unet.named_parameters())
    for name, original in before.items():
        assert torch.equal(after[name], original), f"{name} was left modified in the denoiser itself"


def test_the_artifact_carries_exactly_the_trained_tensors(tmp_path: pathlib.Path) -> None:
    """The saved file is the whole of what the method produced, under the denoiser's own key names."""
    from safetensors.torch import load_file

    unlearner = _esd_on_a_stand_in(tmp_path)
    trained = unlearner._fit(_StandInPipeline())
    unlearner._save_weights(trained)

    written = list(tmp_path.glob("*.safetensors"))
    assert len(written) == 1, [path.name for path in written]
    reloaded = load_file(str(written[0]))
    assert sorted(reloaded) == sorted(trained)
    for name in trained:
        assert torch.equal(reloaded[name], trained[name])
    assert written[0].stat().st_size > 0


def test_the_saved_tensors_apply_back_onto_a_denoiser_by_name(tmp_path: pathlib.Path) -> None:
    """The half of the round trip that needs no checkpoint.

    Every saved key matches a parameter, every matched parameter takes the saved value, and nothing
    else moves. The public loader does exactly this against a downloaded base model; asserting it
    here against the stand-in is what makes the key convention testable without the network. The
    mutation that must fail it is saving under a prefixed or renamed key.
    """
    from safetensors.torch import load_file

    unlearner = _esd_on_a_stand_in(tmp_path)
    trained = unlearner._fit(_StandInPipeline())
    unlearner._save_weights(trained)
    saved = load_file(str(next(iter(tmp_path.glob("*.safetensors")))))

    fresh = _StandInPipeline()
    untouched = {name: parameter.detach().clone() for name, parameter in fresh.unet.named_parameters()}

    applied = 0
    with torch.no_grad():
        for name, parameter in fresh.unet.named_parameters():
            if name in saved:
                parameter.copy_(saved[name])
                applied += 1

    assert applied == len(saved), f"{len(saved)} tensors saved, {applied} matched a denoiser parameter"
    for name, parameter in fresh.unet.named_parameters():
        if name in saved:
            assert torch.equal(parameter, saved[name])
        else:
            assert torch.equal(parameter, untouched[name]), f"{name} moved without being saved"


def test_no_unlearner_switches_gradients_off_globally() -> None:
    """Training after a closed-form edit, in one process, must still be able to compute gradients.

    The closed-form method called `torch.set_grad_enabled(False)` and never restored it, so every
    later optimizer step in the same process died with "element 0 of tensors does not require grad
    and does not have a grad_fn". Both training methods failed exactly that way when it had run
    first, and the campaign scripts loop over methods in one process.

    This is a source-level scan and its limitation is worth stating: it catches the bare call, not
    every way global state can leak. The behavioural evidence is the gpu tier, where the closed-form
    method runs before both training methods by test order.

    The scan parses rather than greps, so a docstring explaining this rule does not trip it.
    """
    import ast
    import pathlib as _pathlib

    import vision_unlearning.unlearner as package

    def calls_it(source: str) -> bool:
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr == "set_grad_enabled":
                return True
        return False

    directory = _pathlib.Path(package.__file__).parent
    offenders = [
        path.name for path in sorted(directory.glob("*.py"))
        if calls_it(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"{offenders} change the global gradient mode. Use `torch.no_grad()` as a context manager or "
        "a decorator, which restores the previous mode when it exits."
    )
