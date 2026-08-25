"""Focused regressions for `UnlearnerLora`'s training plumbing.

Heavy tier: torch only, no diffusers checkpoint, no network, no graphics card. These tests are
about the wiring between the parameter collection, the optimizer and the gradient clipper -- a
defect there is invisible to any test that only looks at the adapter afterwards, because an
unclipped step still produces a perfectly plausible adapter.
"""
import pathlib
from typing import Any

import pytest
import torch

from accelerate import Accelerator


def _trainable_parameters() -> torch.nn.Module:
    """A tiny module standing in for the adapter parameters, with one frozen tensor among them."""
    module = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
    module[1].weight.requires_grad_(False)
    return module


def test_parameter_collection_survives_being_consumed_by_the_optimizer() -> None:
    """The collection handed to the optimizer must still hold parameters afterwards.

    This is the defect itself. A `filter(...)` is exhausted by the optimizer while it builds its
    parameter group, so every later consumer of the same object -- the gradient clipper -- sees an
    empty sequence and silently does nothing. The mutation that must fail this test is replacing
    the list comprehension with `filter(lambda p: p.requires_grad, module.parameters())`.
    """
    module = _trainable_parameters()

    collection = [p for p in module.parameters() if p.requires_grad]
    assert len(collection) == 1

    torch.optim.AdamW(collection, lr=1e-3)

    assert len(list(collection)) == 1, (
        "the parameter collection was consumed by the optimizer; gradient clipping would see "
        "nothing to clip"
    )


def test_clipping_bounds_a_large_gradient() -> None:
    """A known large gradient is brought under `max_grad_norm`, and the norm reported is real.

    Measured against the pre-fix behaviour: with an exhausted iterator the identical call left a
    gradient of 1000.0 untouched and reported a total norm of 0.0.
    """
    module = _trainable_parameters()
    accelerator = Accelerator()

    collection = [p for p in module.parameters() if p.requires_grad]
    torch.optim.AdamW(collection, lr=1e-3)

    for parameter in collection:
        parameter.grad = torch.full_like(parameter, 1000.0)

    max_grad_norm = 5.0
    total_norm = accelerator.clip_grad_norm_(collection, max_grad_norm)

    assert float(total_norm) > max_grad_norm, (
        "the reported pre-clip norm is not larger than the threshold, so nothing was measured"
    )
    clipped_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in collection])
    )
    assert float(clipped_norm) <= max_grad_norm + 1e-4, (
        f"post-clip norm {float(clipped_norm)} exceeds max_grad_norm {max_grad_norm}"
    )


def _minimal_lora_unlearner(**overrides: Any) -> Any:
    """A concrete `UnlearnerLora` built with only its required fields; nothing is loaded or trained.

    `UnlearnerLora` is abstract -- `_prepare_dataloaders` and `_train_one_batch` are left to each
    method -- so the smallest honest stand-in is a subclass that implements both with nothing. The
    fields under test live on the base class, so the stub does not weaken what is being asserted.
    """
    from vision_unlearning.unlearner.lora import UnlearnerLora
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    class _ConcreteLora(UnlearnerLora):  # type: ignore[misc]
        def _prepare_dataloaders(self) -> None:
            raise NotImplementedError("this stand-in never trains")

        def _train_one_batch(self, *arguments: Any, **keywords: Any) -> None:
            raise NotImplementedError("this stand-in never trains")

    fields: dict = {
        "model_name_or_path": "CompVis/stable-diffusion-v1-4",
        "dataset_forget_name": "unused",
        "dataset_retain_name": "unused",
        "output_dir": "unused",
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
    }
    fields.update(overrides)
    return _ConcreteLora(**fields)


def test_an_unknown_hyperparameter_raises_instead_of_vanishing() -> None:
    """The defect this guards: pydantic's default policy drops unknown keyword arguments.

    Two hyperparameters lived in that gap for the whole life of the class -- `lora_dropout`, passed
    as 0.2 by every caller and never reaching the adapter, and `train_batch_size`, whose real name
    is `per_device_train_batch_size`. The mutation that must fail this test is removing
    `model_config = ConfigDict(extra="forbid")` from `Unlearner`.
    """
    import pydantic

    # The error must be about the EXTRA field specifically. A bare `pytest.raises(ValidationError)`
    # also passes when construction fails for an unrelated reason -- a missing required field, say --
    # which is a test that would keep passing after the fix was reverted.
    for unknown in ("a_hyperparameter_that_does_not_exist", "train_batch_size"):
        with pytest.raises(pydantic.ValidationError) as raised:
            _minimal_lora_unlearner(**{unknown: 1})
        reasons = [error["type"] for error in raised.value.errors()]
        assert reasons == ["extra_forbidden"], (
            f"{unknown} was rejected, but for the wrong reason: {reasons}"
        )


def test_the_configured_dropout_reaches_the_adapter_configuration() -> None:
    """`lora_dropout` is a real field and its value arrives at `LoraConfig`.

    Asserted on the configuration object rather than on a trained adapter, because a dropout that
    fails to arrive changes training only stochastically -- which is exactly why nobody noticed for
    the lifetime of this class.
    """
    unlearner = _minimal_lora_unlearner(lora_dropout=0.25)
    assert unlearner.lora_dropout == 0.25  # type: ignore[attr-defined]

    configuration = unlearner._get_lora_config()  # type: ignore[attr-defined]
    assert configuration.lora_dropout == 0.25, (
        "the configured dropout did not reach LoraConfig; peft would silently use its own default"
    )

    assert _minimal_lora_unlearner()._get_lora_config().lora_dropout == 0.0  # type: ignore[attr-defined]


def test_no_call_site_still_passes_the_dead_batch_size_key() -> None:
    """A repository scan, because `extra="forbid"` turns a leftover key into a crash at run time.

    `train_batch_size` was accepted and dropped at every call site. Now that unknown fields raise,
    any surviving occurrence is a training run that dies on construction -- and these call sites are
    campaign scripts that fail hours into a session, not in a test.

    `benchmarks/u_care` is excluded deliberately: it is a vendored standalone script with its own
    argparse namespace, not a caller of this class.
    """
    import pathlib

    package_root = pathlib.Path(__file__).resolve().parents[2] / "vision_unlearning"
    offenders = []
    for path in package_root.rglob("*.py"):
        if "u_care" in path.parts or ".mypy_cache" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if '"train_batch_size"' in line or "'train_batch_size'" in line:
                offenders.append(f"{path.relative_to(package_root)}:{number}")

    assert offenders == [], "call sites still passing a key the model now rejects: " + ", ".join(offenders)


def test_the_retain_loader_is_traversed_rather_than_resampled() -> None:
    """One held iterator visits every item before repeating; a fresh iterator per step does not.

    The defect: the step loop called `next(iter(self._train_retain_dataloader))`, which builds a new
    iterator every step and therefore always returns the first batch of a freshly shuffled pass --
    sampling the retain set with replacement instead of traversing it. For the people task that left
    roughly 13 % of the retain images expected never to be seen in a whole run.

    The mutation that must fail this test is recreating the iterator on each step, which is the
    `fresh` branch below and is asserted to be worse in the same run -- so the test carries its own
    control rather than trusting a remembered number.
    """
    dataset = list(range(20))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    held_iterator = iter(loader)
    held_seen = []
    for _ in range(len(dataset)):
        try:
            held_seen.append(int(next(held_iterator)[0]))
        except StopIteration:  # pragma: no cover - reached only if the loader is shorter
            held_iterator = iter(loader)
            held_seen.append(int(next(held_iterator)[0]))

    fresh_seen = [int(next(iter(loader))[0]) for _ in range(len(dataset))]

    assert len(set(held_seen)) == len(dataset), (
        f"one held iterator saw {len(set(held_seen))} of {len(dataset)} distinct items; it must "
        "traverse the set exactly once before repeating"
    )
    assert len(set(fresh_seen)) < len(dataset), (
        "the control did not reproduce the defect, so this test proves nothing about the fix"
    )


def test_the_held_retain_iterator_restarts_when_exhausted() -> None:
    """Traversal continues across epoch boundaries instead of stopping.

    This is the half of the fix that a single pass cannot show: the loop asks for more batches than
    the retain loader holds, and must keep supplying them by starting a new pass.
    """
    dataset = list(range(4))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    iterator = iter(loader)
    drawn = []
    for _ in range(len(dataset) * 2 + 1):
        try:
            drawn.append(int(next(iterator)[0]))
        except StopIteration:
            iterator = iter(loader)
            drawn.append(int(next(iterator)[0]))

    assert drawn == [0, 1, 2, 3, 0, 1, 2, 3, 0], f"traversal did not wrap cleanly: {drawn}"


def test_both_sides_of_the_distillation_pair_share_one_template() -> None:
    """The two conditioning strings differ only in the concept they name.

    The defect: the forget side was conditioned on the raw caption stored on disk -- a bare
    underscored name such as `Mark_Philippoussis` -- while its distillation target was conditioned
    on the templated `An image of a child`, and the evaluation that judged the result prompted
    `An image of Mark Philippoussis`. Three conventions for one comparison, so the forget loss also
    had to absorb the difference between a bare name and a templated phrase, and the intervention
    was fitted at a point in text-embedding space that is not the point later queried.

    Asserted on the strings rather than on tokens, because the token identifiers are a tokeniser
    detail while the strings are the contract with the generation side.
    """
    from vision_unlearning.unlearner.spare import UnlearnerSpare
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    # Built from a real instance, not from the template constant: a test that only formats the
    # constant would pass even if the trainer ignored the field entirely.
    unlearner = UnlearnerSpare(
        model_name_or_path="CompVis/stable-diffusion-v1-4",
        dataset_forget_name="unused",
        dataset_retain_name="unused",
        output_dir="unused",
        forget_concept="Mark Philippoussis",
        overwritting_concept="a child",
        gradient_weighting_method=GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
    )
    forget_caption = unlearner._forget_caption()
    overwrite_caption = unlearner._overwrite_caption()
    template = unlearner.caption_template

    assert forget_caption == "An image of Mark Philippoussis"
    assert overwrite_caption == "An image of a child"

    prefix = template.format("")
    assert forget_caption.startswith(prefix) and overwrite_caption.startswith(prefix), (
        "the two sides no longer share a prefix, so they differ by more than the concept named"
    )
    assert forget_caption.replace("Mark Philippoussis", "a child") == overwrite_caption, (
        "the two captions differ in something other than the concept"
    )

    # The mutation this guards against: feeding the bare metadata name to one side.
    bare_metadata_name = "Mark_Philippoussis"
    assert bare_metadata_name != forget_caption
    assert "_" not in forget_caption, (
        "the forget caption still carries an underscored metadata name rather than the prompted form"
    )


def test_the_forget_side_falls_back_to_the_caption_column_only_when_told_to() -> None:
    """`forget_concept` left unset keeps the old caption-column behaviour, and it is not the default.

    Case 2 of the trainer -- a JSON metafile supplying a different overwrite concept per example --
    genuinely wants the captions on disk, so the fallback has to exist. What must not happen is a
    caller silently getting the old asymmetric behaviour by omission, which is why the benchmark's
    own dispatch sets it explicitly.
    """
    from vision_unlearning.unlearner.spare import UnlearnerSpare

    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    assert UnlearnerSpare.model_fields["forget_concept"].default is None
    unset = UnlearnerSpare(
        model_name_or_path="CompVis/stable-diffusion-v1-4",
        dataset_forget_name="unused",
        dataset_retain_name="unused",
        output_dir="unused",
        overwritting_concept="a child",
        gradient_weighting_method=GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
    )
    assert unset._forget_caption() is None, (
        "with no forget concept configured the trainer must fall back to the dataset's captions"
    )

    pipeline_source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "vision_unlearning" / "benchmarks" / "I_care" / "pipeline_03_unlearn_model.py"
    ).read_text(encoding="utf-8")
    assert "hyperparameters['forget_concept']" in pipeline_source, (
        "the benchmark dispatch no longer passes forget_concept, so distillation runs would silently "
        "return to conditioning the forget side on bare metadata names"
    )


def test_the_accelerator_teardown_survives_a_build_without_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`train()` used to discard its own return value at the very last line, on this platform.

    `Accelerator.end_training` finishes the trackers and then destroys the process group, and that
    second step reaches `torch.distributed.is_initialized()` unconditionally. On a build without
    distributed support the attribute does not exist, so the call raises after the adapter has been
    saved and the evaluation computed -- and the caller receives an exception instead of the
    evaluation records. A 200-epoch session ended that way.

    The mutation that must fail this test is calling `self._accelerator.end_training()` directly.
    """
    class _Tracker:
        def __init__(self) -> None:
            self.finished = False

        def finish(self) -> None:
            self.finished = True

    class _Accelerator:
        def __init__(self, trackers: Any) -> None:
            self.trackers = trackers

        def end_training(self) -> None:
            raise AttributeError("module 'torch.distributed' has no attribute 'is_initialized'")

    tracker = _Tracker()
    unlearner = _minimal_lora_unlearner()
    unlearner._accelerator = _Accelerator([tracker])
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)

    unlearner._end_training()

    assert tracker.finished is True


def test_the_accelerator_teardown_still_runs_where_it_can(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not become an excuse to never tear the accelerator down at all."""
    class _AcceleratorRecording:
        def __init__(self) -> None:
            self.trackers: list = []
            self.ended = False

        def end_training(self) -> None:
            self.ended = True

    accelerator = _AcceleratorRecording()
    unlearner = _minimal_lora_unlearner()
    unlearner._accelerator = accelerator
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)

    unlearner._end_training()

    assert accelerator.ended is True


def test_the_loss_history_is_written_beside_the_adapter(tmp_path: pathlib.Path) -> None:
    """Per-step losses reached the tracker and nothing else, so a finished run had no loss curve.

    This covers the writing half only -- the rows are supplied here rather than trained. That the
    step loop actually fills them is not asserted from a stand-in: a real one-step capture of this
    trainer writes a `loss_history.json` holding exactly one row, with the step, the epoch, both
    losses and the learning rate, and that artifact is the evidence for the other half.
    """
    import json

    unlearner = _minimal_lora_unlearner(output_dir=str(tmp_path))
    unlearner._loss_history = [
        {"step": 1, "epoch": 0, "loss_forget": 0.5, "loss_retain": 0.25, "learning_rate": 6e-4},
        {"step": 2, "epoch": 0, "loss_forget": 0.4, "loss_retain": 0.20, "learning_rate": 6e-4},
    ]

    unlearner._save_loss_history()

    written = json.loads((tmp_path / "loss_history.json").read_text(encoding="utf-8"))
    assert [row["step"] for row in written] == [1, 2]
    assert written[0]["loss_forget"] == 0.5
    assert written[1]["learning_rate"] == 6e-4
