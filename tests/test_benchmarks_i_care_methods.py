"""The I-CARE benchmark's contract with an unlearning method.

This is not the library contract. `Unlearner` promises `train() -> List[EvalResult]` and nothing
more, and a method that satisfies only that is a perfectly good library method. What the benchmark
adds is a registry entry, a declared artifact kind, a dispatch branch, and two exact metric names
that its equalization procedure tunes on -- and every one of those is a place where adding a fifth
method silently does the wrong thing rather than failing.

**Everything here is parametrized over `ALGORITHM_REGISTRY` rather than over a hand-written list.**
That is the whole design: a method registered without an artifact strategy fails at collection of
the parametrization, not at run time in a campaign that has already spent a week of graphics time.
"""
from __future__ import annotations

import inspect
import io
import os
from typing import Any, Dict, List, Tuple

import pytest

from vision_unlearning.benchmarks.I_care.configuration import (
    ALGORITHM_REGISTRY,
    _UNLEARNING_ALGORITHM_DISPLAY_ORDER,
    type_unlearning_algorithm,
)

# The two strings the benchmark gates on. `check_eval_results` matches with `startswith`, so what a
# method must guarantee is the prefix, not the whole formatted name.
FORGET_METRIC_PREFIX = "ForgetSet clip score difference between original and unlearned mean"
RETAIN_METRIC_PREFIX = "RetainSet clip score difference between original and unlearned mean"

_REGISTERED: List[str] = sorted(ALGORITHM_REGISTRY)

# The folder strings the three methods that existed before this file produced. Recorded rather than
# recomputed: the point of the assertion is that they never change, and a check that recomputes them
# from the same code cannot fail.
_FOLDER_STRINGS_BEFORE: Dict[Tuple[str, int], str] = {
    ("distil", 100): os.path.join("assets", "models", "people_Bush_distil_100"),
    ("munba", 200): os.path.join("assets", "models", "people_Bush_munba_200"),
    ("uce", 0): os.path.join("assets", "models", "people_Bush_uce_000"),
}


def _repository_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source_of(relative_path: str) -> str:
    with io.open(os.path.join(_repository_root(), relative_path), encoding="utf-8") as handle:
        return handle.read()


##########################################
# The registry agrees with itself
##########################################

def test_the_registry_and_the_type_agree_in_both_directions() -> None:
    """A member of one and not the other is a method that half exists."""
    from typing import get_args

    declared = set(get_args(type_unlearning_algorithm))
    assert set(ALGORITHM_REGISTRY) == declared
    assert set(_UNLEARNING_ALGORITHM_DISPLAY_ORDER) == declared
    assert len(_UNLEARNING_ALGORITHM_DISPLAY_ORDER) == len(declared), "a method appears twice in the display order"


@pytest.mark.parametrize("method", _REGISTERED)
def test_every_registered_method_declares_how_its_artifact_is_loaded(method: str) -> None:
    """The knowledge "what kind of file does this method produce" is declared once, here.

    It used to be encoded four times as a method-name test with a default of "it is a low-rank
    adapter", which is invisible to any test that only exercises the methods that already existed.
    """
    specification = ALGORITHM_REGISTRY[method]  # type: ignore[index]
    assert specification.artifact_kind in ("lora_adapter", "lora_adapter_inverted", "partial_weights")
    assert specification.artifact_filename.endswith(".safetensors")


@pytest.mark.parametrize("method", _REGISTERED)
def test_every_registered_method_has_a_dispatch_branch(method: str) -> None:
    """A registered method with no branch constructs nothing and fails only when a campaign runs.

    This is a source-level check, and its limitation is worth stating rather than hiding: the two
    benchmark pipelines execute at import (they read an environment token and load metadata), so
    they cannot be imported into a test process. What this catches is the real failure -- a method
    added to the registry and forgotten in the dispatch. What it cannot catch is a branch that
    constructs the wrong class.
    """
    for script in ("pipeline_03_unlearn_model.py", "pipeline_04_generate_dataset.py"):
        source = _source_of(os.path.join("vision_unlearning", "benchmarks", "I_care", script))
        assert f'"{method}"' in source or f"'{method}'" in source, (
            f"{script} has no branch naming the registered method {method!r}"
        )


@pytest.mark.parametrize("method", _REGISTERED)
def test_the_artifact_kind_dispatch_is_exhaustive(method: str) -> None:
    """Every declared kind is handled where the benchmark loads a model."""
    from vision_unlearning.datasets.testbed import ARTIFACT_KIND_LOADERS

    specification = ALGORITHM_REGISTRY[method]  # type: ignore[index]
    assert specification.artifact_kind in ARTIFACT_KIND_LOADERS


##########################################
# Paths
##########################################

@pytest.mark.parametrize("method", _REGISTERED)
def test_two_strengths_do_not_collide_on_one_path(method: str) -> None:
    """The strength integer is what separates two runs of one method on one entity."""
    from vision_unlearning.datasets.testbed import get_unlearned_model_folder

    weak = get_unlearned_model_folder("people", method, 10, "Bush")  # type: ignore[arg-type]
    strong = get_unlearned_model_folder("people", method, 20, "Bush")  # type: ignore[arg-type]
    assert weak != strong


@pytest.mark.parametrize("method,epochs,expected", [
    (method, epochs, expected) for (method, epochs), expected in _FOLDER_STRINGS_BEFORE.items()
])
def test_the_folder_strings_of_the_existing_methods_are_unchanged(method: str, epochs: int, expected: str) -> None:
    """Registering a fourth method must not move a single artifact that already exists on disk."""
    from vision_unlearning.datasets.testbed import get_unlearned_model_folder

    assert get_unlearned_model_folder("people", method, epochs, "Bush") == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("method", _REGISTERED)
def test_the_existence_check_looks_for_the_declared_filename(method: str, tmp_path: Any) -> None:
    """The existence check took the method name and inferred a filename from it, twice.

    It now takes the filename, so `testbed` holds no opinion about what any method produces. The
    mutation that must fail this test is going back to inferring it.
    """
    from vision_unlearning.datasets.testbed import exists_unlearned_model, get_unlearned_model_folder

    specification = ALGORITHM_REGISTRY[method]  # type: ignore[index]
    base = str(tmp_path)
    folder = get_unlearned_model_folder("people", method, 7, "Bush", base_folder=base)  # type: ignore[arg-type]

    assert not exists_unlearned_model(
        "people", method, 7, "Bush", artifact_filename=specification.artifact_filename, base_folder=base,  # type: ignore[arg-type]
    )

    os.makedirs(folder, exist_ok=True)
    with io.open(os.path.join(folder, specification.artifact_filename), "w", encoding="utf-8") as handle:
        handle.write("")

    assert exists_unlearned_model(
        "people", method, 7, "Bush", artifact_filename=specification.artifact_filename, base_folder=base,  # type: ignore[arg-type]
    )


##########################################
# The baseline route
##########################################

def test_the_baseline_images_are_never_produced_by_an_edited_model() -> None:
    """The `off` pass is the base model, for every method, unconditionally.

    It was inside the per-method branch, and the closed-form method's branch built the *edited*
    pipeline for both passes -- so its baseline images were of the edited model. The fix is that the
    branch which could do that no longer exists, and this asserts the shape rather than the outcome:
    the generation call for the `off` pass is made outside any method test.
    """
    source = _source_of(os.path.join("vision_unlearning", "benchmarks", "I_care", "pipeline_04_generate_dataset.py"))
    assert "def _generate_baseline_pass(" in source, (
        "the off pass must be one named function, so it cannot acquire a per-method branch again"
    )
    baseline = source.split("def _generate_baseline_pass(", 1)[1].split("\ndef ", 1)[0]
    for forbidden in ("method ==", "method !=", "artifact_kind"):
        assert forbidden not in baseline, (
            f"the baseline pass tests {forbidden!r}; a baseline image must not depend on the method"
        )


##########################################
# The two metric names equalization tunes on
##########################################

class _StandInImage:
    """Whatever the evaluator does with a generated image, it does with this."""

    def __init__(self) -> None:
        from PIL import Image

        self.image = Image.new("RGB", (8, 8))


class _StandInPipeline:
    """Callable like a diffusion pipeline, returning one image per prompt."""

    def __init__(self, score: float) -> None:
        self.score = score

    def __call__(self, prompt: str, **keywords: Any) -> Any:
        from PIL import Image

        class _Result:
            images = [Image.new("RGB", (8, 8))]

        return _Result()


class _StandInMetric:
    """A fixed score per pipeline identity, so the differences are known in advance."""

    def __init__(self, scores: Dict[int, float]) -> None:
        self.scores = scores
        self.current = 0.0

    def score(self, image: Any, prompt: str) -> Dict[str, float]:
        return {"clip": self.current}


def test_the_two_metric_names_equalization_tunes_on_are_produced() -> None:
    """A method returning other names type-checks perfectly and is unusable in this benchmark.

    `check_eval_results` matches with `startswith`, so the contract is on the prefix. Driven with
    stand-in pipelines and a stand-in metric: no card, no network, no checkpoint.
    """
    import matplotlib

    matplotlib.use("Agg")

    from vision_unlearning.evaluator import EvaluatorTextToImage

    original = _StandInPipeline(0.9)
    unlearned = _StandInPipeline(0.1)
    metric = _StandInMetric({})

    evaluator = EvaluatorTextToImage.model_construct(
        pipeline_original=original,
        pipeline_learned=None,
        pipeline_unlearned=unlearned,
        prompts_forget=["a cat"],
        prompts_retain=["a dog"],
        metric_clip=metric,
        compute_runtimes=False,
        plot_show=False,
    )

    results, _ = evaluator.evaluate()
    names = [record.metric_name for record in results]

    for prefix in (FORGET_METRIC_PREFIX, RETAIN_METRIC_PREFIX):
        assert any(name.startswith(prefix) for name in names), (
            f"no evaluation record starts with {prefix!r}; equalization has nothing to tune on.\n"
            f"produced: {names}"
        )


def test_the_benchmark_gate_reads_those_records_by_prefix() -> None:
    """The other half of the same contract: the gate finds a record by that prefix and nothing else."""
    from vision_unlearning.benchmarks.I_care import check_eval_results

    source = inspect.getsource(check_eval_results)
    assert "startswith" in source, (
        "the gate no longer matches by prefix, so the contract this file asserts is the wrong one"
    )


##########################################
# The substitute concept comes from the accessor, not from a literal
##########################################

def test_the_substitute_concept_is_read_from_the_accessor_not_written_as_a_literal() -> None:
    """A distillation method's target concept must come from `get_target_overwrite`.

    This is the defect that was found in one existing method by running it: the trainer had been
    conditioned on a phrase written into the call site, while the images were generated and evaluated
    with the phrase the accessor returns. The two agreed until one of them was edited, and the
    disagreement was invisible because both are plausible strings.

    The check is on the shape of the wiring rather than on a value, because a value test would pass
    against a literal that happened to be spelled the same way today. What must be true is that the
    two pipelines read the concept from the accessor and hand *that* to the unlearner.
    """
    for script in ("pipeline_03_unlearn_model.py", "pipeline_04_generate_dataset.py"):
        source = _source_of(os.path.join("vision_unlearning", "benchmarks", "I_care", script))
        assert "get_target_overwrite(" in source, (
            f"{script} does not call get_target_overwrite; the concept must not be spelled out locally"
        )
        assert '"overwriting_concept": target_overwrite' in source, (
            f"{script} does not pass the accessor's result as the overwriting concept"
        )
        for literal in ('"overwriting_concept": "', "'overwriting_concept': '"):
            assert literal not in source, (
                f"{script} writes the overwriting concept as a literal, which can drift away from "
                "the phrase the images are generated and scored with"
            )


def test_the_forget_and_retain_splits_are_the_benchmark_s_own() -> None:
    """Nothing is generated for the new method: it consumes the splits that already exist.

    Its two dataset arguments must be the same variables every other data-driven method is given, so
    that adding it created no new corpus and no new generation step.
    """
    source = _source_of(os.path.join("vision_unlearning", "benchmarks", "I_care", "pipeline_03_unlearn_model.py"))
    salun_block = source.split("elif method == 'salun':", 1)[1].split("    else:", 1)[0]
    assert '"dataset_forget_name": dataset_forget_name' in salun_block
    assert '"dataset_retain_name": dataset_retain_name' in salun_block
