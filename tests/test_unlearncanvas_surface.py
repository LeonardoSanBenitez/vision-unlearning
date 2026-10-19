"""The UnlearnCanvas benchmark's view of this library is unchanged by the I-CARE regeneration work.

``benchmarks/u_care`` lives on its own branch. This file records the exact set of library names that
branch imports, read from its sources, and asserts each still resolves here, together with the
defaults of every setting the regeneration work added to code it uses: each was added opt-in, so a
caller that never passes it behaves as before.

Four of the names that branch imports are defined on the branch itself and never existed on this
one -- ``MetricImageClassifier`` in its own ``metrics/image.py`` and the three
``UnlearnerLoraDistillation`` classes in its ``unlearner/fade.py``, which predate the rename of that
trainer to SPARE -- so they are listed apart and only their absence is recorded here: the branch
brings them with it.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

# Every (module, name) the UnlearnCanvas branch imports from this library.
SURFACE = [
    ('vision_unlearning.artifact', 'Artifact'),
    ('vision_unlearning.artifact', 'ArtifactNotAvailableError'),
    ('vision_unlearning.artifact', 'SingleFileArtifact'),
    ('vision_unlearning.benchmarks.care', 'MetricEffectPerEntity'),
    ('vision_unlearning.benchmarks.care', 'MetricEffectPerEntityPair'),
    ('vision_unlearning.benchmarks.configuration', 'MetricWithDirectionSpec'),
    ('vision_unlearning.benchmarks.configuration', 'UnlearningAlgorithmSpec'),
    ('vision_unlearning.benchmarks.result_template', 'ResultTemplate'),
    ('vision_unlearning.integrations.huggingface', 'huggingface_dataset_download'),
    ('vision_unlearning.integrations.huggingface', 'huggingface_dataset_exists'),
    ('vision_unlearning.integrations.huggingface', 'huggingface_dataset_upload'),
    ('vision_unlearning.utils.gradient_weighting', 'GradientWeightingMethodSimple'),
    ('vision_unlearning.utils.logger', 'get_logger'),
    ('vision_unlearning.utils.logger', 'setup_loggers'),
]


# Imported by the branch, defined by the branch: not part of what this branch must keep providing.
BRANCH_LOCAL = [
    ('vision_unlearning.metrics.image', 'MetricImageClassifier'),
    ('vision_unlearning.unlearner', 'UnlearnerLoraDistillation'),
    ('vision_unlearning.unlearner', 'UnlearnerLoraDistillationSparsePerModule'),
    ('vision_unlearning.unlearner', 'UnlearnerLoraDistillationSparsePerWeight'),
]


@pytest.mark.parametrize('module, name', SURFACE)
def test_every_imported_name_resolves(module: str, name: str) -> None:
    assert hasattr(importlib.import_module(module), name)


def test_the_trainer_and_evaluator_defaults_are_the_historical_ones() -> None:
    from vision_unlearning.evaluator.text_to_image import EvaluatorTextToImage
    from vision_unlearning.unlearner import UnlearnerSpare
    from vision_unlearning.unlearner.base import Unlearner
    from vision_unlearning.unlearner.salun import SalUn

    assert Unlearner.model_fields['evaluation_noise_seed'].default is None
    assert UnlearnerSpare.model_fields['evaluation_noise_seed'].default is None
    assert EvaluatorTextToImage.model_fields['noise_seed'].default is None
    assert EvaluatorTextToImage.model_fields['plot_show'].default is True
    assert SalUn.model_fields['save_at_epochs'].default_factory() == []  # type: ignore[misc]
    assert SalUn.model_fields['allow_tf32'].default is False


def test_the_generation_and_metric_defaults_are_the_historical_ones() -> None:
    from vision_unlearning.datasets.testbed import GeneratedDataset
    from vision_unlearning.metrics.image_and_image import MetricImageImage
    from vision_unlearning.utils.data_generation import generate_dataset

    parameters = inspect.signature(generate_dataset).parameters
    assert parameters['noise'].default == 'stream' and parameters['provenance'].default is None
    assert GeneratedDataset.model_fields['noise'].default == 'stream'
    assert GeneratedDataset.model_fields['provenance'].default is None
    assert MetricImageImage.model_fields['max_pixel_value'].default is None


def test_the_artifact_default_repository_is_the_historical_one() -> None:
    from vision_unlearning.artifact import DEFAULT_REMOTE_REPOSITORY_NAME
    assert DEFAULT_REMOTE_REPOSITORY_NAME == 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'


@pytest.mark.parametrize('module, name', BRANCH_LOCAL)
def test_the_branch_local_names_are_still_the_branch_s_own(module: str, name: str) -> None:
    """If one of these appears here, it belongs in SURFACE, and this branch now has to keep it."""
    assert not hasattr(importlib.import_module(module), name)
