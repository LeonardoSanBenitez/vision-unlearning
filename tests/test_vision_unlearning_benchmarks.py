"""Tests for vision_unlearning.benchmarks.I_care.

Scope of this suite (decided with the project owner):

1. static-analysis smoke: the module imports cleanly (no heavy model load
   at import time, no GPU).
2. pure / deterministic helpers: GUI->backend mapping, jacc similarity,
   metric-column resolution, path builders, target overwrite, and the
   SHAP Explanation <-> dict serialization roundtrip.
3. plot generation from FAKE in-memory data for every ResultTemplate that
   defines a ``plot`` (no files, no Hugging Face, no GPU).
4. ``_compute_from_scratch`` / ``compute`` for the representative
   ResultTemplates, with all file readers and Hugging Face existence
   checks mocked so nothing touches the network or the real datasets.

These tests deliberately do NOT assert anything about the methodological
correctness of ResultTemplateMetricSimilarityAlignmentMulti (the
train/test split, MAPE, feature-name strip): those are known open review
items that the project owner has explicitly deferred. The Multi RT is
still exercised end-to-end here to prove the pipeline runs.

All tests are CPU-only and fast; none require the ``gpu`` marker.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402
import vision_unlearning.benchmarks.I_care.result_templates as _rt_mod  # noqa: E402
import vision_unlearning.benchmarks.I_care.similarity as _sim_mod  # noqa: E402
import vision_unlearning.artifact as _artifact_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Static-analysis / import smoke
# ---------------------------------------------------------------------------
class TestImportSmoke:
    def test_module_imports(self) -> None:
        assert vb is not None

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") is not None,
        reason=(
            "torch is installed in this environment — the I_care package is "
            "allowed to transitively import torch via vision_unlearning.datasets "
            "when torch is present. This test only makes sense in torch-free "
            "environments (e.g. the forgety web container)."
        ),
    )
    def test_no_gpu_or_heavy_load_at_import(self) -> None:
        # The module must not pull torch / a model at import time in
        # torch-free environments. It only needs pandas/numpy/sklearn/shap/matplotlib.
        #
        # Run in a subprocess to guarantee a clean sys.modules state.
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import vision_unlearning.benchmarks.I_care; "
                "import vision_unlearning.benchmarks.I_care.result_templates; "
                "import sys; assert 'torch' not in sys.modules, "
                "'torch was imported at I_care module load time'"
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"I_care module loaded torch at import time.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_rt_registry_consistent(self) -> None:
        # Every class in the name->class map must also be in the params map.
        assert set(vb.rt_name_to_class) == set(vb.rt_name_to_params)
        for cls in vb.rt_name_to_class.values():
            assert issubclass(cls, vb.ResultTemplate)


# ---------------------------------------------------------------------------
# 2. Pure / deterministic helpers
# ---------------------------------------------------------------------------
class TestConvertParamsFromGuiToBackend:
    def test_known_mappings(self) -> None:
        out = vb.convert_params_from_gui_to_backend(
            {
                "unlearning_algorithm": "spare",
                "task": "People",
                "model": "Stable Diffusion 1.4",
                "interference_pair": "RMSE",
                "similarity_metric": "Jacc Similarity",
            }
        )
        assert out == {
            "unlearning_algorithm": "distil",
            "task": "people",
            "model": "sd1.4",
            "interference_pair": "rmse",
            "similarity_metric": "jacc",
        }

    def test_none_passthrough(self) -> None:
        assert vb.convert_params_from_gui_to_backend({"task": None}) == {"task": None}

    def test_unknown_key_passthrough(self) -> None:
        assert vb.convert_params_from_gui_to_backend({"foo": "bar"}) == {"foo": "bar"}

    def test_unknown_value_passthrough(self) -> None:
        assert vb.convert_params_from_gui_to_backend({"task": "Mars"}) == {"task": "Mars"}


class TestJaccMetricScore:
    @staticmethod
    def _meta() -> List[Dict[str, Any]]:
        return [
            {"name": "a", "cat": "x", "num01": 0.2, "num100": 50.0},
            {"name": "b", "cat": "x", "num01": 0.2, "num100": 50.0},
            {"name": "c", "cat": "y", "num01": 0.8, "num100": 90.0},
        ]

    def test_self_similarity_is_one(self) -> None:
        # Comparing an entity with itself: every attribute (including
        # 'name') matches exactly -> score 1.0.
        meta = self._meta()
        assert vb.jacc_metric_score("a", "a", meta) == pytest.approx(1.0)

    def test_entities_sharing_all_but_name(self) -> None:
        # 'name' is treated as an attribute by the function, so two
        # otherwise-identical rows differ on 1 of 4 attributes -> 0.75.
        meta = self._meta()
        assert vb.jacc_metric_score("a", "b", meta) == pytest.approx(0.75)

    def test_different_entities_score_in_range(self) -> None:
        meta = self._meta()
        score = vb.jacc_metric_score("a", "c", meta)
        assert 0.0 <= score <= 1.0
        assert score < 1.0

    def test_missing_entity_raises(self) -> None:
        with pytest.raises(ValueError):
            vb.jacc_metric_score("a", "missing", self._meta())


class TestChooseMetricColumn:
    def test_single_match(self) -> None:
        cols = [
            "metric_distil_400_emitter_average_rmse (↓)",
            "metric_distil_400_emitter_average_ssim (↑)",
        ]
        chosen = vb.choose_metric_column_interference_per_entity(
            "distil", "Emitter average rmse", cols
        )
        assert chosen == "metric_distil_400_emitter_average_rmse (↓)"

    def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError):
            vb.choose_metric_column_interference_per_entity(
                "munba", "Emitter average rmse", ["metric_distil_400_emitter_average_rmse (↓)"]
            )

    def test_forget_clip_diff_in_type_me(self) -> None:
        from vision_unlearning.benchmarks.I_care.configuration import type_me
        assert "Forget clip diff" in type_me.__args__

    def test_retain_average_clip_diff_in_type_me(self) -> None:
        from vision_unlearning.benchmarks.I_care.configuration import type_me
        assert "Retain average clip diff" in type_me.__args__

    def test_choose_metric_column_forget_clip_diff(self) -> None:
        cols = [
            "metric_distil_400_forget_clip_diff (↓)",
            "metric_distil_400_emitter_average_clip_diff (↑)",
        ]
        chosen = vb.choose_metric_column_interference_per_entity(
            "distil", "Forget clip diff", cols
        )
        assert chosen == "metric_distil_400_forget_clip_diff (↓)"

    def test_choose_metric_column_retain_average_clip_diff(self) -> None:
        cols = [
            "metric_distil_400_retain_average_clip_diff (↑)",
            "metric_distil_400_emitter_average_clip_diff (↑)",
        ]
        chosen = vb.choose_metric_column_interference_per_entity(
            "distil", "Retain average clip diff", cols
        )
        assert chosen == "metric_distil_400_retain_average_clip_diff (↑)"

    def test_forget_in_domain_me(self) -> None:
        from vision_unlearning.benchmarks.I_care.configuration import domain_me
        assert "Forget clip diff" in domain_me

    def test_retain_in_domain_me(self) -> None:
        from vision_unlearning.benchmarks.I_care.configuration import domain_me
        assert "Retain average clip diff" in domain_me


class TestPathHelpers:
    def test_interference_per_entity_path(self) -> None:
        p = vb.get_interference_per_entity_path("people", base_folder="x")
        assert p == os.path.join("x", "interference_per_entity_people.json")

    def test_interference_per_pair_path(self) -> None:
        p = vb.get_interference_per_pair_path("people", 3, "distil", 400, base_folder="x")
        assert p == os.path.join(
            "x", "datasets", "interferences_caused_by_people_3_distil_400.json"
        )

    def test_metadata_filtered_path(self) -> None:
        p = vb.get_metadata_filtered_path("breeds", base_folder="x")
        assert p == os.path.join("x", "metadata_breeds_2_enriched_filtered.json")

    def test_generated_dataset_file(self) -> None:
        assert vb.get_generated_dataset_file("on", 4, "a cat") == "on_04_a cat.png"

    def test_interference_per_pair_inverse_respects_base_folder(
        self, tmp_path: "Any", monkeypatch: "Any"
    ) -> None:
        """get_interference_per_pair_inverse must use base_folder, not hardcode 'assets'."""
        import json as _json
        import vision_unlearning.benchmarks.I_care.metadata as _meta_mod

        # Stub metadata so we don't need real files on disk
        fake_meta = [{"name": "alice"}, {"name": "bob"}, {"name": "carol"}]
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: fake_meta)
        monkeypatch.setattr(_meta_mod, "get_metadata_filtered", lambda task, **k: fake_meta)

        # Write a fake interference file for emitter index 0 under tmp_path/datasets/
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        emitter_file = (
            datasets_dir / "interferences_caused_by_people_0_distil_400.json"
        )
        # bob (index=1) is our target; alice (index=0) is the emitter
        fake_data: Dict[str, float] = {"bob": 0.42, "carol": 0.11}
        emitter_file.write_text(_json.dumps(fake_data))

        result = vb.get_interference_per_pair_inverse(
            task="people",
            index=1,      # target = bob
            method="distil",
            num_train_epochs=400,
            index_start=0,
            max_identities=1,  # only check emitter index 0 (= alice)
            base_folder=str(tmp_path),
        )
        # alice (emitter 0) should appear in result
        assert "alice" in result, f"Expected 'alice' in result, got {result}"
        # Value should be bob's interference from alice's emitter file
        assert result["alice"] == pytest.approx(0.42)


class TestGetTargetOverwrite:
    def test_people(self) -> None:
        target, overwrite = vb.get_target_overwrite("people", "distil", "brad_pitt")
        assert overwrite == "a child"
        assert target == "brad pitt"

    def test_breeds_adds_article(self) -> None:
        target, overwrite = vb.get_target_overwrite("breeds", "uce", "poodle")
        assert overwrite == "a cat"
        assert target == "a poodle"

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            vb.get_target_overwrite("objects", "uce", "thing")  # type: ignore[arg-type]


class TestShapSerializationRoundtrip:
    def test_roundtrip_preserves_arrays(self) -> None:
        shap = pytest.importorskip("shap", reason="shap not installed; install vision-unlearning[testbed]")

        original = shap.Explanation(
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            base_values=np.array([0.5, 0.5]),
            data=np.array([[10.0, 20.0], [30.0, 40.0]]),
            feature_names=["f1", "f2"],
        )
        as_dict = vb.explanation_to_dict(original)
        # Must be JSON serializable (it is stored inside the result json).
        json.dumps(as_dict)
        restored = vb.dict_to_explanation(as_dict)
        np.testing.assert_array_equal(restored.values, original.values)
        np.testing.assert_array_equal(restored.base_values, original.base_values)
        np.testing.assert_array_equal(restored.data, original.data)
        assert list(restored.feature_names) == ["f1", "f2"]


# ---------------------------------------------------------------------------
# 3. Plot generation from fake data
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _close_figs() -> Any:
    yield
    plt.close("all")


class TestPlotFromFakeData:
    def test_metric_similarity_alignment_plot(self) -> None:
        data = {
            "metadata": {
                "task": "people",
                "unlearning_algorithm": "distil",
                "interference_pair": "rmse",
                "interference_pair_direction": "↓",
                "similarity_metric": "jacc",
                "similarity_metric_direction": "↑",
            },
            "result": {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.1, 1.9, 3.2, 3.8],
                "pearson_statistic": 0.99,
                "pearson_pvalue": 0.001,
                "spearman_statistic": 0.97,
                "spearman_pvalue": 0.002,
            },
        }
        fig, ax = vb.ResultTemplateMetricSimilarityAlignment.plot(data, return_fig=True)
        assert isinstance(fig, Figure)

    def test_significant_relationship_numerical_plot(self) -> None:
        data = {
            "metadata": {
                "task": "people",
                "unlearning_algorithm": "munba",
                "interference_entity": "Emitter average rmse",
                "interference_entity_direction": "↓",
                "attribute": "hpi_bin",
            },
            "result": {
                "x": [0.1, 0.5, 0.9, 1.2],
                "y": [2.0, 2.5, 3.0, 3.5],
                "pearson_statistic": 0.98,
                "pearson_pvalue": 0.01,
                "spearman_statistic": 0.95,
                "spearman_pvalue": 0.02,
            },
        }
        fig, ax = vb.ResultTemplateSignificantRelationshipNumerical.plot(
            data, return_fig=True
        )
        assert isinstance(fig, Figure)

    def test_significant_relationship_categorical_plot(self) -> None:
        data = {
            "metadata": {
                "task": "people",
                "unlearning_algorithm": "uce",
                "interference_entity": "Emitter average ssim",
                "interference_entity_direction": "↑",
                "attribute": "occupation_simplified",
            },
            "result": {
                "x": ["artist", "artist", "politician", "politician"],
                "y": [1.0, 1.2, 2.0, 2.4],
                "anova_pvalue": 0.03,
                "kruskal_pvalue": 0.04,
            },
        }
        fig, ax = vb.ResultTemplateSignificantRelationshipCategorical.plot(
            data, return_fig=True
        )
        assert isinstance(fig, Figure)

    def test_significant_relationship_categorical_directional_plot(self) -> None:
        data = {
            "metadata": {
                "task": "scenes",
                "unlearning_algorithm": "distil",
                "interference_pair": "clip_diff",
                "interference_pair_direction": "↑",
                "attribute": "sports",
                "source_attribute_value": "True",
            },
            "result": {
                "x": ["True", "True", "False", "False", "False"],
                "y": [-2.0, -2.5, -0.5, -0.3, -0.8],
                "anova_pvalue": 0.02,
                "kruskal_pvalue": 0.03,
            },
        }
        fig, ax = vb.ResultTemplateSignificantRelationshipCategoricalDirectional.plot(
            data, return_fig=True
        )
        assert isinstance(fig, Figure)

    def test_interference_matrix_plot(self) -> None:
        labels = ["a", "b", "c"]
        result = []
        for i, em in enumerate(labels):
            row: Dict[str, Any] = {"emitter": em}
            for j, rc in enumerate(labels):
                row[rc] = float(i + j)
            result.append(row)
        data = {
            "metadata": {
                "RT": "ResultTemplateInterferenceMatrix",
                "task": "people",
                "unlearning_algorithm": "distil",
                "interference_pair": "rmse",
                "_metric_key_name": "interference_pair",
                "metric_direction": "↓",
            },
            "result": result,
        }
        fig, ax = vb.ResultTemplateInterferenceMatrix.plot(data, return_fig=True)
        assert isinstance(fig, Figure)

    def test_similarity_matrix_plot(self) -> None:
        labels = ["a", "b"]
        result = [
            {"emitter": "a", "a": 1.0, "b": 0.4},
            {"emitter": "b", "a": 0.4, "b": 1.0},
        ]
        data = {
            "metadata": {
                "RT": "ResultTemplateSimilarityMatrix",
                "task": "people",
                "similarity_metric": "jacc",
                "_metric_key_name": "similarity_metric",
            },
            "result": result,
        }
        fig, ax = vb.ResultTemplateSimilarityMatrix.plot(data, return_fig=True)
        assert isinstance(fig, Figure)

    def test_metric_metric_alignment_plot(self) -> None:
        data = {
            "metadata": {
                "task": "people",
                "unlearning_algorithm": "distil",
                "interference_entity_1": "Forget clip diff",
                "interference_entity_2": "Retain average clip diff",
                "direction_1": "↓",
                "direction_2": "↑",
            },
            "result": {
                "x": [-14.0, -12.0, -10.0, -9.0, -13.5, -11.0],
                "y": [-0.5, -0.2, -0.1, 0.0, -0.3, -0.4],
                "entity_names": ["e0", "e1", "e2", "e3", "e4", "e5"],
                "pearson_statistic": -0.22,
                "pearson_pvalue": 0.03,
                "spearman_statistic": -0.21,
                "spearman_pvalue": 0.04,
            },
        }
        fig, ax = vb.ResultTemplateMetricMetricAlignment.plot(data, return_fig=True)
        assert isinstance(fig, Figure)

    def test_metric_metric_alignment_plot_multi_method(self) -> None:
        def _fake_data(method: str, xvals: List[float], yvals: List[float]) -> dict:
            return {
                "metadata": {
                    "task": "people",
                    "unlearning_algorithm": method,
                    "interference_entity_1": "Forget clip diff",
                    "interference_entity_2": "Retain average clip diff",
                    "direction_1": "↓",
                    "direction_2": "↑",
                },
                "result": {
                    "x": xvals,
                    "y": yvals,
                    "entity_names": [f"{method}_e{i}" for i in range(len(xvals))],
                    "pearson_statistic": 0.0,
                    "pearson_pvalue": 0.5,
                    "spearman_statistic": 0.0,
                    "spearman_pvalue": 0.5,
                },
            }

        method_data = {
            "distil": _fake_data("distil", [-13.0, -12.0, -14.0], [-0.2, -0.3, -0.1]),
            "munba": _fake_data("munba", [-6.0, -7.0, -5.0], [-5.5, -6.0, -5.0]),
            "uce": _fake_data("uce", [-0.5, -0.3, -0.7], [0.0, 0.1, -0.1]),
        }
        fig, ax = vb.ResultTemplateMetricMetricAlignment.plot_multi_method(
            method_data, return_fig=True
        )
        assert isinstance(fig, Figure)
        legend_labels = [h.get_label() for h in ax.get_legend().legend_handles]
        # Labels show publication display names (spare = distil, UCE = uce, Munba = munba)
        assert set(legend_labels) == {"spare", "UCE", "Munba"}

    def test_metric_similarity_alignment_multi_plot(self) -> None:
        # Build a tiny but real SHAP Explanation, serialize it the way the
        # RT stores it, and confirm the plot renders. This exercises the
        # Multi RT's plot (including dict_to_explanation) WITHOUT asserting
        # anything about the deferred methodological review items.
        shap = pytest.importorskip("shap", reason="shap not installed; install vision-unlearning[testbed]")

        expl = shap.Explanation(
            values=np.array([[0.1, -0.2], [0.3, 0.05]]),
            base_values=np.array([1.0, 1.0]),
            data=np.array([[0.5, 0.6], [0.7, 0.8]]),
            feature_names=["jacc", "clip"],
        )
        data = {
            "metadata": {"RT": "ResultTemplateMetricSimilarityAlignmentMulti"},
            "result": {
                "y_test_true": [1.0, 2.0, 3.0],
                "y_test_pred": [1.1, 1.9, 3.2],
                "r2_test": 0.95,
                "rmse_test": 0.15,
                "mae_test": 0.12,
                "shap_explanations": vb.explanation_to_dict(expl),
            },
        }
        fig, ax = vb.ResultTemplateMetricSimilarityAlignmentMulti.plot(
            data, return_fig=True
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# 4. compute_from_scratch with mocked file I/O and mocked Hugging Face
# ---------------------------------------------------------------------------
def _fake_metadata(labels: List[str]) -> List[Dict[str, Any]]:
    """Minimal metadata rows: one categorical + one 0-1 numeric attribute."""
    out = []
    for idx, name in enumerate(labels):
        out.append(
            {
                "name": name,
                "hpi_bin": "low" if idx % 2 == 0 else "high",
                "occupation_simplified": "artist" if idx % 2 == 0 else "politician",
            }
        )
    return out


@pytest.fixture
def no_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every HF existence check to False so compute() never hits the
    network and always takes the compute-from-scratch branch."""
    monkeypatch.setattr(vb, "huggingface_dataset_file_exists", lambda *a, **k: False)
    # Also patch at the cascade call site in vision_unlearning.artifact, where the storage
    # hooks look up the HuggingFace helpers.
    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_exists", lambda *a, **k: False)


class TestComputeFromScratchMocked:
    def test_interference_matrix_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        labels = ["a", "b", "c"]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        monkeypatch.setattr(_rt_mod, "get_metadata_filtered", lambda task, **k: meta)

        def fake_pair(task: str, index: int, method: str, epochs: int, **k: Any) -> Dict[str, Any]:
            return {lab: {"rmse": float(index + j)} for j, lab in enumerate(labels)}

        monkeypatch.setattr(vb, "get_interference_per_pair", fake_pair)
        monkeypatch.setattr(_rt_mod, "get_interference_per_pair", fake_pair)
        monkeypatch.setattr(
            vb, "get_interference_per_pair_path", lambda *a, **k: "exists"
        )
        monkeypatch.setattr(
            _rt_mod, "get_interference_per_pair_path", lambda *a, **k: "exists"
        )
        monkeypatch.setattr(os.path, "exists", lambda p: p == "exists")

        rt = vb.ResultTemplateInterferenceMatrix(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert data["metadata"]["RT"] == "ResultTemplateInterferenceMatrix"
        assert len(data["result"]) == len(labels)
        assert "emitter" in data["result"][0]

    def test_similarity_matrix_jacc_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        labels = ["a", "b", "c"]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        # Similarity binds get_metadata_filtered in similarity.py, so patch it there.
        monkeypatch.setattr(_sim_mod, "get_metadata_filtered", lambda task, **k: meta)

        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="jacc",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        # The Similarity artifact writes an incremental ``.partial`` checkpoint next to the
        # canonical file and creates its own parent directory; tmp_path is empty, so no stale
        # partial is read -> fresh compute.
        data = rt._compute_from_scratch()
        assert data["metadata"]["similarity_metric"] == "jacc"
        assert len(data["result"]) == len(labels)
        # Diagonal self-similarity must be 1.0.
        first = data["result"][0]
        assert first[first["emitter"]] == pytest.approx(1.0)

    def test_similarity_matrix_dino_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        """SimilarityMatrix dino branch reads embeddings from base_folder/datasets/."""
        import json as _json
        labels = ["alpha", "beta", "gamma"]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        # Similarity binds get_metadata_filtered in similarity.py, so patch it there.
        monkeypatch.setattr(_sim_mod, "get_metadata_filtered", lambda task, **k: meta)

        # Write a minimal fake embeddings file at the correct sub-path.
        # For task='people', distil epochs=400, the expected path is:
        #   base_folder/datasets/embeddings_people_original.json
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        emb_file = datasets_dir / "embeddings_people_original.json"
        rng = [1.0, 0.0, 0.0]
        # The code matches by 'prompt' field: "An image of {get_target_overwrite(...)[0]}"
        # For task='people', get_target_overwrite returns the name unchanged (spaces only).
        # labels have no underscores, so the prompt is just "An image of {label}".
        fake_embeddings = {
            "embeddings": [
                {"prompted_entity": e, "prompt": f"An image of {e}", "embedding": rng}
                for e in labels
            ]
        }
        emb_file.write_text(_json.dumps(fake_embeddings), encoding="utf-8")

        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="dino",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert data["metadata"]["similarity_metric"] == "dino"
        assert len(data["result"]) == len(labels)
        # All embeddings are identical unit vectors -> cosine sim == 1.0 for all pairs.
        first = data["result"][0]
        for label in labels:
            assert first[label] == pytest.approx(1.0), (
                f"Expected cosine sim 1.0 for pair ({first['emitter']}, {label})"
            )

    def test_metric_similarity_alignment_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        labels = ["a", "b", "c", "d"]

        def fake_im_compute(self: Any) -> Dict[str, Any]:
            res = []
            for i, em in enumerate(labels):
                row: Dict[str, Any] = {"emitter": em}
                for j, rc in enumerate(labels):
                    row[rc] = float(i * 2 + j)
                res.append(row)
            return {"metadata": {}, "result": res}

        def fake_sm_compute(self: Any) -> Dict[str, Any]:
            res = []
            for i, em in enumerate(labels):
                row: Dict[str, Any] = {"emitter": em}
                for j, rc in enumerate(labels):
                    row[rc] = float(i + j) * 0.5
                res.append(row)
            return {"metadata": {}, "result": res}

        monkeypatch.setattr(
            vb.ResultTemplateInterferenceMatrix, "compute", fake_im_compute
        )
        monkeypatch.setattr(
            vb.ResultTemplateSimilarityMatrix, "compute", fake_sm_compute
        )

        rt = vb.ResultTemplateMetricSimilarityAlignment(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric="jacc",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert "pearson_pvalue" in data["result"]
        assert "spearman_pvalue" in data["result"]
        assert len(data["result"]["x"]) == len(labels) * len(labels) - len(labels)

    def test_significant_relationship_numerical_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "name": [f"e{i}" for i in range(10)],
                "hpi_bin": [float(i) for i in range(10)],
                "metric_distil_400_emitter_average_rmse (↓)": [
                    float(i) * 1.5 for i in range(10)
                ],
            }
        )
        path = str(tmp_path / "interference_per_entity_people.json")
        df.to_json(path)
        monkeypatch.setattr(
            vb, "get_interference_per_entity_path", lambda task, **k: path
        )
        monkeypatch.setattr(
            _rt_mod, "get_interference_per_entity_path", lambda task, **k: path
        )

        rt = vb.ResultTemplateSignificantRelationshipNumerical(
            task="people",
            unlearning_algorithm="distil",
            interference_entity="Emitter average rmse",
            attribute="hpi_bin",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert "pearson_pvalue" in data["result"]
        assert len(data["result"]["x"]) == 10

    def test_significant_relationship_categorical_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "name": [f"e{i}" for i in range(12)],
                "occupation_simplified": ["artist"] * 6 + ["politician"] * 6,
                "metric_distil_400_emitter_average_rmse (↓)": [
                    float(i) for i in range(12)
                ],
            }
        )
        path = str(tmp_path / "interference_per_entity_people.json")
        df.to_json(path)
        monkeypatch.setattr(
            vb, "get_interference_per_entity_path", lambda task, **k: path
        )
        monkeypatch.setattr(
            _rt_mod, "get_interference_per_entity_path", lambda task, **k: path
        )

        rt = vb.ResultTemplateSignificantRelationshipCategorical(
            task="people",
            unlearning_algorithm="distil",
            interference_entity="Emitter average rmse",
            attribute="occupation_simplified",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert "anova_pvalue" in data["result"]
        assert "kruskal_pvalue" in data["result"]

    def test_significant_relationship_categorical_directional_compute(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        # 12 entities: 6 "sport", 6 "nonSport". Source = "sport".
        # Each source emitter causes a clip_diff that is more negative for sport receivers.
        labels = [f"e{i}" for i in range(12)]
        meta = [
            {"name": name, "sports": "sport" if i < 6 else "nonSport"}
            for i, name in enumerate(labels)
        ]
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        monkeypatch.setattr(_rt_mod, "get_metadata_filtered", lambda task, **k: meta)

        # Per-pair files: sport emitters cause more negative clip_diff to sport receivers.
        def fake_pair(task: str, index: int, method: str, epochs: int, **k: Any) -> Dict[str, Any]:
            # Source is entity index 0..5 (sport). Sport receivers (0..5) get -3.0, others -1.0.
            result = {}
            for j, lab in enumerate(labels):
                val = -3.0 if j < 6 else -1.0
                result[lab] = {"clip_diff": val}
            return result

        monkeypatch.setattr(vb, "get_interference_per_pair", fake_pair)
        monkeypatch.setattr(_rt_mod, "get_interference_per_pair", fake_pair)
        # Make all source emitter paths appear to exist.
        monkeypatch.setattr(vb, "get_interference_per_pair_path", lambda *a, **k: __file__)
        monkeypatch.setattr(_rt_mod, "get_interference_per_pair_path", lambda *a, **k: __file__)

        rt = vb.ResultTemplateSignificantRelationshipCategoricalDirectional(
            task="scenes",
            unlearning_algorithm="distil",
            interference_pair="clip_diff",
            attribute="sports",
            source_attribute_value="sport",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()

        assert "anova_pvalue" in data["result"]
        assert "kruskal_pvalue" in data["result"]
        assert "significant" in data["result"]
        # With the fake data, sport receivers should have lower (more negative) mean.
        sport_means = [data["result"]["y"][i] for i, x in enumerate(data["result"]["x"]) if x == "sport"]
        nonsport_means = [data["result"]["y"][i] for i, x in enumerate(data["result"]["x"]) if x == "nonSport"]
        assert len(sport_means) == 6
        assert len(nonsport_means) == 6
        # Sport receivers get the -3.0 value, non-sport get -1.0
        # (diagonal entries where receiver==emitter get -1.0 for self, but since exclude_diagonal=True,
        # the self-pair is excluded — a sport receiver accumulates contributions from the other 5 sport emitters)
        assert all(m < -1.5 for m in sport_means), f"Expected sport means < -1.5, got {sport_means}"
        assert all(m > -2.0 for m in nonsport_means), f"Expected non-sport means > -2.0, got {nonsport_means}"

    def test_metric_similarity_alignment_multi_compute_runs(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        # End-to-end run of the regression/SHAP RT on mocked data. We only
        # assert the pipeline produces a well-formed result; methodological
        # review items (entity-leaking split, MAPE, feature strip) are
        # deferred and intentionally NOT tested here.
        pytest.importorskip("shap", reason="shap not installed; install vision-unlearning[testbed]")
        labels = [f"p{i}" for i in range(8)]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        monkeypatch.setattr(_rt_mod, "get_metadata_filtered", lambda task, **k: meta)

        def fake_im_compute(self: Any) -> Dict[str, Any]:
            res = []
            rng = np.random.default_rng(0)
            for em in labels:
                row: Dict[str, Any] = {"emitter": em}
                for rc in labels:
                    row[rc] = float(rng.uniform(0, 5))
                res.append(row)
            return {"metadata": {}, "result": res}

        def fake_sm_compute(self: Any) -> Dict[str, Any]:
            res = []
            rng = np.random.default_rng(1)
            for em in labels:
                row: Dict[str, Any] = {"emitter": em}
                for rc in labels:
                    row[rc] = float(rng.uniform(0, 1))
                res.append(row)
            return {"metadata": {}, "result": res}

        monkeypatch.setattr(
            vb.ResultTemplateInterferenceMatrix, "compute", fake_im_compute
        )
        monkeypatch.setattr(
            vb.ResultTemplateSimilarityMatrix, "compute", fake_sm_compute
        )

        rt = vb.ResultTemplateMetricSimilarityAlignmentMulti(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric_list=["jacc", "clip"],
            include_attribute_diff_similarity=True,
            include_attribute_value_similarity=False,
            include_emitter_forget_quality=False,  # Me file not mocked in this test
            regression_algorithm="linear_regression",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert "r2_train" in data["result"]
        assert "r2_test" in data["result"]
        assert "mae_train" in data["result"]
        assert "mae_test" in data["result"]
        assert "shap_explanations" in data["result"]
        # The serialized SHAP must roundtrip back into an Explanation.
        restored = vb.dict_to_explanation(data["result"]["shap_explanations"])
        assert restored.values is not None

    def test_metric_similarity_alignment_multi_serialize_includes_forget_quality_flag(
        self,
    ) -> None:
        """_serialize_parameters must embed the include_emitter_forget_quality flag
        so that cached results with different flag values do not collide."""
        rt_with = vb.ResultTemplateMetricSimilarityAlignmentMulti(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric_list=["clip"],
            include_emitter_forget_quality=True,
            save_outputs=False,
        )
        rt_without = vb.ResultTemplateMetricSimilarityAlignmentMulti(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric_list=["clip"],
            include_emitter_forget_quality=False,
            save_outputs=False,
        )
        s_with = rt_with._serialize_parameters()
        s_without = rt_without._serialize_parameters()
        assert s_with != s_without, (
            "_serialize_parameters must differ for include_emitter_forget_quality=True vs False"
        )
        assert "_1_" in s_with or s_with.endswith("_1"), (
            "include_emitter_forget_quality=True must embed '1' in serialization"
        )
        assert "_0_" in s_without or s_without.endswith("_0"), (
            "include_emitter_forget_quality=False must embed '0' in serialization"
        )

    def test_metric_similarity_alignment_multi_emitter_forget_quality_feature(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        """When include_emitter_forget_quality=True, the emitter_forget_clip_diff
        column must appear in the training data and mae_test must be in result."""
        import pandas as pd

        pytest.importorskip("shap", reason="shap not installed")
        labels = [f"p{i}" for i in range(8)]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        monkeypatch.setattr(_rt_mod, "get_metadata_filtered", lambda task, **k: meta)

        def fake_im_compute(self: Any) -> Dict[str, Any]:
            rng = np.random.default_rng(0)
            return {
                "metadata": {},
                "result": [
                    {"emitter": em, **{rc: float(rng.uniform(0, 5)) for rc in labels}}
                    for em in labels
                ],
            }

        def fake_sm_compute(self: Any) -> Dict[str, Any]:
            rng = np.random.default_rng(1)
            return {
                "metadata": {},
                "result": [
                    {"emitter": em, **{rc: float(rng.uniform(0, 1)) for rc in labels}}
                    for em in labels
                ],
            }

        monkeypatch.setattr(vb.ResultTemplateInterferenceMatrix, "compute", fake_im_compute)
        monkeypatch.setattr(vb.ResultTemplateSimilarityMatrix, "compute", fake_sm_compute)

        # Build a fake Me file with forget_clip_diff for each entity
        df_me = pd.DataFrame({
            "name": labels,
            "metric_distil_400_forget_clip_diff (↓)": [float(-i) for i in range(len(labels))],
        })
        me_path = str(tmp_path / "interference_per_entity_people.json")
        df_me.to_json(me_path)
        monkeypatch.setattr(vb, "get_interference_per_entity_path", lambda task, **k: me_path)
        monkeypatch.setattr(_rt_mod, "get_interference_per_entity_path", lambda task, **k: me_path)

        rt = vb.ResultTemplateMetricSimilarityAlignmentMulti(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric_list=["clip"],
            include_attribute_diff_similarity=False,
            include_attribute_value_similarity=False,
            include_emitter_forget_quality=True,
            regression_algorithm="linear_regression",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()

        # emitter_forget_clip_diff must appear in the feature list
        # Note: ColumnTransformer prefixes numeric features with 'num__'
        assert any("emitter_forget_clip_diff" in f for f in data["result"]["features"]), (
            "emitter_forget_clip_diff must be a feature when include_emitter_forget_quality=True. "
            f"Got features: {data['result']['features']}"
        )
        # mae metrics must be present
        assert "mae_train" in data["result"]
        assert "mae_test" in data["result"]
        # metadata must record the flag
        assert data["metadata"]["include_emitter_forget_quality"] is True

    def test_metric_similarity_alignment_multi_missing_me_file_raises(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        """When include_emitter_forget_quality=True and Me file is absent, a
        clear FileNotFoundError must be raised — no silent NaN fallback."""
        labels = [f"p{i}" for i in range(4)]
        meta = _fake_metadata(labels)
        monkeypatch.setattr(vb, "get_metadata_filtered", lambda task, **k: meta)
        monkeypatch.setattr(_rt_mod, "get_metadata_filtered", lambda task, **k: meta)

        def fake_im_compute(self: Any) -> Dict[str, Any]:
            rng = np.random.default_rng(0)
            return {
                "metadata": {},
                "result": [
                    {"emitter": em, **{rc: float(rng.uniform(0, 5)) for rc in labels}}
                    for em in labels
                ],
            }

        def fake_sm_compute(self: Any) -> Dict[str, Any]:
            rng = np.random.default_rng(1)
            return {
                "metadata": {},
                "result": [
                    {"emitter": em, **{rc: float(rng.uniform(0, 1)) for rc in labels}}
                    for em in labels
                ],
            }

        monkeypatch.setattr(vb.ResultTemplateInterferenceMatrix, "compute", fake_im_compute)
        monkeypatch.setattr(vb.ResultTemplateSimilarityMatrix, "compute", fake_sm_compute)

        # Me file does NOT exist
        missing_path = str(tmp_path / "interference_per_entity_people.json")
        monkeypatch.setattr(vb, "get_interference_per_entity_path", lambda task, **k: missing_path)
        monkeypatch.setattr(_rt_mod, "get_interference_per_entity_path", lambda task, **k: missing_path)

        rt = vb.ResultTemplateMetricSimilarityAlignmentMulti(
            task="people",
            unlearning_algorithm="distil",
            interference_pair="rmse",
            similarity_metric_list=["clip"],
            include_emitter_forget_quality=True,
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match="Me file not found"):
            rt._compute_from_scratch()


# ---------------------------------------------------------------------------
# MetricMetricAlignment
# ---------------------------------------------------------------------------

class TestMetricMetricAlignment:
    """Unit tests for ResultTemplateMetricMetricAlignment."""

    def test_serialize_parameters(self) -> None:
        rt = vb.ResultTemplateMetricMetricAlignment(
            model="sd1.4",
            task="people",
            unlearning_algorithm="distil",
            interference_entity_1="Forget clip diff",
            interference_entity_2="Retain average clip diff",
            save_outputs=False,
        )
        s = rt._serialize_parameters()
        assert s == "sd1.4_people_distil_forget_clip_diff_retain_average_clip_diff"

    def test_compute_from_scratch_correlates(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        import pandas as pd

        # Use perfectly correlated values to get r=1
        df = pd.DataFrame(
            {
                "name": [f"e{i}" for i in range(10)],
                "metric_distil_400_forget_clip_diff (↓)": [
                    float(-i) for i in range(10)
                ],
                "metric_distil_400_retain_average_clip_diff (↑)": [
                    float(-i) * 0.1 for i in range(10)
                ],
            }
        )
        path = str(tmp_path / "interference_per_entity_people.json")
        df.to_json(path)
        monkeypatch.setattr(
            vb, "get_interference_per_entity_path", lambda task, **k: path
        )
        monkeypatch.setattr(
            _rt_mod, "get_interference_per_entity_path", lambda task, **k: path
        )

        rt = vb.ResultTemplateMetricMetricAlignment(
            task="people",
            unlearning_algorithm="distil",
            interference_entity_1="Forget clip diff",
            interference_entity_2="Retain average clip diff",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        result = data["result"]
        assert "pearson_statistic" in result
        assert "spearman_statistic" in result
        assert "entity_names" in result
        assert len(result["entity_names"]) == 10
        assert abs(result["pearson_statistic"] - 1.0) < 1e-6, (
            "Perfectly correlated inputs must give Pearson r=1"
        )
        meta = data["metadata"]
        assert meta["col1"].startswith("metric_distil")
        assert meta["col2"].startswith("metric_distil")
        assert meta["direction_1"] == "↓"
        assert meta["direction_2"] == "↑"

    def test_compute_drops_nan_rows(
        self, monkeypatch: pytest.MonkeyPatch, no_remote: None, tmp_path: Any
    ) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "name": [f"e{i}" for i in range(5)],
                "metric_distil_400_forget_clip_diff (↓)": [
                    -1.0, -2.0, None, -4.0, -5.0
                ],
                "metric_distil_400_retain_average_clip_diff (↑)": [
                    -0.1, -0.2, -0.3, None, -0.5
                ],
            }
        )
        path = str(tmp_path / "interference_per_entity_people.json")
        df.to_json(path)
        monkeypatch.setattr(
            _rt_mod, "get_interference_per_entity_path", lambda task, **k: path
        )

        rt = vb.ResultTemplateMetricMetricAlignment(
            task="people",
            unlearning_algorithm="distil",
            interference_entity_1="Forget clip diff",
            interference_entity_2="Retain average clip diff",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        # Rows with NaN in either column are dropped: 3 remain (e0, e1, e4)
        assert len(data["result"]["x"]) == 3
        assert len(data["result"]["entity_names"]) == 3


# ---------------------------------------------------------------------------
# upload_if_recomputed — ResultTemplate base class
# ---------------------------------------------------------------------------

class _FakeRT(vb.ResultTemplate):
    """Minimal concrete ResultTemplate for testing upload_if_recomputed."""

    def _serialize_parameters(self) -> str:
        return "fake_params"

    def _compute_from_scratch(self) -> dict:  # type: ignore[override]
        return {"metadata": {"RT": "Fake"}, "result": {"value": 42}}


class TestResultTemplateUploadIfRecomputed:
    """upload_if_recomputed=True triggers HF file upload after compute-from-scratch."""

    def test_default_is_false(self) -> None:
        rt = _FakeRT()
        assert rt.upload_if_recomputed is False

    def test_can_be_set_true(self) -> None:
        rt = _FakeRT(upload_if_recomputed=True)
        assert rt.upload_if_recomputed is True

    def test_upload_called_after_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True: huggingface_dataset_file_upload is called."""
        rt = _FakeRT(base_folder=str(tmp_path), upload_if_recomputed=True)
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )

        upload_calls: List[Any] = []

        def fake_upload(**kw: Any) -> None:
            upload_calls.append(kw)

        monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_upload", fake_upload)

        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            data = rt.compute()

        assert data["result"]["value"] == 42
        assert len(upload_calls) == 1, "upload must be called exactly once"
        assert upload_calls[0]["dataset_path"] == rt._get_data_path_remote()
        assert upload_calls[0]["dataset_repository"] == rt.remote_repository_name

    def test_upload_not_called_when_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=False (default): no upload even after scratch."""
        rt = _FakeRT(base_folder=str(tmp_path), upload_if_recomputed=False)
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )

        upload_calls: List[Any] = []
        monkeypatch.setattr(
            _artifact_mod,
            "huggingface_dataset_file_upload",
            lambda **kw: upload_calls.append(kw),
        )

        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            rt.compute()

        assert len(upload_calls) == 0, "upload must NOT be called when upload_if_recomputed=False"

    def test_upload_requires_save_outputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True with save_outputs=False raises AssertionError."""
        rt = _FakeRT(
            base_folder=str(tmp_path),
            upload_if_recomputed=True,
            save_outputs=False,
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            with pytest.raises(AssertionError, match="save_outputs"):
                rt.compute()

    def test_upload_requires_hf_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True without HF_TOKEN raises AssertionError."""
        rt = _FakeRT(base_folder=str(tmp_path), upload_if_recomputed=True)
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(AssertionError, match="HF_TOKEN"):
            rt.compute()


# ---------------------------------------------------------------------------
# Remote path and download token — ResultTemplate base class
# ---------------------------------------------------------------------------

class TestResultTemplateRemotePathAndToken:
    """Regression tests for two unauthenticated/cross-platform download bugs."""

    def test_remote_path_uses_forward_slashes(self) -> None:
        """The remote path is a HuggingFace repository path, never an OS filesystem
        path: on Windows an os.path.join-built path contains backslashes, HF rejects
        it, and compute() silently falls through to _compute_from_scratch."""
        assert _FakeRT()._get_data_path_remote() == "results/_FakeRT/fake_params.json"
        rt = vb.ResultTemplateInterferenceMatrix(
            model="sd1.4", task="people", unlearning_algorithm="uce", interference_pair="clip_diff"
        )
        assert rt._get_data_path_remote() == "results/InterferenceMatrix/sd1.4_people_uce_clip_diff.json"

    def test_download_token_is_none_when_hf_token_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Without HF_TOKEN, the download must receive token=None, not token="":
        an empty string becomes an illegal 'Authorization: Bearer ' header and breaks
        unauthenticated downloads from public repositories."""
        rt = _FakeRT(base_folder=str(tmp_path))
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: True
        )

        download_calls: List[Any] = []

        def fake_download(**kw: Any) -> None:
            download_calls.append(kw)
            local_path = rt._get_data_path_local()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump({"metadata": {"RT": "Fake"}, "result": {"value": 42}}, f)

        monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_download", fake_download)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        data = rt.compute()

        assert data["result"]["value"] == 42
        assert len(download_calls) == 1
        assert download_calls[0]["token"] is None


# ---------------------------------------------------------------------------
# Example of a GPU-marked test (none of the above need a GPU). This single
# placeholder documents the marker convention and is skipped on CI.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_gpu_marker_is_registered() -> None:
    """Sanity test for the gpu marker plumbing.

    It does no GPU work itself; it only proves the marker exists and is
    excluded by the default ``-m 'not gpu'`` selector.
    """
    assert True
