"""Tests for Items 1, 2, 3 of the I-CARE benchmark extensions.

These tests cover:
  - Item 1: S_dino — DINOv2 as a new similarity metric (type_s, s_to_direction,
    GUI_TO_BACKEND, and ResultTemplateSimilarityMatrix._compute_from_scratch dino branch)
  - Item 2: me_dino — Embedding specificity ratio as a new type_me, and
    choose_metric_column_interference_per_entity matching the new column name
  - Item 3: ResultTemplateMethodComparisonByMetricEntity — registry, compute, plot

All tests are CPU-only and require no GPU, no network, and no real data files.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402
import vision_unlearning.benchmarks.I_care.metadata as _meta_mod  # noqa: E402
import vision_unlearning.benchmarks.I_care.result_templates as _rt_mod  # noqa: E402
import vision_unlearning.artifact as _artifact_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs() -> Any:
    yield
    plt.close("all")


def _make_embedding_file(entities: List[str], dim: int = 8) -> dict:
    """Build a minimal embeddings JSON structure for testing.

    Prompt format matches the real files: ``"An image of {entity}"``.
    The code uses the ``prompt`` field (not ``prompted_entity``) to identify
    embeddings, so this must be consistent with what ``get_target_overwrite``
    produces for the given task/entity combination.
    """
    rng = np.random.default_rng(42)
    embeddings = []
    for entity in entities:
        for seed in range(3):
            vec = rng.standard_normal(dim).tolist()
            embeddings.append({
                "prompted_entity": entity,
                "seed": seed,
                "prompt": f"An image of {entity}",
                "embedding": vec,
            })
    return {"embeddings": embeddings}


def _write_embedding_file(path: str, entities: List[str], dim: int = 8) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_embedding_file(entities, dim), f)


# ---------------------------------------------------------------------------
# Item 1 — S_dino: DINOv2 similarity metric
# ---------------------------------------------------------------------------

class TestTypeSContainsDino:
    def test_dino_in_type_s(self) -> None:
        assert "dino" in vb.type_s.__args__

    def test_s_to_direction_dino(self) -> None:
        assert vb.s_to_direction["dino"] == "↑"

    def test_gui_to_backend_dino(self) -> None:
        assert vb.GUI_TO_BACKEND["similarity_metric"]["DINOv2 Cosine Similarity"] == "dino"


class TestSimilarityMatrixDinoCompute:
    def test_dino_smoke(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """3-entity embedding file → matrix is symmetric with diagonal ≈ 1."""
        entities = ["Alice", "Bob", "Carol"]
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in entities],
        )
        # Patch at the actual call site in result_templates (where the function is bound)
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in entities],
        )
        # Write a baseline embedding file in tmp_path/datasets/ (correct sub-path).
        emb_path = str(tmp_path / "datasets" / "embeddings_people_original_distil_400.json")
        _write_embedding_file(emb_path, entities)

        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="dino",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert data["metadata"]["similarity_metric"] == "dino"
        assert len(data["result"]) == len(entities)
        # Diagonal: each entity vs itself → cosine similarity = 1
        for row in data["result"]:
            assert row[row["emitter"]] == pytest.approx(1.0, abs=1e-5)

    def test_dino_known_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Two identical mean embeddings → similarity 1; orthogonal → 0."""
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        # Construct embeddings manually: Alpha has e0 direction, Beta has e1 direction.
        # Names must be >=3 chars (get_target_overwrite assertion).
        # For task='people', get_target_overwrite('people','distil','Alpha')[0] == 'Alpha',
        # so the expected prompt is "An image of Alpha".
        dim = 4
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        entities = ["Alpha", "Beta"]
        embedding_data = {
            "embeddings": [
                {"prompted_entity": "Alpha", "seed": 0, "prompt": "An image of Alpha", "embedding": vec_a},
                {"prompted_entity": "Beta", "seed": 0, "prompt": "An image of Beta", "embedding": vec_b},
            ]
        }
        # Write embedding file in the correct sub-path (datasets/).
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        emb_path = str(datasets_dir / "embeddings_people_original_distil_400.json")
        with open(emb_path, "w") as f:
            json.dump(embedding_data, f)

        monkeypatch.setattr(
            vb, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in entities],
        )
        # Patch at the actual call site in result_templates (where the function is bound)
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in entities],
        )
        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="dino",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        # Build lookup: result is list of dicts with 'emitter' key
        result = {row["emitter"]: row for row in data["result"]}
        assert result["Alpha"]["Alpha"] == pytest.approx(1.0, abs=1e-5)
        assert result["Beta"]["Beta"] == pytest.approx(1.0, abs=1e-5)
        assert result["Alpha"]["Beta"] == pytest.approx(0.0, abs=1e-5)
        assert result["Beta"]["Alpha"] == pytest.approx(0.0, abs=1e-5)

    def test_dino_missing_file_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """AssertionError when baseline embedding file is absent."""
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb, "get_metadata_filtered",
            lambda task, **kw: [{"name": "Alice"}],
        )
        # Patch at the actual call site in result_templates (where the function is bound)
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": "Alice"}],
        )
        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="dino",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        with pytest.raises(AssertionError, match="Baseline DINOv2 embeddings not found"):
            rt._compute_from_scratch()


# ---------------------------------------------------------------------------
# Item 2 — me_dino: Embedding specificity ratio as a new type_me
# ---------------------------------------------------------------------------

class TestTypeMeContainsEmbeddingSpecificityRatio:
    def test_embedding_specificity_ratio_in_type_me(self) -> None:
        assert "Embedding specificity ratio" in vb.type_me.__args__

    def test_embedding_specificity_ratio_in_domain_me(self) -> None:
        assert "Embedding specificity ratio" in vb.domain_me

    def test_choose_metric_column_embedding_specificity_ratio(self) -> None:
        cols = [
            "metric_distil_400_embedding_specificity_ratio (↑)",
            "metric_uce_000_embedding_specificity_ratio (↑)",
            "metric_distil_400_emitter_average_rmse (↓)",
        ]
        col = vb.choose_metric_column_interference_per_entity(
            "distil", "Embedding specificity ratio", cols
        )
        assert col == "metric_distil_400_embedding_specificity_ratio (↑)"

    def test_choose_metric_column_uce_embedding_specificity_ratio(self) -> None:
        cols = [
            "metric_distil_400_embedding_specificity_ratio (↑)",
            "metric_uce_000_embedding_specificity_ratio (↑)",
        ]
        col = vb.choose_metric_column_interference_per_entity(
            "uce", "Embedding specificity ratio", cols
        )
        assert col == "metric_uce_000_embedding_specificity_ratio (↑)"


class TestSpecificityRatioFormula:
    """Unit-test the ratio formula in isolation (no file I/O)."""

    def _compute_ratio(
        self,
        cos_distances: Dict[str, float],
        target_hf_name: str,
    ) -> float:
        """Replicate the ratio logic from the script."""
        if target_hf_name not in cos_distances:
            return float('nan')
        d_self = cos_distances[target_hf_name]
        d_others = float(np.mean([v for k, v in cos_distances.items() if k != target_hf_name]))
        return float(d_self / d_others) if d_others > 0 else float('nan')

    def test_ratio_above_one(self) -> None:
        # d_self = 0.5, d_others = 0.25 → ratio = 2.0
        distances = {"Alice": 0.5, "Bob": 0.25, "Carol": 0.25}
        assert self._compute_ratio(distances, "Alice") == pytest.approx(2.0)

    def test_ratio_below_one(self) -> None:
        # d_self = 0.1, d_others mean = 0.3 → ratio ≈ 0.333
        distances = {"Alice": 0.1, "Bob": 0.3, "Carol": 0.3}
        r = self._compute_ratio(distances, "Alice")
        assert r == pytest.approx(0.1 / 0.3, rel=1e-5)

    def test_ratio_nan_when_target_missing(self) -> None:
        distances = {"Bob": 0.3, "Carol": 0.4}
        import math
        assert math.isnan(self._compute_ratio(distances, "Missing"))


# ---------------------------------------------------------------------------
# Item 3 — ResultTemplateMethodComparisonByMetricEntity
# ---------------------------------------------------------------------------

class TestMethodSpecificityRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "MethodComparisonByMetricEntity" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["MethodComparisonByMetricEntity"] is vb.ResultTemplateMethodComparisonByMetricEntity

    def test_params_in_registry(self) -> None:
        expected = ["model", "task", "interference_entity", "unlearning_algorithm_list"]
        assert vb.rt_name_to_params["MethodComparisonByMetricEntity"] == expected

    def test_registry_consistent(self) -> None:
        assert set(vb.rt_name_to_class) == set(vb.rt_name_to_params)


class TestMethodSpecificityCompute:
    def _fake_interference_data(
        self, labels: List[str], distil_vals: List[float], uce_vals: List[float]
    ) -> List[Dict[str, Any]]:
        rows = []
        for i, name in enumerate(labels):
            row: Dict[str, Any] = {"name": name}
            row["metric_distil_400_embedding_specificity_ratio (↑)"] = distil_vals[i]
            row["metric_uce_000_embedding_specificity_ratio (↑)"] = uce_vals[i]
            rows.append(row)
        return rows

    def test_smoke(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """compute() returns expected keys with mocked InterferencePerEntity."""
        labels = ["Alice", "Bob", "Carol", "Dave"]
        distil_vals = [1.8, 1.9, 1.7, 1.85]
        uce_vals = [1.2, 1.4, 1.0, 1.3]
        fake_data = self._fake_interference_data(labels, distil_vals, uce_vals)

        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb.InterferencePerEntity, "compute", lambda self: fake_data
        )

        rt = vb.ResultTemplateMethodComparisonByMetricEntity(
            task="people",
            interference_entity="Embedding specificity ratio",
            unlearning_algorithm_list=["distil", "uce"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()

        assert "result" in data
        assert "metadata" in data
        assert data["metadata"]["RT"] == "ResultTemplateMethodComparisonByMetricEntity"
        assert data["metadata"]["interference_entity"] == "Embedding specificity ratio"

    def test_summary_keys_per_algo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Each algorithm entry must have values, mean, median, std, n."""
        labels = ["Alice", "Bob"]
        fake_data = self._fake_interference_data(labels, [1.8, 1.9], [1.2, 1.3])

        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb.InterferencePerEntity, "compute", lambda self: fake_data
        )

        rt = vb.ResultTemplateMethodComparisonByMetricEntity(
            task="people",
            interference_entity="Embedding specificity ratio",
            unlearning_algorithm_list=["distil", "uce"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        result = data["result"]

        for algo in ["distil", "uce"]:
            assert algo in result
            for key in ("values", "mean", "median", "std", "n"):
                assert key in result[algo], f"Missing key '{key}' for algo '{algo}'"
            assert isinstance(result[algo]["values"], list)
            assert result[algo]["n"] == len(labels)

    def test_missing_column_skips_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """When a column is absent, that algo is skipped — no crash."""
        labels = ["Alice", "Bob"]
        # Only distil column present; uce is absent
        fake_data = [
            {"name": "Alice", "metric_distil_400_embedding_specificity_ratio (↑)": 1.8},
            {"name": "Bob", "metric_distil_400_embedding_specificity_ratio (↑)": 1.9},
        ]

        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb.InterferencePerEntity, "compute", lambda self: fake_data
        )

        rt = vb.ResultTemplateMethodComparisonByMetricEntity(
            task="people",
            interference_entity="Embedding specificity ratio",
            unlearning_algorithm_list=["distil", "uce"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        result = data["result"]

        assert "distil" in result
        assert "uce" not in result  # silently skipped

    def test_mean_values_correct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Mean and median are numerically correct for known inputs."""
        labels = ["Alice", "Bob", "Carol"]
        distil_vals = [1.0, 2.0, 3.0]
        uce_vals = [0.5, 0.5, 0.5]
        fake_data = self._fake_interference_data(labels, distil_vals, uce_vals)

        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            vb.InterferencePerEntity, "compute", lambda self: fake_data
        )

        rt = vb.ResultTemplateMethodComparisonByMetricEntity(
            task="people",
            interference_entity="Embedding specificity ratio",
            unlearning_algorithm_list=["distil", "uce"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        result = data["result"]

        assert result["distil"]["mean"] == pytest.approx(2.0)
        assert result["distil"]["median"] == pytest.approx(2.0)
        assert result["uce"]["mean"] == pytest.approx(0.5)


class TestMethodSpecificityPlot:
    def test_plot_returns_fig(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """plot(..., return_fig=True) returns a (Figure, Axes) tuple."""
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )

        data = {
            "metadata": {
                "RT": "ResultTemplateMethodComparisonByMetricEntity",
                "model": "sd1.4",
                "task": "people",
                "interference_entity": "Embedding specificity ratio",
                "unlearning_algorithm_list": ["distil", "uce"],
                "direction": "↑",
            },
            "result": {
                "distil": {"values": [1.8, 1.9, 1.7], "mean": 1.8, "median": 1.8, "std": 0.1, "n": 3},
                "uce": {"values": [1.2, 1.3, 1.1], "mean": 1.2, "median": 1.2, "std": 0.1, "n": 3},
            },
        }
        result = vb.ResultTemplateMethodComparisonByMetricEntity.plot(data, return_fig=True)
        assert result is not None
        fig, ax = result
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# ResultTemplateInterferenceVisualSummary — off-image path selection
# ---------------------------------------------------------------------------

class TestInterferenceVisualSummaryOffImagePath:
    """Verify that _compute_from_scratch uses get_off_image_path for 'off' images
    and get_generated_dataset_folder for 'on' images."""

    def test_off_images_use_get_off_image_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """get_off_image_path must be called for the 'off' state."""
        import vision_unlearning.benchmarks.I_care.result_templates as _rt

        # Track calls to off-image path function vs entity-folder function
        off_image_calls: List[str] = []
        entity_folder_calls: List[str] = []

        # Fake entities — two people suffice: the target and one other
        fake_metadata = [
            {"name": "Person A", "type": "politician"},
            {"name": "Person B", "type": "politician"},
            {"name": "Person C", "type": "politician"},
            {"name": "Person D", "type": "politician"},
            {"name": "Person E", "type": "politician"},
            {"name": "Person F", "type": "politician"},
            {"name": "Person G", "type": "politician"},
            {"name": "Person H", "type": "politician"},
            {"name": "Person I", "type": "politician"},
        ]

        fake_interference: Dict[str, Dict[str, float]] = {
            m["name"]: {"ssim": float(i)} for i, m in enumerate(fake_metadata)
        }

        def fake_get_off_image_path(
            task: str, target: str, method: str, num_train_epochs: int,
            seed: int, prompt: str, base_folder: str = "assets",
        ) -> str:
            off_image_calls.append(prompt)
            return os.path.join(tmp_path, "baseline", f"off_{seed}_{target}.png")

        def fake_get_generated_dataset_folder(
            task: str, method: str, num_train_epochs: int, target: str,
            base_folder: str = "assets",
        ) -> str:
            entity_folder_calls.append(target)
            return str(tmp_path / "entity")

        def fake_encode_image_file(path: str, max_dim: int = 128) -> str:
            return "base64fake"

        def fake_get_interference_per_pair(
            task: str, entity_index: int, method: str, num_train_epochs: int,
        ) -> Dict[str, Dict[str, float]]:
            return {m["name"]: {"ssim": float(i)} for i, m in enumerate(fake_metadata)}

        monkeypatch.setattr(_rt, "get_metadata_filtered", lambda *a, **kw: fake_metadata)
        monkeypatch.setattr(_rt, "get_target_overwrite", lambda task, method, name: (name, name))
        monkeypatch.setattr(_rt, "get_off_image_path", fake_get_off_image_path)
        monkeypatch.setattr(_rt, "get_generated_dataset_folder", fake_get_generated_dataset_folder)
        monkeypatch.setattr(_rt, "_encode_image_file", fake_encode_image_file)
        monkeypatch.setattr(_rt, "get_interference_per_pair",
                            lambda *a, **kw: fake_get_interference_per_pair(*a, **kw))
        monkeypatch.setattr(_rt, "unlearning_algorithm_to_epochs", {"people": {"distil": 400}})
        monkeypatch.setattr(_rt, "mp_to_direction", {"ssim": "↑"})
        monkeypatch.setattr(_rt, "get_generated_dataset_file",
                            lambda state, seed, prompt: f"{state}_{seed}.png")

        rt = _rt.ResultTemplateInterferenceVisualSummary(
            unlearning_algorithm="distil",
            interference_pair="ssim",
            entity="Person A",
        )
        result = rt._compute_from_scratch()

        # off-image path must have been resolved via get_off_image_path
        assert len(off_image_calls) > 0, "get_off_image_path was never called for 'off' images"
        # 'on' images must use the entity folder (get_generated_dataset_folder)
        assert len(entity_folder_calls) > 0, "get_generated_dataset_folder was never called for 'on' images"
        # Both image states must appear in the result
        assert "off" in result["result"]["images"]
        assert "on" in result["result"]["images"]

    def test_on_images_use_entity_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """get_generated_dataset_folder is called at least once for 'on' images."""
        import vision_unlearning.benchmarks.I_care.result_templates as _rt

        entity_folder_calls: List[str] = []

        fake_metadata = [
            {"name": f"Person {chr(65+i)}", "type": "politician"} for i in range(9)
        ]

        def fake_get_off_image_path(*args: Any, **kwargs: Any) -> str:
            return str(tmp_path / "off.png")

        def fake_get_generated_dataset_folder(*args: Any, **kwargs: Any) -> str:
            entity_folder_calls.append("called")
            return str(tmp_path / "entity")

        monkeypatch.setattr(_rt, "get_metadata_filtered", lambda *a, **kw: fake_metadata)
        monkeypatch.setattr(_rt, "get_target_overwrite", lambda task, method, name: (name, name))
        monkeypatch.setattr(_rt, "get_off_image_path", fake_get_off_image_path)
        monkeypatch.setattr(_rt, "get_generated_dataset_folder", fake_get_generated_dataset_folder)
        monkeypatch.setattr(_rt, "_encode_image_file", lambda *a, **kw: "base64fake")
        monkeypatch.setattr(_rt, "get_interference_per_pair",
                            lambda *a, **kw: {m["name"]: {"ssim": float(i)} for i, m in enumerate(fake_metadata)})
        monkeypatch.setattr(_rt, "unlearning_algorithm_to_epochs", {"people": {"distil": 400}})
        monkeypatch.setattr(_rt, "mp_to_direction", {"ssim": "↑"})
        monkeypatch.setattr(_rt, "get_generated_dataset_file",
                            lambda state, seed, prompt: f"{state}_{seed}.png")

        rt = _rt.ResultTemplateInterferenceVisualSummary(
            unlearning_algorithm="distil",
            interference_pair="ssim",
            entity="Person A",
        )
        rt._compute_from_scratch()

        assert len(entity_folder_calls) > 0, "get_generated_dataset_folder was never called for 'on' images"


# ---------------------------------------------------------------------------
# upload_if_recomputed — InterferencePerEntity
# ---------------------------------------------------------------------------

_FAKE_IPE_DATA = [
    {
        "name": "Alice",
        "metric_distil_400_emitter_minus_receiver_average_clip_diff (↑)": 0.5,
    },
    {
        "name": "Bob",
        "metric_distil_400_emitter_minus_receiver_average_clip_diff (↑)": 0.3,
    },
]


class TestInterferencePerEntityUploadIfRecomputed:
    """upload_if_recomputed=True triggers HF upload after compute-from-scratch."""

    def test_default_is_false(self) -> None:
        """upload_if_recomputed defaults to False."""
        ipe = vb.InterferencePerEntity(task='people')
        assert ipe.upload_if_recomputed is False

    def test_can_be_set_true(self) -> None:
        ipe = vb.InterferencePerEntity(task='people', upload_if_recomputed=True)
        assert ipe.upload_if_recomputed is True

    def test_upload_called_after_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True: huggingface_dataset_file_upload is called."""
        ipe = vb.InterferencePerEntity(
            task='people',
            base_folder=str(tmp_path),
            upload_if_recomputed=True,
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(ipe, "_compute_from_scratch", lambda: list(_FAKE_IPE_DATA))

        upload_calls: List[Any] = []

        def fake_upload(**kw: Any) -> None:
            upload_calls.append(kw)

        monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_upload", fake_upload)

        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            result = ipe.compute()

        assert len(result) == 2
        assert len(upload_calls) == 1, "upload must be called exactly once"
        assert upload_calls[0]["dataset_path"] == ipe._get_data_path_remote()
        assert upload_calls[0]["dataset_repository"] == ipe.remote_repository_name

    def test_upload_not_called_when_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=False (default): no upload even after scratch."""
        ipe = vb.InterferencePerEntity(
            task='people',
            base_folder=str(tmp_path),
            upload_if_recomputed=False,
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(ipe, "_compute_from_scratch", lambda: list(_FAKE_IPE_DATA))

        upload_calls: List[Any] = []
        monkeypatch.setattr(
            _artifact_mod,
            "huggingface_dataset_file_upload",
            lambda **kw: upload_calls.append(kw),
        )

        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            ipe.compute()

        assert len(upload_calls) == 0, "upload must NOT be called when upload_if_recomputed=False"

    def test_upload_requires_save_outputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True with save_outputs=False raises AssertionError."""
        ipe = vb.InterferencePerEntity(
            task='people',
            base_folder=str(tmp_path),
            upload_if_recomputed=True,
            save_outputs=False,
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(ipe, "_compute_from_scratch", lambda: list(_FAKE_IPE_DATA))
        with patch.dict("os.environ", {"HF_TOKEN": "fake_token"}):
            with pytest.raises(AssertionError, match="save_outputs"):
                ipe.compute()

    def test_upload_requires_hf_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """upload_if_recomputed=True without HF_TOKEN raises AssertionError."""
        ipe = vb.InterferencePerEntity(
            task='people',
            base_folder=str(tmp_path),
            upload_if_recomputed=True,
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(ipe, "_compute_from_scratch", lambda: list(_FAKE_IPE_DATA))
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(AssertionError, match="HF_TOKEN"):
            ipe.compute()

    def test_download_token_is_none_when_hf_token_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Without HF_TOKEN, the download must receive token=None, not token="":
        an empty string becomes an illegal 'Authorization: Bearer ' header and breaks
        unauthenticated downloads from public repositories."""
        ipe = vb.InterferencePerEntity(task='people', base_folder=str(tmp_path))
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: True
        )

        download_calls: List[Any] = []

        def fake_download(**kw: Any) -> None:
            download_calls.append(kw)
            local_path = ipe._get_data_path_local()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(list(_FAKE_IPE_DATA), f)

        monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_download", fake_download)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        result = ipe.compute()

        assert len(result) == len(_FAKE_IPE_DATA)
        assert len(download_calls) == 1
        assert download_calls[0]["token"] is None


# ---------------------------------------------------------------------------
# ResultTemplateMetricSimilarityAlignmentOne (MSAOne)
# Single-emitter slice of MetricSimilarityAlignment.
# ---------------------------------------------------------------------------

def _records_from_matrix(labels: List[str], values: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Build the list-of-records form returned by the matrix RTs."""
    records: List[Dict[str, Any]] = []
    for emitter in labels:
        row: Dict[str, Any] = {"emitter": emitter}
        for receiver in labels:
            row[receiver] = values[emitter][receiver]
        records.append(row)
    return records


# 4 receivers around emitter "Ada"; other rows are filler (only row Ada is read).
# Names must be >=3 chars (get_target_overwrite assertion, exercised by plot()).
_MSAONE_LABELS = ["Ada", "Bob", "Cleo", "Dora", "Evan"]
_FILLER = {a: {b: 0.0 for b in _MSAONE_LABELS} for a in _MSAONE_LABELS}
_INTERF_VALUES = {**_FILLER, "Ada": {"Ada": 0.0, "Bob": 0.1, "Cleo": 0.4, "Dora": 0.2, "Evan": 0.3}}
_SIM_VALUES = {**_FILLER, "Ada": {"Ada": 1.0, "Bob": 0.9, "Cleo": 0.95, "Dora": 0.5, "Evan": 0.6}}


def _patch_msaone_matrices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _rt_mod, "get_metadata_filtered",
        lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateInterferenceMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_MSAONE_LABELS, _INTERF_VALUES)},
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateSimilarityMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_MSAONE_LABELS, _SIM_VALUES)},
    )


class TestMSAOneRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "MetricSimilarityAlignmentOne" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["MetricSimilarityAlignmentOne"] is \
            _rt_mod.ResultTemplateMetricSimilarityAlignmentOne


class TestMSAOneResolveEntity:
    def test_entity_to_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
        )
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Cleo",
        )
        rt._resolve_entity()
        assert rt.entity == "Cleo"
        assert rt.entity_index == 2

    def test_index_to_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
        )
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity_index=3,
        )
        rt._resolve_entity()
        assert rt.entity == "Dora"
        assert rt.entity_index == 3

    def test_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
        )
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", entity_index=1,
        )
        with pytest.raises(ValueError, match="does not match"):
            rt._resolve_entity()

    def test_neither_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
        )
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino",
        )
        with pytest.raises(ValueError, match="Either entity or entity_index"):
            rt._resolve_entity()


class TestMSAOneCompute:
    def test_basic_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        r = data["result"]
        # 4 receivers (B, C, D, E), emitter A excluded
        assert len(r["x"]) == 4
        assert len(r["y"]) == 4
        assert set(r["receiver_names"]) == {"Bob", "Cleo", "Dora", "Evan"}
        # x = interference row A in receiver order; y = similarity row A
        order = r["receiver_names"]
        assert dict(zip(order, r["x"])) == {"Bob": 0.1, "Cleo": 0.4, "Dora": 0.2, "Evan": 0.3}
        assert dict(zip(order, r["y"])) == {"Bob": 0.9, "Cleo": 0.95, "Dora": 0.5, "Evan": 0.6}

    def test_most_least_direction_worst_biggest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rmse: direction '↓' → worst = biggest. Most interfered = C(0.4), E(0.3)."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        r = rt._compute_from_scratch()["result"]
        assert r["is_worst_biggest"] is True
        assert r["labeled_most"] == ["Cleo", "Evan"]
        # least interfered, least-first: B(0.1), D(0.2)
        assert r["labeled_least"] == ["Bob", "Dora"]

    def test_most_least_direction_worst_smallest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clip_diff: direction '↑' → worst = smallest. Most interfered = B(0.1), D(0.2)."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        r = rt._compute_from_scratch()["result"]
        assert r["is_worst_biggest"] is False
        assert r["labeled_most"] == ["Bob", "Dora"]
        assert r["labeled_least"] == ["Cleo", "Evan"]

    def test_serialize_includes_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: [{"name": e} for e in _MSAONE_LABELS],
        )
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            model="sd1.4", task="people", unlearning_algorithm="uce",
            interference_pair="rmse", similarity_metric="dino", entity="Cleo",
        )
        assert rt._serialize_parameters() == "sd1.4_people_uce_rmse_dino_Cleo"


class TestMSAOnePlot:
    def test_plot_returns_fig(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        out = rt.plot(data, return_fig=True)
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# MSAOne — labeled_most_similar / labeled_least_similar (new fields)
# ---------------------------------------------------------------------------

class TestMSAOneSimilarityLabels:
    """Verify that _compute_from_scratch populates the similarity label fields."""

    def test_similarity_labels_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """labeled_most_similar and labeled_least_similar must be in result."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        r = rt._compute_from_scratch()["result"]
        assert "labeled_most_similar" in r
        assert "labeled_least_similar" in r

    def test_similarity_labels_correct_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        _SIM_VALUES for Ada row: Bob=0.9, Cleo=0.95, Dora=0.5, Evan=0.6.
        Most similar (top-2): Cleo(0.95), Bob(0.9).
        Least similar (bottom-2): Dora(0.5), Evan(0.6) — reversed from worst-first.
        """
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        r = rt._compute_from_scratch()["result"]
        assert r["labeled_most_similar"] == ["Cleo", "Bob"]
        # least-similar: bottom-2 from sorted-by-sim-descending = Dora(0.5), Evan(0.6)
        # reversed → Evan, Dora (but [::-1] on [-2:] of descending list)
        # sorted descending: Cleo=0.95, Bob=0.9, Evan=0.6, Dora=0.5
        # [-2:] = [Evan, Dora]; [::-1] = [Dora, Evan]
        assert r["labeled_least_similar"] == ["Dora", "Evan"]

    def test_similarity_labels_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each similarity label list must have exactly display_name_top_n entries."""
        _patch_msaone_matrices(monkeypatch)
        for n in (1, 2):
            rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
                unlearning_algorithm="uce", interference_pair="rmse",
                similarity_metric="dino", entity="Ada", display_name_top_n=n,
                save_outputs=False,
            )
            r = rt._compute_from_scratch()["result"]
            assert len(r["labeled_most_similar"])  == n
            assert len(r["labeled_least_similar"]) == n


# ---------------------------------------------------------------------------
# MSAOne plot — colour encodes interference only; most/least-similar are named in grey.
# A larger 7-receiver fixture is used so that the most/least-similar sets are
# distinct from the most/least-interfered sets (the small 4-receiver fixture
# above cannot separate them when display_name_top_n=2).
#
# Ada row (emitter Ada, 7 receivers Bob..Hugo), interference_pair=clip_diff
# (direction '↑' -> worst = SMALLEST), similarity = dino (higher = more similar):
#
#   receiver  interference(x)  similarity(y)  expected label colour
#   Bob       0.0              0.30           crimson   (most-interfered)
#   Cleo      0.1              0.20           crimson   (most-interfered; also
#                                                        least-similar -> interference wins)
#   Dora      0.5              0.95           grey      (most-similar only)
#   Evan      0.6              0.90           grey      (most-similar only)
#   Finn      0.7              0.10           grey      (least-similar only)
#   Gwen      0.9              0.40           seagreen  (least-interfered)
#   Hugo      1.0              0.50           seagreen  (least-interfered)
# ---------------------------------------------------------------------------

_MSAONE7_LABELS = ["Ada", "Bob", "Cleo", "Dora", "Evan", "Finn", "Gwen", "Hugo"]
_FILLER7 = {a: {b: 0.0 for b in _MSAONE7_LABELS} for a in _MSAONE7_LABELS}
_INTERF7 = {**_FILLER7, "Ada": {
    "Ada": 0.0, "Bob": 0.0, "Cleo": 0.1, "Dora": 0.5,
    "Evan": 0.6, "Finn": 0.7, "Gwen": 0.9, "Hugo": 1.0,
}}
_SIM7 = {**_FILLER7, "Ada": {
    "Ada": 1.0, "Bob": 0.3, "Cleo": 0.2, "Dora": 0.95,
    "Evan": 0.9, "Finn": 0.1, "Gwen": 0.4, "Hugo": 0.5,
}}


def _patch_msaone7(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _rt_mod, "get_metadata_filtered",
        lambda task, **kw: [{"name": e} for e in _MSAONE7_LABELS],
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateInterferenceMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_MSAONE7_LABELS, _INTERF7)},
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateSimilarityMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_MSAONE7_LABELS, _SIM7)},
    )


class TestMSAOnePlotColouring:
    """The plot must encode interference by colour and name the similar entities in grey."""

    @staticmethod
    def _plot(monkeypatch: pytest.MonkeyPatch) -> Any:
        _patch_msaone7(monkeypatch)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        out = rt.plot(data, return_fig=True)
        assert out is not None
        return out

    def test_label_colour_by_position(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Every annotation must carry the colour dictated by the interference groups, with the
        most/least-similar (but not interfered) receivers named in grey. The annotation xy
        equals the receiver's (interference, similarity) pair, so we map position -> colour.
        """
        fig, ax = self._plot(monkeypatch)
        expected = {
            (0.0, 0.30): "crimson",   # Bob  — most-interfered
            (0.1, 0.20): "crimson",   # Cleo — most-interfered (priority over least-similar)
            (0.5, 0.95): "grey",      # Dora — most-similar only
            (0.6, 0.90): "grey",      # Evan — most-similar only
            (0.7, 0.10): "grey",      # Finn — least-similar only
            (0.9, 0.40): "seagreen",  # Gwen — least-interfered
            (1.0, 0.50): "seagreen",  # Hugo — least-interfered
        }
        got = {
            (round(float(t.xy[0]), 3), round(float(t.xy[1]), 3)): t.get_color()
            for t in ax.texts
        }
        assert got == expected
        plt.close(fig)

    def test_colour_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two crimson, two seagreen, three grey labels — and no other label colours."""
        fig, ax = self._plot(monkeypatch)
        counts: Dict[str, int] = {}
        for t in ax.texts:
            counts[t.get_color()] = counts.get(t.get_color(), 0) + 1
        assert counts == {"crimson": 2, "seagreen": 2, "grey": 3}
        plt.close(fig)

    def test_legend_lists_only_interference_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Legend must list ONLY the two coloured interference groups. The most-/least-similar
        receivers have no distinct visual encoding (grey), so they must not appear in the legend.
        """
        fig, ax = self._plot(monkeypatch)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["2 most-interfered", "2 least-interfered"]
        plt.close(fig)


# ---------------------------------------------------------------------------
# ResultTemplateInterferenceBySimilarityRank — single-session interference-vs-similarity-rank.
# Reuses the MSAOne 4-receiver fixture (emitter Ada). Ada similarity row:
#   Bob=0.9, Cleo=0.95, Dora=0.5, Evan=0.6  -> ranked by similarity descending:
#   rank 1 Cleo(0.95), rank 2 Bob(0.90), rank 3 Evan(0.60), rank 4 Dora(0.50)
# Ada interference row: Bob=0.1, Cleo=0.4, Dora=0.2, Evan=0.3
# ---------------------------------------------------------------------------

class TestInterferenceBySimilarityRankRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "InterferenceBySimilarityRank" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["InterferenceBySimilarityRank"] is \
            _rt_mod.ResultTemplateInterferenceBySimilarityRank

    def test_in_rt_name_to_params(self) -> None:
        assert "InterferenceBySimilarityRank" in vb.rt_name_to_params
        assert vb.rt_name_to_params["InterferenceBySimilarityRank"] == [
            "model", "task", "unlearning_algorithm", "interference_pair",
            "similarity_metric", "entity",
        ]


class TestInterferenceBySimilarityRankCompute:
    def _rt(self) -> Any:
        return _rt_mod.ResultTemplateInterferenceBySimilarityRank(
            unlearning_algorithm="uce", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )

    def test_one_point_per_receiver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        r = self._rt()._compute_from_scratch()["result"]
        # 4 receivers (emitter Ada excluded), each at a unique rank 1..4
        assert r["rank"] == [1, 2, 3, 4]
        assert len(r["interference"]) == 4
        assert len(r["receiver_names"]) == 4

    def test_ranked_by_similarity_descending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        r = self._rt()._compute_from_scratch()["result"]
        assert r["receiver_names"] == ["Cleo", "Bob", "Evan", "Dora"]
        # similarity must be non-increasing along the rank axis
        assert r["similarity"] == sorted(r["similarity"], reverse=True)
        # interference is reported in the same (similarity-ranked) order
        assert r["interference"] == [0.4, 0.1, 0.3, 0.2]

    def test_most_least_interfered_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clip_diff direction '↑' -> worst = smallest. Ada interference: Bob=0.1, Dora=0.2, Evan=0.3, Cleo=0.4."""
        _patch_msaone_matrices(monkeypatch)
        r = self._rt()._compute_from_scratch()["result"]
        assert r["labeled_most"] == ["Bob", "Dora"]
        assert r["labeled_least"] == ["Cleo", "Evan"]

    def test_spearman_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        r = self._rt()._compute_from_scratch()["result"]
        assert "spearman_statistic" in r
        assert "spearman_pvalue" in r


class TestInterferenceBySimilarityRankPlot:
    def test_plot_returns_fig(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateInterferenceBySimilarityRank(
            unlearning_algorithm="uce", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        out = rt.plot(rt._compute_from_scratch(), return_fig=True)
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)
        # x axis = similarity rank, y axis = interference metric with direction arrow
        assert "Similarity rank" in ax.get_xlabel()
        assert "clip_diff" in ax.get_ylabel()
        plt.close(fig)

    def test_legend_names_most_and_least_interfered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legend must name the most- and least-interfered receivers (2 + 2 with top_n=2)."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateInterferenceBySimilarityRank(
            unlearning_algorithm="uce", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        fig, ax = rt.plot(rt._compute_from_scratch(), return_fig=True)
        legend = ax.get_legend()
        labels = {t.get_text() for t in legend.get_texts()}
        # most-interfered Bob, Dora ; least-interfered Cleo, Evan (names passed through _short_entity_display)
        assert {"Bob", "Dora", "Cleo", "Evan"} <= labels
        plt.close(fig)

    def test_title_uses_display_method_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Internal method name 'distil' must be shown as its display name 'SPARE'."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateInterferenceBySimilarityRank(
            unlearning_algorithm="distil", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        fig, ax = rt.plot(rt._compute_from_scratch(), return_fig=True)
        title = ax.get_title()
        assert "SPARE" in title
        assert "distil" not in title
        plt.close(fig)


# ---------------------------------------------------------------------------
# ResultTemplateMostSimilarMostInterferedGrid — count grid across sessions.
# Tiny 3-entity fixture (A, B, C), one task / method / interference_pair / similarity_metric.
#
# Similarity (dino):            Interference (clip_diff, direction '↑' -> worst = SMALLEST):
#   A: B=0.9 C=0.1                A: B=-5.0 C=-1.0   -> most-interfered = B
#   B: A=0.9 C=0.5                B: A=-1.0 C=-3.0   -> most-interfered = C
#   C: A=0.1 B=0.5                C: A=-2.0 B=-1.0   -> most-interfered = A
#
# Most-similar receiver per emitter: A->B, B->A, C->B.
# top-1 matches (most-similar == most-interfered): only emitter A (B==B). count = 1 of 3.
# ---------------------------------------------------------------------------

_GRID_LABELS = ["A", "B", "C"]
_GRID_INTERF = {
    "A": {"A": 0.0, "B": -5.0, "C": -1.0},
    "B": {"A": -1.0, "B": 0.0, "C": -3.0},
    "C": {"A": -2.0, "B": -1.0, "C": 0.0},
}
_GRID_SIM = {
    "A": {"A": 1.0, "B": 0.9, "C": 0.1},
    "B": {"A": 0.9, "B": 1.0, "C": 0.5},
    "C": {"A": 0.1, "B": 0.5, "C": 1.0},
}


def _patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _rt_mod, "get_metadata_filtered",
        lambda task, **kw: [{"name": e} for e in _GRID_LABELS],
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateInterferenceMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_GRID_LABELS, _GRID_INTERF)},
    )
    monkeypatch.setattr(
        _rt_mod.ResultTemplateSimilarityMatrix, "compute",
        lambda self: {"result": _records_from_matrix(_GRID_LABELS, _GRID_SIM)},
    )


class TestMostSimilarMostInterferedGridRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "MostSimilarMostInterferedGrid" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["MostSimilarMostInterferedGrid"] is \
            _rt_mod.ResultTemplateMostSimilarMostInterferedGrid

    def test_in_rt_name_to_params(self) -> None:
        assert vb.rt_name_to_params["MostSimilarMostInterferedGrid"] == [
            "model", "tasks", "unlearning_algorithms",
            "interference_pairs", "similarity_metrics", "top_k",
        ]


class TestMostSimilarMostInterferedGridCompute:
    @staticmethod
    def _rt(top_k: int, tasks: List[str], methods: List[str]) -> Any:
        return _rt_mod.ResultTemplateMostSimilarMostInterferedGrid(
            tasks=tasks, unlearning_algorithms=methods,
            interference_pairs=["clip_diff"], similarity_metrics=["dino"],
            top_k=top_k, save_outputs=False,
        )

    def test_top1_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_grid(monkeypatch)
        r = self._rt(1, ["people"], ["uce"])._compute_from_scratch()["result"]
        assert r["counts"] == [[1]]
        assert r["denominators"] == [[3]]

    def test_topk_covers_all_receivers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With top_k == number of receivers, the most-similar is always among the top-k."""
        _patch_grid(monkeypatch)
        r = self._rt(2, ["people"], ["uce"])._compute_from_scratch()["result"]
        assert r["counts"] == [[3]]
        assert r["denominators"] == [[3]]

    def test_grid_shape_methods_by_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_grid(monkeypatch)
        r = self._rt(1, ["people", "breeds"], ["uce", "distil"])._compute_from_scratch()["result"]
        # rows = methods, cols = tasks
        assert len(r["counts"]) == 2
        assert all(len(row) == 2 for row in r["counts"])
        assert r["max_per_cell"] == 3 * 1 * 1  # 3 entities x 1 mp x 1 sim

    def test_direction_handling_worst_biggest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """For an '↓' metric (rmse), worst = BIGGEST; the count must flip accordingly."""
        _patch_grid(monkeypatch)
        rt = _rt_mod.ResultTemplateMostSimilarMostInterferedGrid(
            tasks=["people"], unlearning_algorithms=["uce"],
            interference_pairs=["rmse"], similarity_metrics=["dino"],
            top_k=1, save_outputs=False,
        )
        # rmse direction '↓' -> worst = biggest value.
        # A receivers B=-5,C=-1 -> biggest=-1=C; most-similar=B -> no match
        # B receivers A=-1,C=-3 -> biggest=-1=A; most-similar=A -> match
        # C receivers A=-2,B=-1 -> biggest=-1=B; most-similar=B -> match
        r = rt._compute_from_scratch()["result"]
        assert r["counts"] == [[2]]


class TestMostSimilarMostInterferedGridPlot:
    def test_plot_returns_fig(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_grid(monkeypatch)
        rt = _rt_mod.ResultTemplateMostSimilarMostInterferedGrid(
            tasks=["people", "breeds"], unlearning_algorithms=["uce", "distil"],
            interference_pairs=["clip_diff"], similarity_metrics=["dino"],
            top_k=1, save_outputs=False,
        )
        out = rt.plot(rt._compute_from_scratch(), return_fig=True)
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)
        assert ax.get_xlabel() == "Task"
        assert ax.get_ylabel() == "Unlearning method"
        # y-tick labels use display method names (uce -> UCE, distil -> SPARE)
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert ytick_labels == ["UCE", "SPARE"]
        # dense parameter-listing title: RT name first line, then comma-separated params (no braces,
        # no max-per-cell, no discursive explanation)
        title = ax.get_title()
        assert title.startswith("Result Template: MostSimilarInterferedMatrix\n")
        assert "top_k=1" in title
        assert "interference_pairs=clip_diff" in title
        assert "similarity_metrics=dino" in title
        assert "{" not in title and "}" not in title
        assert "max per cell" not in title
        assert "nominal maximum" not in title
        plt.close(fig)


# ---------------------------------------------------------------------------
# ResultTemplateSimilarityVisualSummary — registry + compute + plot
# ---------------------------------------------------------------------------

_SVS_LABELS = [f"Entity{i}" for i in range(9)]  # 9 entities; entity0 is the emitter
_SVS_SIM: Dict[str, Dict[str, float]] = {
    a: {b: (0.9 - 0.1 * abs(int(a[-1]) - int(b[-1]))) for b in _SVS_LABELS}
    for a in _SVS_LABELS
}


def _patch_svs(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Patch all external dependencies for SimilarityVisualSummary tests."""
    import vision_unlearning.benchmarks.I_care.result_templates as _rt

    fake_metadata = [{"name": n} for n in _SVS_LABELS]
    monkeypatch.setattr(_rt, "get_metadata_filtered", lambda *a, **kw: fake_metadata)
    monkeypatch.setattr(_rt, "get_target_overwrite", lambda task, method, name: (name, name))
    monkeypatch.setattr(_rt, "unlearning_algorithm_to_epochs", {"people": {"distil": 400}})
    monkeypatch.setattr(_rt, "s_to_direction", {"dino": "↑", "jacc": "↑", "clip": "↑", "act": "↑"})
    monkeypatch.setattr(
        _rt.ResultTemplateSimilarityMatrix,
        "compute",
        lambda self: {
            "result": [
                {"emitter": a, **{b: _SVS_SIM[a][b] for b in _SVS_LABELS}}
                for a in _SVS_LABELS
            ]
        },
    )
    monkeypatch.setattr(_rt, "_encode_image_file", lambda *a, **kw: "base64fake")
    monkeypatch.setattr(
        _rt, "get_off_image_path", lambda *a, **kw: str(tmp_path / "off.png")
    )
    monkeypatch.setattr(
        _rt, "get_generated_dataset_folder", lambda *a, **kw: str(tmp_path / "on")
    )
    monkeypatch.setattr(
        _rt, "get_generated_dataset_file", lambda state, seed, prompt: f"{state}_{seed}.png"
    )

    # _decode_image and plt.imread must not crash with our fake base64
    monkeypatch.setattr(
        _rt, "_decode_image",
        lambda b64: __import__('io').BytesIO(__import__('base64').b64decode(
            __import__('base64').b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        )),
    )


def _make_tiny_png_b64() -> str:
    """Return a minimal 1×1 white PNG as base64 string for image mocking."""
    import base64
    import struct
    import zlib

    def png_chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


class TestSimilarityVisualSummaryRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "SimilarityVisualSummary" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["SimilarityVisualSummary"] is \
            _rt_mod.ResultTemplateSimilarityVisualSummary

    def test_in_rt_name_to_params(self) -> None:
        assert "SimilarityVisualSummary" in vb.rt_name_to_params

    def test_params_include_similarity_metric(self) -> None:
        assert "similarity_metric" in vb.rt_name_to_params["SimilarityVisualSummary"]


class TestSimilarityVisualSummaryCompute:
    def test_basic_result_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """_compute_from_scratch returns mandatory result keys."""
        _patch_svs(monkeypatch, tmp_path)
        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        r = data["result"]
        assert "displayed_entities" in r
        assert "most_similar" in r
        assert "least_similar" in r
        assert "similarity_values" in r
        assert "images" in r

    def test_displayed_entities_length(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """displayed_entities must have 9 entries: target + 4 most + 4 least."""
        _patch_svs(monkeypatch, tmp_path)
        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert len(data["result"]["displayed_entities"]) == 9

    def test_target_is_first_entity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _patch_svs(monkeypatch, tmp_path)
        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert data["result"]["displayed_entities"][0] == "Entity0"

    def test_most_and_least_do_not_overlap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _patch_svs(monkeypatch, tmp_path)
        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        r = data["result"]
        assert len(r["most_similar"]) == 4
        assert len(r["least_similar"]) == 4
        assert set(r["most_similar"]).isdisjoint(set(r["least_similar"]))

    def test_images_have_on_off_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _patch_svs(monkeypatch, tmp_path)
        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert "off" in data["result"]["images"]
        assert "on"  in data["result"]["images"]


class TestSimilarityVisualSummaryPlot:
    def test_plot_returns_fig(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _patch_svs(monkeypatch, tmp_path)
        import vision_unlearning.benchmarks.I_care.result_templates as _rt
        tiny_b64 = _make_tiny_png_b64()
        monkeypatch.setattr(_rt, "_encode_image_file", lambda *a, **kw: tiny_b64)

        import io, base64
        monkeypatch.setattr(
            _rt, "_decode_image",
            lambda b64: io.BytesIO(base64.b64decode(b64)),
        )

        rt = _rt_mod.ResultTemplateSimilarityVisualSummary(
            task="people", unlearning_algorithm="distil",
            similarity_metric="dino", entity="Entity0", save_outputs=False,
        )
        data = rt._compute_from_scratch()
        out = rt.plot(data, return_fig=True)
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# CountSignificantRelationship (CSR)
# ---------------------------------------------------------------------------

def _make_sr_result(significant: bool) -> dict:
    """Minimal SRC/SRN compute() result for mocking."""
    return {
        "metadata": {
            "RT": "ResultTemplateSignificantRelationshipCategorical",
            "model": "sd1.4",
            "task": "people",
            "unlearning_algorithm": "distil",
            "interference_entity": "Emitter average clip diff",
            "attribute": "hpi_bin",
            "interference_entity_direction": "↑",
            "chosen_metric_col": "metric_distil_400_emitter_average_clip_diff (↑)",
            "significance_threshold": 0.05,
        },
        "result": {
            "x": ["low", "high"] * 5,
            "y": [0.1, 0.2] * 5,
            "anova_statistic": 3.5,
            "anova_pvalue": 0.02 if significant else 0.5,
            "kruskal_statistic": 2.5,
            "kruskal_pvalue": 0.03 if significant else 0.6,
            "significant": significant,
        },
    }


class TestCountSignificantRelationshipRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "CountSignificantRelationship" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert (
            vb.rt_name_to_class["CountSignificantRelationship"]
            is _rt_mod.ResultTemplateCountSignificantRelationship
        )

    def test_registry_consistent(self) -> None:
        assert set(vb.rt_name_to_class) == set(vb.rt_name_to_params)


class TestCountSignificantRelationshipCompute:
    def _setup_mocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        significant_map: Dict[str, bool],
    ) -> None:
        """Mock SRC.compute() to return significant=True/False per interference_entity."""
        def _fake_compute(self_inner: Any) -> dict:
            sig = significant_map.get(self_inner.interference_entity, False)
            return _make_sr_result(sig)

        monkeypatch.setattr(
            _rt_mod.ResultTemplateSignificantRelationshipCategorical,
            "compute",
            _fake_compute,
        )
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )

    def test_grouped_dicts_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """_compute_from_scratch populates all three grouped_by_* dicts."""
        self._setup_mocks(
            monkeypatch,
            {"Emitter average clip diff": True, "Emitter average rmse": False},
        )

        rt = _rt_mod.ResultTemplateCountSignificantRelationship(
            task="people",
            unlearning_algorithm_list=["distil"],
            interference_entity_list=["Emitter average clip diff", "Emitter average rmse"],
            attribute_list=["hpi_bin"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        r = data["result"]

        assert "grouped_by_unlearning_algorithm" in r
        assert "grouped_by_attribute" in r
        assert "grouped_by_interference_entity" in r

        by_algo = r["grouped_by_unlearning_algorithm"]
        assert "distil" in by_algo
        assert by_algo["distil"]["total"] == 2  # 1 algo × 2 me × 1 attr
        assert by_algo["distil"]["count"] == 1  # only clip_diff is sig

    def test_fraction_correct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """fraction = count / total for each group key."""
        self._setup_mocks(
            monkeypatch,
            {"Emitter average clip diff": True, "Emitter average rmse": True},
        )

        rt = _rt_mod.ResultTemplateCountSignificantRelationship(
            task="people",
            unlearning_algorithm_list=["distil"],
            interference_entity_list=["Emitter average clip diff", "Emitter average rmse"],
            attribute_list=["hpi_bin"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        r = data["result"]

        by_algo = r["grouped_by_unlearning_algorithm"]
        assert by_algo["distil"]["count"] == 2
        assert by_algo["distil"]["total"] == 2
        assert by_algo["distil"]["fraction"] == pytest.approx(1.0)

    def test_rows_field_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """result['rows'] contains one record per successful (algo, me, attr) triple."""
        self._setup_mocks(monkeypatch, {"Emitter average clip diff": False})

        rt = _rt_mod.ResultTemplateCountSignificantRelationship(
            task="people",
            unlearning_algorithm_list=["distil", "uce"],
            interference_entity_list=["Emitter average clip diff"],
            attribute_list=["hpi_bin"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        rows = data["result"]["rows"]
        assert len(rows) == 2  # 2 algos × 1 me × 1 attr

    def test_exception_in_src_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Combinations that raise InsufficientSamplesError are silently skipped."""
        def _raise(_self: Any) -> dict:
            raise vb.InsufficientSamplesError("not enough")

        monkeypatch.setattr(
            _rt_mod.ResultTemplateSignificantRelationshipCategorical, "compute", _raise
        )
        monkeypatch.setattr(
            _rt_mod.ResultTemplateSignificantRelationshipNumerical, "compute", _raise
        )
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )

        rt = _rt_mod.ResultTemplateCountSignificantRelationship(
            task="people",
            unlearning_algorithm_list=["distil"],
            interference_entity_list=["Emitter average clip diff"],
            attribute_list=["hpi_bin"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()
        assert data["result"]["total"] == 0
        assert data["result"]["total_count"] == 0
        assert data["result"]["grouped_by_unlearning_algorithm"] == {}


class TestCountSignificantRelationshipPlot:
    def _make_data(self) -> dict:
        return {
            "metadata": {
                "RT": "ResultTemplateCountSignificantRelationship",
                "model": "sd1.4",
                "task": "people",
                "unlearning_algorithm_list": ["distil", "uce", "munba"],
                "interference_entity_list": ["Emitter average clip diff"],
                "attribute_list": ["hpi_bin"],
            },
            "result": {
                "rows": [],
                "total_count": 3,
                "total": 6,
                "grouped_by_unlearning_algorithm": {
                    "distil": {"count": 1, "total": 2, "fraction": 0.5},
                    "uce": {"count": 2, "total": 2, "fraction": 1.0},
                    "munba": {"count": 0, "total": 2, "fraction": 0.0},
                },
                "grouped_by_attribute": {
                    "hpi_bin": {"count": 3, "total": 6, "fraction": 0.5},
                },
                "grouped_by_interference_entity": {
                    "Emitter average clip diff": {"count": 3, "total": 6, "fraction": 0.5},
                },
            },
        }

    def test_plot_by_algorithm_returns_fig(self) -> None:
        data = self._make_data()
        out = _rt_mod.ResultTemplateCountSignificantRelationship.plot(
            data, group_by="unlearning_algorithm", return_fig=True
        )
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)

    def test_plot_by_attribute_returns_fig(self) -> None:
        data = self._make_data()
        out = _rt_mod.ResultTemplateCountSignificantRelationship.plot(
            data, group_by="attribute", return_fig=True
        )
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)

    def test_plot_by_interference_entity_returns_fig(self) -> None:
        data = self._make_data()
        out = _rt_mod.ResultTemplateCountSignificantRelationship.plot(
            data, group_by="interference_entity", return_fig=True
        )
        assert out is not None
        fig, ax = out
        assert isinstance(fig, Figure)

    def test_plot_empty_grouped_returns_none(self) -> None:
        data = self._make_data()
        data["result"]["grouped_by_unlearning_algorithm"] = {}
        out = _rt_mod.ResultTemplateCountSignificantRelationship.plot(
            data, group_by="unlearning_algorithm", return_fig=True
        )
        assert out is None


class TestMetadataFilteredArtifact:
    """MetadataFiltered wraps the filtered task metadata file with the shared storage cascade,
    and get_metadata_filtered delegates to it."""

    def test_remote_path(self) -> None:
        from vision_unlearning.datasets.testbed import MetadataFiltered
        mf = MetadataFiltered(task='people')
        assert mf._get_data_path_remote() == 'metadata_people_2_enriched_filtered.json'

    def test_local_hit(self, tmp_path: Any) -> None:
        from vision_unlearning.datasets.testbed import MetadataFiltered
        mf = MetadataFiltered(task='people', base_folder=str(tmp_path))
        with open(mf._get_data_path_local(), 'w', encoding='utf-8') as f:
            json.dump([{'name': 'a'}], f)
        assert mf.compute() == [{'name': 'a'}]

    def test_get_metadata_filtered_delegates_to_artifact(self, tmp_path: Any) -> None:
        from vision_unlearning.datasets import testbed as _tb
        path = os.path.join(str(tmp_path), 'metadata_people_2_enriched_filtered.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([{'name': 'a'}, {'name': 'b'}], f)
        assert _tb.get_metadata_filtered('people', base_folder=str(tmp_path)) == [
            {'name': 'a'}, {'name': 'b'}
        ]

    def test_missing_everywhere_raises_not_implemented(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vision_unlearning.datasets.testbed import MetadataFiltered
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        mf = MetadataFiltered(task='people', base_folder=str(tmp_path))
        with pytest.raises(NotImplementedError):
            mf.compute()


class TestInterferencePerPairArtifact:
    """InterferencePerPair wraps a single per-pair interference file with the shared cascade;
    it complements (does not replace) get_interference_per_pair."""

    def test_remote_path(self) -> None:
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=3, method='distil', num_train_epochs=400
        )
        assert ipp._get_data_path_remote() == (
            'datasets/interferences_caused_by_people_3_distil_400.json'
        )

    def test_local_hit(self, tmp_path: Any) -> None:
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=0, method='distil', num_train_epochs=400,
            max_identities=2, base_folder=str(tmp_path),
        )
        os.makedirs(os.path.dirname(ipp._get_data_path_local()), exist_ok=True)
        payload = {'a': {'rmse': 1.0}, 'b': {'rmse': 2.0}}
        with open(ipp._get_data_path_local(), 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        assert ipp.compute() == payload

    def test_validate_rejects_wrong_length(self, tmp_path: Any) -> None:
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=0, method='distil', num_train_epochs=400,
            max_identities=5, base_folder=str(tmp_path),
        )
        os.makedirs(os.path.dirname(ipp._get_data_path_local()), exist_ok=True)
        with open(ipp._get_data_path_local(), 'w', encoding='utf-8') as f:
            json.dump({'a': {'rmse': 1.0}}, f)  # only 1 entry, expected 5
        with pytest.raises(AssertionError):
            ipp.compute()

    def test_missing_everywhere_raises_not_implemented(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=0, method='distil', num_train_epochs=400,
            base_folder=str(tmp_path),
        )
        with pytest.raises(NotImplementedError):
            ipp.compute()
