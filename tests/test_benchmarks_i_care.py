"""Tests for Items 1, 2, 3 of the I-CARE benchmark extensions.

These tests cover:
  - Item 1: S_dino — DINOv2 as a new similarity metric (type_s, s_to_direction,
    GUI_TO_BACKEND, and ResultTemplateSimilarityMatrix._compute_from_scratch dino branch)
  - Item 2: me_dino — Embedding specificity ratio as a new type_me, and
    choose_metric_column_interference_per_entity matching the new column name
  - Item 3: ResultTemplateMethodSpecificity — registry, compute, plot

The code under test lives in forgety/libs/vision_unlearning_benchmarks_I_care_TEMP.py
(currently there for convenience; will move to vision-unlearning once stable).

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

# Add forgety/libs to path so we can import the TEMP module.
# When this code moves to vision-unlearning, this path manipulation goes away.
# Support both local (relative to this file) and Docker (/src/libs) layouts.
_FORGETY_LIBS_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'forgety', 'libs')),
    '/src/libs',  # Docker: forgety libs mounted at /src/libs
]
for _candidate in _FORGETY_LIBS_CANDIDATES:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import vision_unlearning_benchmarks_I_care_TEMP as vb  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs() -> Any:
    yield
    plt.close("all")


def _make_embedding_file(entities: List[str], dim: int = 8) -> dict:
    """Build a minimal embeddings JSON structure for testing."""
    rng = np.random.default_rng(42)
    embeddings = []
    for entity in entities:
        for seed in range(3):
            vec = rng.standard_normal(dim).tolist()
            embeddings.append({
                "prompted_entity": entity,
                "seed": seed,
                "prompt": f"a photo of {entity}",
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
        # Write a baseline embedding file in tmp_path
        emb_path = str(tmp_path / f"embeddings_people_original_distil_400.json")
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
        # Construct embeddings manually: entity A has all-ones, entity B has e1 direction
        dim = 4
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        embedding_data = {
            "embeddings": [
                {"prompted_entity": "A", "seed": 0, "prompt": "p", "embedding": vec_a},
                {"prompted_entity": "B", "seed": 0, "prompt": "p", "embedding": vec_b},
            ]
        }
        emb_path = str(tmp_path / "embeddings_people_original_distil_400.json")
        with open(emb_path, "w") as f:
            json.dump(embedding_data, f)

        entities = ["A", "B"]
        monkeypatch.setattr(
            vb, "get_metadata_filtered",
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
        assert result["A"]["A"] == pytest.approx(1.0, abs=1e-5)
        assert result["B"]["B"] == pytest.approx(1.0, abs=1e-5)
        assert result["A"]["B"] == pytest.approx(0.0, abs=1e-5)
        assert result["B"]["A"] == pytest.approx(0.0, abs=1e-5)

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
# Item 3 — ResultTemplateMethodSpecificity
# ---------------------------------------------------------------------------

class TestMethodSpecificityRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "MethodSpecificity" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert vb.rt_name_to_class["MethodSpecificity"] is vb.ResultTemplateMethodSpecificity

    def test_params_in_registry(self) -> None:
        expected = ["model", "task", "interference_entity", "unlearning_algorithm_list"]
        assert vb.rt_name_to_params["MethodSpecificity"] == expected

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

        rt = vb.ResultTemplateMethodSpecificity(
            task="people",
            interference_entity="Embedding specificity ratio",
            unlearning_algorithm_list=["distil", "uce"],
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        data = rt._compute_from_scratch()

        assert "result" in data
        assert "metadata" in data
        assert data["metadata"]["RT"] == "ResultTemplateMethodSpecificity"
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

        rt = vb.ResultTemplateMethodSpecificity(
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

        rt = vb.ResultTemplateMethodSpecificity(
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

        rt = vb.ResultTemplateMethodSpecificity(
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
                "RT": "ResultTemplateMethodSpecificity",
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
        result = vb.ResultTemplateMethodSpecificity.plot(data, return_fig=True)
        assert result is not None
        fig, ax = result
        assert isinstance(fig, Figure)
