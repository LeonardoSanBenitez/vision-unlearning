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
import vision_unlearning.benchmarks.I_care.similarity as _sim_mod  # noqa: E402
import vision_unlearning.artifact as _artifact_mod  # noqa: E402
from vision_unlearning.artifact import ArtifactNotAvailableError  # noqa: E402


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
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(vb.MetadataFiltered, "compute", lambda self: [{"name": e} for e in entities],)
        # Patch at the actual call site in similarity.py (where the Similarity artifact binds it)
        monkeypatch.setattr(_sim_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in entities],)
        # Write a baseline embedding file in tmp_path/datasets/ (correct sub-path).
        emb_path = str(tmp_path / "datasets" / "embeddings_people_original.json")
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
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
        emb_path = str(datasets_dir / "embeddings_people_original.json")
        with open(emb_path, "w") as f:
            json.dump(embedding_data, f)

        monkeypatch.setattr(vb.MetadataFiltered, "compute", lambda self: [{"name": e} for e in entities],)
        # Patch at the actual call site in similarity.py (where the Similarity artifact binds it)
        monkeypatch.setattr(_sim_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in entities],)
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
        """The typed cascade error when the baseline embeddings are absent everywhere."""
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(vb.MetadataFiltered, "compute", lambda self: [{"name": "Alice"}],)
        # Patch at the actual call site in similarity.py (where the Similarity artifact binds it)
        monkeypatch.setattr(_sim_mod.MetadataFiltered, "compute", lambda self: [{"name": "Alice"}],)
        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="dino",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        with pytest.raises(ArtifactNotAvailableError, match="BaselineEmbeddings"):
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
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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

class TestInterferenceVisualSummaryImageResolution:
    """The image folders must be resolved through GeneratedDataset, not hand-built paths.

    The previous version of these tests asserted the opposite -- that _load_images called
    get_off_image_path/get_generated_dataset_folder directly -- which is exactly the bypass
    that made a fresh clone raise instead of downloading images that were on HuggingFace.
    """

    @staticmethod
    def _fake_metadata() -> List[Dict[str, Any]]:
        return [{"name": f"Person {chr(65 + i)}", "type": "politician"} for i in range(9)]

    def _patch_common(
        self, monkeypatch: pytest.MonkeyPatch, _rt: Any, calls: List[Dict[str, Any]]
    ) -> None:
        fake_metadata = self._fake_metadata()

        def fake_compute(self_ds: Any, seeds: List[int], prompts: List[str],
                         batch_size: int = 16) -> str:
            calls.append({
                "is_baseline": self_ds.method is None,
                "target": self_ds.target,
                "seeds": list(seeds),
                "prompts": list(prompts),
            })
            return "/fake/folder"

        monkeypatch.setattr(_rt.MetadataFiltered, "compute", lambda self: fake_metadata)
        monkeypatch.setattr(_rt.GeneratedDataset, "compute", fake_compute)
        monkeypatch.setattr(_rt, "get_target_overwrite", lambda task, method, name: (name, name))
        monkeypatch.setattr(_rt, "_encode_image_file", lambda *a, **kw: "base64fake")
        monkeypatch.setattr(
            _rt.InterferencePerPair, "compute",
            lambda self: {m["name"]: {"ssim": float(i)} for i, m in enumerate(fake_metadata)},
        )
        monkeypatch.setattr(_rt, "unlearning_algorithm_to_epochs", {"people": {"distil": 400}})
        monkeypatch.setattr(_rt, "mp_to_direction", {"ssim": "↑"})
        monkeypatch.setattr(_rt, "get_generated_dataset_file",
                            lambda state, seed, prompt: f"{state}_{seed}.png")

    def test_both_image_folders_resolve_through_generated_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Baseline (off) and entity (on) folders both go through GeneratedDataset.compute."""
        import vision_unlearning.benchmarks.I_care.result_templates as _rt
        calls: List[Dict[str, Any]] = []
        self._patch_common(monkeypatch, _rt, calls)

        rt = _rt.ResultTemplateInterferenceVisualSummary(
            unlearning_algorithm="distil", interference_pair="ssim", entity="Person A",
        )
        result = rt._compute_from_scratch()

        assert any(c["is_baseline"] for c in calls),             "the shared baseline folder was not resolved through GeneratedDataset"
        assert any(not c["is_baseline"] for c in calls),             "the entity folder was not resolved through GeneratedDataset"
        assert "off" in result["result"]["images"]
        assert "on" in result["result"]["images"]

    def test_full_task_prompt_list_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The COMPLETE task prompt list is passed, not just the 9 displayed entities.

        GeneratedDataset.exists() compares a file count against len(seeds) * len(prompts),
        so a partial list silently reports the folder as incomplete and triggers a spurious
        regeneration. Passing no list at all (the original bug) skipped the HuggingFace
        download entirely and fell through to a non-existent legacy folder.
        """
        import vision_unlearning.benchmarks.I_care.result_templates as _rt
        calls: List[Dict[str, Any]] = []
        self._patch_common(monkeypatch, _rt, calls)

        rt = _rt.ResultTemplateInterferenceVisualSummary(
            unlearning_algorithm="distil", interference_pair="ssim", entity="Person A",
        )
        rt._compute_from_scratch()

        expected_prompts = [f"An image of {m['name']}" for m in self._fake_metadata()]
        assert calls, "GeneratedDataset.compute was never called"
        for call in calls:
            assert call["prompts"] == expected_prompts, (
                "GeneratedDataset.compute must receive every prompt of the task"
            )
            assert call["seeds"] == [42, 43, 44, 45], (
                "GeneratedDataset.compute must receive the canonical seed list"
            )


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
    monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
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
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Cleo",
        )
        rt._resolve_entity()
        assert rt.entity == "Cleo"
        assert rt.entity_index == 2

    def test_index_to_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity_index=3,
        )
        rt._resolve_entity()
        assert rt.entity == "Dora"
        assert rt.entity_index == 3

    def test_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
        rt = _rt_mod.ResultTemplateMetricSimilarityAlignmentOne(
            unlearning_algorithm="uce", interference_pair="rmse",
            similarity_metric="dino", entity="Ada", entity_index=1,
        )
        with pytest.raises(ValueError, match="does not match"):
            rt._resolve_entity()

    def test_neither_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
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
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE_LABELS],)
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
    monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _MSAONE7_LABELS],)
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
        """Internal method name 'distil' must be shown as its display name 'spare'."""
        _patch_msaone_matrices(monkeypatch)
        rt = _rt_mod.ResultTemplateInterferenceBySimilarityRank(
            unlearning_algorithm="distil", interference_pair="clip_diff",
            similarity_metric="dino", entity="Ada", display_name_top_n=2,
            save_outputs=False,
        )
        fig, ax = rt.plot(rt._compute_from_scratch(), return_fig=True)
        title = ax.get_title()
        assert "spare" in title
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
    monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{"name": e} for e in _GRID_LABELS],)
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
        # y-tick labels use display method names (uce -> UCE, distil -> spare)
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert ytick_labels == ["UCE", "spare"]
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
    monkeypatch.setattr(_rt.MetadataFiltered, "compute", lambda self: fake_metadata)
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
    # Image folders are resolved through GeneratedDataset (local -> HuggingFace), so the
    # seam to stub is its compute(), not a raw path helper.
    monkeypatch.setattr(
        _rt.GeneratedDataset, "compute",
        lambda self, seeds, prompts, batch_size=16: str(tmp_path / ("on" if self.method else "off")),
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
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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
        monkeypatch.setattr(
            _artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False
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

    def test_missing_everywhere_raises_artifact_not_available(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vision_unlearning.datasets.testbed import MetadataFiltered
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        mf = MetadataFiltered(task='people', base_folder=str(tmp_path))
        with pytest.raises(ArtifactNotAvailableError):
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

    def test_missing_everywhere_raises_artifact_not_available(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=0, method='distil', num_train_epochs=400,
            base_folder=str(tmp_path),
        )
        with pytest.raises(ArtifactNotAvailableError):
            ipp.compute()

    def test_remote_hit_downloads_and_returns_data(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for a fresh-clone crash: a fresh local cache (nothing downloaded
        yet) must still resolve data that is only present on HuggingFace, not raise. This is
        the exact scenario a freshly-cloned Forgety instance hits on first use."""
        ipp = _meta_mod.InterferencePerPair(
            task='people', index=0, method='distil', num_train_epochs=400,
            max_identities=2, base_folder=str(tmp_path),
        )
        payload = {'a': {'rmse': 1.0}, 'b': {'rmse': 2.0}}
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: True
        )

        def fake_download(**kw: Any) -> None:
            local_path = ipp._get_data_path_local()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)

        monkeypatch.setattr(_artifact_mod, 'huggingface_dataset_file_download', fake_download)
        assert not os.path.exists(ipp._get_data_path_local())  # nothing local yet
        assert ipp.exists()  # cascade-aware existence check sees the HuggingFace copy
        assert ipp.compute() == payload

    def test_functional_wrapper_delegates_to_cascade(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_interference_per_pair / exists_interference_per_pair are thin wrappers over
        InterferencePerPair; both must see a HuggingFace-only file, not just a local one."""
        payload = {'a': {'rmse': 1.0}, 'b': {'rmse': 2.0}}
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: True
        )

        def fake_download(**kw: Any) -> None:
            path = _meta_mod.get_interference_per_pair_path(
                'people', 0, 'distil', 400, base_folder=str(tmp_path)
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)

        monkeypatch.setattr(_artifact_mod, 'huggingface_dataset_file_download', fake_download)
        assert _meta_mod.exists_interference_per_pair(
            'people', 0, 'distil', 400, base_folder=str(tmp_path)
        )
        assert _meta_mod.get_interference_per_pair(
            'people', 0, 'distil', 400, max_identities=2, base_folder=str(tmp_path)
        ) == payload


class TestSimilarityArtifact:
    """Similarity wraps the canonical similarity_{s}_{task}.json with the shared cascade and
    owns the heavy per-metric computation; SimilarityMatrix is a thin reader over it."""

    def test_remote_path(self) -> None:
        sim = _rt_mod.Similarity(task='people', similarity_metric='dino')
        assert sim._get_data_path_remote() == 'similarity_dino_people.json'

    def test_local_hit_returns_matrix(self, tmp_path: Any) -> None:
        sim = _rt_mod.Similarity(
            task='people', similarity_metric='dino', base_folder=str(tmp_path)
        )
        matrix = [
            {'emitter': 'Alice', 'Alice': 1.0, 'Bob': 0.3},
            {'emitter': 'Bob', 'Alice': 0.3, 'Bob': 1.0},
        ]
        with open(sim._get_data_path_local(), 'w', encoding='utf-8') as f:
            json.dump(matrix, f)
        assert sim.compute() == matrix

    def test_clip_recompute_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{'name': 'Alice'}])
        sim = _rt_mod.Similarity(
            task='people', similarity_metric='clip', base_folder=str(tmp_path)
        )
        with pytest.raises(ArtifactNotAvailableError):
            sim.compute()

    def test_unet_latent_remote_path(self) -> None:
        sim = _rt_mod.Similarity(task='breeds', similarity_metric='unet_latent')
        assert sim._get_data_path_remote() == 'similarity_unet_latent_breeds.json'

    def test_unet_latent_missing_cache_raises_artifact_not_available(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not NotImplementedError: callers treat only ArtifactNotAvailableError as missing data."""
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        monkeypatch.setattr(_rt_mod.MetadataFiltered, "compute", lambda self: [{'name': 'basenji'}])
        sim = _rt_mod.Similarity(
            task='breeds', similarity_metric='unet_latent', base_folder=str(tmp_path)
        )
        with pytest.raises(ArtifactNotAvailableError):
            sim.compute()

    def test_similarity_matrix_rt_reads_artifact(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RT wraps the artifact's matrix with display metadata (thin reader)."""
        matrix = [
            {'emitter': 'Alice', 'Alice': 1.0, 'Bob': 0.3},
            {'emitter': 'Bob', 'Alice': 0.3, 'Bob': 1.0},
        ]
        monkeypatch.setattr(
            _rt_mod.Similarity, 'compute', lambda self: list(matrix)
        )
        rt = _rt_mod.ResultTemplateSimilarityMatrix(
            task='people', similarity_metric='dino', base_folder=str(tmp_path),
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert data['metadata']['similarity_metric'] == 'dino'
        assert data['metadata']['_metric_key_name'] == 'similarity_metric'
        assert data['result'] == matrix


# ---------------------------------------------------------------------------
# UnetLatentSimilarity — the unet_latent metric (capture, cache, aggregation, matrix)
# ---------------------------------------------------------------------------

def _write_unet_latent_cache(
    path: str,
    entities: List[Dict[str, Any]],
    seeds: List[int],
    task: str = 'breeds',
    num_inference_steps: int = 50,
    allow_nan: bool = False,
) -> None:
    """Write a cache file directly, so a test can produce contents the class would refuse.

    ``entities`` items are ``{'name': ..., 'prompt': ..., 'latents': {seed: np.ndarray}}``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'metadata': {
            'task': task, 'model': 'sd1.4', 'model_name': 'CompVis/stable-diffusion-v1-4',
            'seeds': list(seeds), 'num_inference_steps': num_inference_steps,
            'batch_size': 1, 'dtype': 'float16', 'captured_at': '2026-08-06T00:00:00+00:00',
            'aggregation': 'mean over seeds, flatten C order, L2-normalise',
        },
        'entities': [
            {
                'name': entity['name'],
                'prompt': entity['prompt'],
                'latents': {
                    str(seed): np.asarray(values).tolist()
                    for seed, values in entity['latents'].items()
                },
            }
            for entity in entities
        ],
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, allow_nan=allow_nan)


def _single_channel_latent(values: List[float]) -> np.ndarray:
    """A (4, 1, 1) latent whose flattened C-order vector is exactly ``values``."""
    assert len(values) == 4
    return np.array(values, dtype=np.float16).reshape(4, 1, 1)


class TestUnetLatentSimilarity:
    """The unet_latent metric: cache round trip, aggregation, matrix, resume safety.

    All CPU-only: the GPU capture has no CI test (there is no GPU runner), its gate is
    ``validate_capture`` run manually before the bulk pass.
    """

    def _four_entity_cache(self, tmp_path: Any) -> Any:
        """A, B identical; C = -A; D orthogonal to all three. Labels are not alphabetical."""
        latents = _sim_mod.UnetLatentSimilarity(task='breeds', base_folder=str(tmp_path))
        vectors = {
            'zulu': [1.0, 0.0, 0.0, 0.0],     # A
            'alpha': [1.0, 0.0, 0.0, 0.0],    # B == A
            'mike': [-1.0, 0.0, 0.0, 0.0],    # C == -A
            'bravo': [0.0, 1.0, 0.0, 0.0],    # D orthogonal
        }
        _write_unet_latent_cache(
            latents.cache_path(),
            [
                {
                    'name': name,
                    'prompt': f"An image of a {name}",
                    'latents': {
                        seed: _single_channel_latent(values) for seed in latents.seeds
                    },
                }
                for name, values in vectors.items()
            ],
            seeds=list(latents.seeds),
        )
        return latents

    def test_cache_path_follows_the_per_entity_source_convention(self, tmp_path: Any) -> None:
        latents = _sim_mod.UnetLatentSimilarity(task='breeds', base_folder=str(tmp_path))
        assert latents.cache_path() == os.path.join(
            str(tmp_path), 'datasets', 'unet_latents_breeds_sd1.4.json',
        )

    def test_known_value_matrix(self, tmp_path: Any) -> None:
        """Rejects an all-ones matrix, a label-permuted matrix, and a wrong
        entity-to-vector association -- none of which symmetry/diagonal/range would catch."""
        latents = self._four_entity_cache(tmp_path)
        labels = ['zulu', 'alpha', 'mike', 'bravo']
        matrix = latents.matrix(labels)
        assert matrix.index.to_list() == labels
        assert matrix.columns.to_list() == labels
        expected = np.array([
            [1.0, 1.0, -1.0, 0.0],
            [1.0, 1.0, -1.0, 0.0],
            [-1.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        assert np.allclose(matrix.to_numpy(), expected, atol=1e-6)

    def test_matrix_follows_the_requested_label_order(self, tmp_path: Any) -> None:
        latents = self._four_entity_cache(tmp_path)
        matrix = latents.matrix(['bravo', 'mike'])
        assert matrix.index.to_list() == ['bravo', 'mike']
        assert np.allclose(matrix.to_numpy(), np.array([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6)

    def test_matrix_raises_for_an_entity_with_no_capture(self, tmp_path: Any) -> None:
        latents = self._four_entity_cache(tmp_path)
        with pytest.raises(ValueError, match='no captured latent'):
            latents.matrix(['zulu', 'never_captured'])

    def test_aggregation_is_mean_then_normalise(self, tmp_path: Any) -> None:
        """Rejects the reversed order of operations: normalising per seed and then
        averaging gives a different unit vector for this input."""
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42, 43],
        )
        _write_unet_latent_cache(
            latents.cache_path(),
            [{
                'name': 'basenji', 'prompt': 'An image of a basenji dog',
                'latents': {
                    42: _single_channel_latent([3.0, 4.0, 0.0, 0.0]),
                    43: _single_channel_latent([1.0, 0.0, 0.0, 0.0]),
                },
            }],
            seeds=[42, 43],
        )
        vector = latents.entity_vectors()['basenji']
        mean_then_normalise = np.array([2.0, 2.0, 0.0, 0.0]) / np.linalg.norm([2.0, 2.0, 0.0, 0.0])
        normalise_then_mean = (np.array([0.6, 0.8, 0.0, 0.0]) + np.array([1.0, 0.0, 0.0, 0.0])) / 2
        normalise_then_mean = normalise_then_mean / np.linalg.norm(normalise_then_mean)
        assert np.allclose(vector, mean_then_normalise, atol=1e-6)
        assert not np.allclose(vector, normalise_then_mean, atol=1e-2)

    def test_seed_selection_is_honoured(self, tmp_path: Any) -> None:
        """Rejects a seed_indices argument that is silently ignored."""
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42, 43],
        )
        _write_unet_latent_cache(
            latents.cache_path(),
            [{
                'name': 'basenji', 'prompt': 'An image of a basenji dog',
                'latents': {
                    42: _single_channel_latent([3.0, 4.0, 0.0, 0.0]),
                    43: _single_channel_latent([0.0, 0.0, 1.0, 0.0]),
                },
            }],
            seeds=[42, 43],
        )
        all_seeds = latents.entity_vectors()['basenji']
        first_seed = latents.entity_vectors(seed_indices=[0])['basenji']
        assert not np.allclose(all_seeds, first_seed)
        assert np.isclose(np.linalg.norm(first_seed), 1.0)

    def test_common_mode_removal_changes_the_vectors_and_keeps_unit_norm(
        self, tmp_path: Any
    ) -> None:
        """Rejects a diagnostic that is silently a no-op."""
        latents = self._four_entity_cache(tmp_path)
        raw = latents.entity_vectors()
        centred = latents.entity_vectors(remove_common_mode=True)
        assert not np.allclose(raw['bravo'], centred['bravo'])
        for vector in centred.values():
            assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_empty_seed_selection_raises(self, tmp_path: Any) -> None:
        latents = self._four_entity_cache(tmp_path)
        with pytest.raises(ValueError, match='nothing to average over'):
            latents.entity_vectors(seed_indices=[])

    def test_zero_vector_raises(self, tmp_path: Any) -> None:
        """A zero vector has no direction; normalising it would emit NaN cosines."""
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42],
        )
        _write_unet_latent_cache(
            latents.cache_path(),
            [{
                'name': 'basenji', 'prompt': 'An image of a basenji dog',
                'latents': {42: _single_channel_latent([0.0, 0.0, 0.0, 0.0])},
            }],
            seeds=[42],
        )
        with pytest.raises(ValueError, match='zero vector'):
            latents.entity_vectors()

    def test_non_finite_values_raise_on_read(self, tmp_path: Any) -> None:
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42],
        )
        _write_unet_latent_cache(
            latents.cache_path(),
            [{
                'name': 'basenji', 'prompt': 'An image of a basenji dog',
                'latents': {42: _single_channel_latent([float('nan'), 0.0, 0.0, 1.0])},
            }],
            seeds=[42],
            allow_nan=True,
        )
        with pytest.raises(ValueError, match='non-finite'):
            latents.entity_vectors()

    def test_non_finite_values_raise_on_write(self, tmp_path: Any) -> None:
        latents = _sim_mod.UnetLatentSimilarity(task='breeds', base_folder=str(tmp_path))
        with pytest.raises(ValueError, match='non-finite'):
            latents._write_cache(
                latents.cache_path(),
                {'basenji': {
                    'name': 'basenji', 'prompt': 'An image of a basenji dog',
                    'latents': {
                        str(seed): _single_channel_latent([np.inf, 0.0, 0.0, 1.0])
                        for seed in latents.seeds
                    },
                }},
            )

    def test_json_round_trip_is_bit_identical(self, tmp_path: Any) -> None:
        """Five significant digits round-trip float16 exactly; entity order and the exact
        prompt each entity was captured with must survive too."""
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42, 43],
        )
        rng = np.random.default_rng(0)
        original = {
            name: {
                'name': name,
                'prompt': f"An image of a {name} dog",
                'latents': {
                    str(seed): rng.standard_normal((4, 2, 3)).astype(np.float16)
                    for seed in [42, 43]
                },
            }
            for name in ['zulu', 'alpha', 'mike']
        }
        latents._write_cache(latents.cache_path(), original)
        metadata, reloaded = latents._read_cache(latents.cache_path())

        assert list(reloaded.keys()) == ['zulu', 'alpha', 'mike']
        assert metadata['seeds'] == [42, 43]
        for name, entity in original.items():
            assert reloaded[name]['prompt'] == entity['prompt']
            for seed, array in entity['latents'].items():
                assert reloaded[name]['latents'][seed].dtype == np.float16
                assert np.array_equal(reloaded[name]['latents'][seed], array)

    def test_load_stacks_seeds_in_order(self, tmp_path: Any) -> None:
        latents = self._four_entity_cache(tmp_path)
        stacked = latents.load()['zulu']
        assert stacked.shape == (4, 4, 1, 1)

    def test_load_without_a_cache_raises_artifact_not_available(self, tmp_path: Any) -> None:
        latents = _sim_mod.UnetLatentSimilarity(task='breeds', base_folder=str(tmp_path))
        with pytest.raises(ArtifactNotAvailableError):
            latents.load()

    @pytest.mark.parametrize(
        'field,value,expected',
        [
            ('seeds', [42, 99], 'seeds'),
            ('num_inference_steps', 10, 'num_inference_steps'),
            ('task', 'people', 'task'),
        ],
    )
    def test_resume_with_mismatched_metadata_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
        field: str, value: Any, expected: str,
    ) -> None:
        """Two capture generations must never be silently mixed."""
        monkeypatch.setattr(
            _rt_mod.MetadataFiltered, 'compute', lambda self: [{'name': 'basenji dog'}]
        )
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42, 43],
        )
        overrides: Dict[str, Any] = {'seeds': [42, 43], 'num_inference_steps': 50, 'task': 'breeds'}
        overrides[field] = value
        _write_unet_latent_cache(
            latents.cache_path_partial(),
            [{
                'name': 'basenji dog', 'prompt': 'An image of a basenji dog',
                'latents': {
                    seed: _single_channel_latent([1.0, 0.0, 0.0, 0.0]) for seed in [42, 43]
                },
            }],
            seeds=overrides['seeds'],
            task=overrides['task'],
            num_inference_steps=overrides['num_inference_steps'],
        )
        with pytest.raises(ValueError, match=expected):
            latents.capture(device='cpu')

    def test_resume_with_a_mismatched_prompt_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _rt_mod.MetadataFiltered, 'compute', lambda self: [{'name': 'basenji dog'}]
        )
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42],
        )
        _write_unet_latent_cache(
            latents.cache_path_partial(),
            [{
                'name': 'basenji dog', 'prompt': 'A photograph of a basenji',
                'latents': {42: _single_channel_latent([1.0, 0.0, 0.0, 0.0])},
            }],
            seeds=[42],
        )
        with pytest.raises(ValueError, match='prompt'):
            latents.capture(device='cpu')

    def test_resume_finalises_a_complete_partial_without_loading_a_pipeline(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every entity already captured: the cache is written and the partial removed,
        with no GPU work -- which is also why this test can run in the lite tier."""
        monkeypatch.setattr(
            _rt_mod.MetadataFiltered, 'compute', lambda self: [{'name': 'basenji dog'}]
        )
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42],
        )
        _write_unet_latent_cache(
            latents.cache_path_partial(),
            [{
                'name': 'basenji dog', 'prompt': 'An image of a basenji dog',
                'latents': {42: _single_channel_latent([1.0, 0.0, 0.0, 0.0])},
            }],
            seeds=[42],
        )
        latents.capture(device='cpu')
        assert os.path.exists(latents.cache_path())
        assert not os.path.exists(latents.cache_path_partial())
        assert list(latents.load().keys()) == ['basenji dog']

    def test_seed_split_stability_reports_both_measures(self, tmp_path: Any) -> None:
        latents = _sim_mod.UnetLatentSimilarity(
            task='breeds', base_folder=str(tmp_path), seeds=[42, 43, 44, 45],
        )
        rng = np.random.default_rng(1)
        _write_unet_latent_cache(
            latents.cache_path(),
            [
                {
                    'name': name,
                    'prompt': f"An image of a {name} dog",
                    'latents': {
                        seed: rng.standard_normal((4, 2, 2)).astype(np.float16)
                        for seed in [42, 43, 44, 45]
                    },
                }
                for name in ['a', 'b', 'c', 'd', 'e', 'f']
            ],
            seeds=[42, 43, 44, 45],
        )
        stability = latents.seed_split_stability(['a', 'b', 'c', 'd', 'e', 'f'], top_k=2)
        # Four seeds split into disjoint halves: 3 distinct unordered splits, 2 measures each.
        assert len(stability) == 6
        assert all(-1.0 <= value <= 1.0 for value in stability.values())

    def test_similarity_artifact_reads_the_cache_through_the_class(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: the unet_latent branch produces the emitter-row records."""
        monkeypatch.setattr(
            _artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: False
        )
        monkeypatch.setattr(
            _rt_mod.MetadataFiltered, 'compute',
            lambda self: [{'name': name} for name in ['zulu', 'alpha', 'mike', 'bravo']],
        )
        self._four_entity_cache(tmp_path)
        sim = _rt_mod.Similarity(
            task='breeds', similarity_metric='unet_latent', base_folder=str(tmp_path),
            save_outputs=False,
        )
        result = sim.compute()
        assert [row['emitter'] for row in result] == ['zulu', 'alpha', 'mike', 'bravo']
        assert result[0]['alpha'] == pytest.approx(1.0)
        assert result[0]['mike'] == pytest.approx(-1.0)
        assert result[0]['bravo'] == pytest.approx(0.0, abs=1e-6)
