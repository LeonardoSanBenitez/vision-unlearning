"""Tests for ResultTemplateEmbeddingUnlearningProfile and
ResultTemplateEmbeddingForgettingEfficiency.

All tests are CPU-only, require no GPU, no network, and no real data files.
Fake data is written to a tmp_path directory to keep tests hermetic.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402
import vision_unlearning.benchmarks.I_care.result_templates as _rt_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs() -> Any:
    yield
    plt.close("all")


ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve"]
FORGOTTEN_ENTITY = "Alice"
DIM = 16


def _make_embedding_records(
    entities: List[str],
    dim: int,
    rng: np.random.Generator,
    lora_state: str,
    forgotten_entity: str,
) -> List[Dict]:
    """Build fake embedding records.

    If lora_state=='on', the forgotten entity's embeddings are rotated to a
    nearly-orthogonal direction so cosine distance is large (~1.0) while
    retained entities remain close to their baseline (small cosine distance).
    This ensures embedding_specificity_ratio > 1 in tests.
    """
    records = []
    for entity in entities:
        for seed in range(4):
            vec = rng.standard_normal(dim).tolist()
            if lora_state == "on" and entity == forgotten_entity:
                # Replace with a vector in a completely different direction.
                # Use a fixed orthogonal direction: [1, 0, 0, ..., 0] flipped
                # to avoid correlation with any random baseline vector.
                flipped = [-v for v in vec]
                vec = flipped
            elif lora_state == "on":
                # Retained entities: tiny perturbation (cosine stays near 0)
                vec = [v + rng.standard_normal() * 0.001 for v in vec]
            records.append({
                "prompted_entity": entity,
                "seed": seed,
                "prompt": f"a photo of {entity}",
                "embedding": vec,
            })
    return records


def _write_baseline_file(tmp_path: Any, task: str, method: str, epochs: int) -> str:
    rng = np.random.default_rng(0)
    fname = f"embeddings_{task}_original_{method}_{epochs:03d}.json"
    path = str(tmp_path / "datasets" / fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "metadata": {
            "task": task, "forgotten_entity": "original",
            "method": method, "num_train_epochs": epochs,
            "lora_state": "off", "embedding_model": "dinov2_vits14", "embedding_dim": DIM,
        },
        "embeddings": _make_embedding_records(ENTITIES, DIM, rng, "off", FORGOTTEN_ENTITY),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_entity_file(
    tmp_path: Any, task: str, entity: str, method: str, epochs: int
) -> str:
    rng = np.random.default_rng(1)
    fname = f"embeddings_{task}_{entity}_{method}_{epochs:03d}.json"
    path = str(tmp_path / "datasets" / fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "metadata": {
            "task": task, "forgotten_entity": entity,
            "method": method, "num_train_epochs": epochs,
            "lora_state": "on", "embedding_model": "dinov2_vits14", "embedding_dim": DIM,
        },
        "embeddings": _make_embedding_records(ENTITIES, DIM, rng, "on", entity),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_interference_per_entity(
    tmp_path: Any, task: str, method: str, epochs: int,
    include_ratio: bool = True,
) -> str:
    """Write a minimal interference_per_entity JSON for testing.

    Includes:
    - clip_diff column (emitter_average_clip_diff) for scatter tests.
    - embedding_specificity_ratio column when include_ratio=True (required by
      ResultTemplateEmbeddingForgettingEfficiency._compute_from_scratch).
    """
    col_clip = f"metric_{method}_{epochs}_emitter_average_clip_diff (↑)"
    col_ratio = f"metric_{method}_{epochs}_embedding_specificity_ratio (↑)"
    rows = []
    for i, ent in enumerate(ENTITIES):
        row: Dict[str, Any] = {
            "name": ent,
            col_clip: float(-5.0 - i),       # decreasing clip_diff
        }
        if include_ratio:
            row[col_ratio] = float(1.5 + i * 0.1)  # ratio > 1 for all entities
        rows.append(row)
    path = str(tmp_path / f"interference_per_entity_{task}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return path


def _write_all_entity_embedding_files(
    tmp_path: Any, task: str, method: str, epochs: int
) -> None:
    """Write baseline + per-entity embedding files for all ENTITIES.

    Each per-entity file has the SELF entity's embeddings flipped (large cosine
    distance from baseline) and retained entities with tiny perturbations
    (small cosine distance), so embedding_specificity_ratio > 1 for all entities.
    """
    _write_baseline_file(tmp_path, task, method, epochs)
    for entity in ENTITIES:
        # Write a custom entity file where THIS entity is the forgotten one
        rng = np.random.default_rng(hash(entity) % (2**31))
        fname = f"embeddings_{task}_{entity}_{method}_{epochs:03d}.json"
        path = str(tmp_path / "datasets" / fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "metadata": {
                "task": task, "forgotten_entity": entity,
                "method": method, "num_train_epochs": epochs,
                "lora_state": "on", "embedding_model": "dinov2_vits14", "embedding_dim": DIM,
            },
            "embeddings": _make_embedding_records(ENTITIES, DIM, rng, "on", entity),
        }
        with open(path, "w") as f:
            json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests for ResultTemplateEmbeddingUnlearningProfile
# ---------------------------------------------------------------------------

class TestEmbeddingUnlearningProfileRegistered:
    def test_in_rt_name_to_class(self) -> None:
        assert "EmbeddingUnlearningProfile" in vb.rt_name_to_class

    def test_in_rt_name_to_params(self) -> None:
        params = vb.rt_name_to_params["EmbeddingUnlearningProfile"]
        assert "model" in params
        assert "task" in params
        assert "unlearning_algorithm" in params
        assert "entity" in params

    def test_class_importable(self) -> None:
        assert hasattr(vb, "ResultTemplateEmbeddingUnlearningProfile")


class TestEmbeddingUnlearningProfileSerialize:
    def test_serialize_parameters(self) -> None:
        rt = vb.ResultTemplateEmbeddingUnlearningProfile(
            task="people",
            unlearning_algorithm="distil",
            entity="George W. Bush",
        )
        s = rt._serialize_parameters()
        assert "distil" in s
        assert "george_w._bush" in s.lower()

    def test_serialize_no_spaces(self) -> None:
        rt = vb.ResultTemplateEmbeddingUnlearningProfile(
            task="people",
            unlearning_algorithm="uce",
            entity="Some Entity",
        )
        s = rt._serialize_parameters()
        assert " " not in s


class TestEmbeddingUnlearningProfileCompute:
    def test_smoke(self, tmp_path: Any) -> None:
        """Minimal compute run with fake data."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)

        with patch.dict(
            _rt_mod.__dict__,
            {"unlearning_algorithm_to_epochs": {"people": {"distil": epochs, "uce": 0, "munba": 200}}},
        ):
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            # Patch epochs lookup
            rt_patched = rt.__class__(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": epochs, "uce": 0, "munba": 200}}
        try:
            data = rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs

        assert "metadata" in data
        assert "result" in data
        assert data["metadata"]["RT"] == "ResultTemplateEmbeddingUnlearningProfile"
        assert data["metadata"]["entity"] == FORGOTTEN_ENTITY
        result = data["result"]
        assert result["forgotten_entity_index"] == ENTITIES.index(FORGOTTEN_ENTITY)
        assert len(result["entity_labels"]) == len(ENTITIES)
        assert len(result["pca_off"]) == len(ENTITIES)
        assert len(result["pca_on"]) == len(ENTITIES)
        assert "self_displacement_magnitude" in result
        assert "embedding_specificity_ratio" in result

    def test_specificity_ratio_greater_than_one(self, tmp_path: Any) -> None:
        """With large shift on forgotten entity, specificity ratio > 1."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": epochs, "uce": 0, "munba": 200}}
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            data = rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs

        ratio = data["result"]["embedding_specificity_ratio"]
        assert ratio > 1.0, f"Expected ratio > 1 for strongly-shifted entity, got {ratio}"
        assert data["result"]["targeted"] is True

    def test_missing_baseline_raises(self, tmp_path: Any) -> None:
        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": 400, "uce": 0, "munba": 200}}
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task="people",
                unlearning_algorithm="distil",
                entity="Alice",
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            with pytest.raises(FileNotFoundError, match="Baseline embedding file not found"):
                rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs

    def test_missing_entity_file_raises(self, tmp_path: Any) -> None:
        task = "people"
        method = "distil"
        epochs = 400
        _write_baseline_file(tmp_path, task, method, epochs)
        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": 400, "uce": 0, "munba": 200}}
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity="Alice",
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            with pytest.raises(FileNotFoundError, match="Entity embedding file not found"):
                rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs


class TestEmbeddingUnlearningProfilePlot:
    def test_plot_runs_without_error(self, tmp_path: Any) -> None:
        task = "people"
        method = "distil"
        epochs = 400
        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": epochs, "uce": 0, "munba": 200}}
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            data = rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs

        result = vb.ResultTemplateEmbeddingUnlearningProfile.plot(data, return_fig=True)
        assert result is not None
        fig, axes = result
        assert fig is not None


# ---------------------------------------------------------------------------
# Tests for ResultTemplateEmbeddingForgettingEfficiency
# ---------------------------------------------------------------------------

class TestEmbeddingForgettingEfficiencyRegistered:
    def test_in_rt_name_to_class(self) -> None:
        assert "EmbeddingForgettingEfficiency" in vb.rt_name_to_class

    def test_in_rt_name_to_params(self) -> None:
        params = vb.rt_name_to_params["EmbeddingForgettingEfficiency"]
        assert "model" in params
        assert "task" in params
        assert "unlearning_algorithm" in params

    def test_class_importable(self) -> None:
        assert hasattr(vb, "ResultTemplateEmbeddingForgettingEfficiency")


class TestEmbeddingForgettingEfficiencySerialize:
    def test_serialize_parameters(self) -> None:
        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task="people",
            unlearning_algorithm="distil",
        )
        s = rt._serialize_parameters()
        assert "distil" in s
        assert "people" in s
        assert "sd1.4" in s


class TestEmbeddingForgettingEfficiencyCompute:
    """EmbeddingForgettingEfficiency reads ratios from interference_per_entity file."""

    def test_smoke(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimal compute run: requires IPE file with embedding_specificity_ratio column."""
        task = "people"
        method = "distil"
        epochs = 400
        # Write IPE with both ratio and clip_diff columns
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=True)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})

        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
            n_permutations=100,
        )
        data = rt._compute_from_scratch()

        assert "metadata" in data
        assert "result" in data
        assert data["metadata"]["RT"] == "ResultTemplateEmbeddingForgettingEfficiency"
        result = data["result"]
        assert result["n_entities"] == len(ENTITIES)
        assert len(result["entity_names"]) == len(ENTITIES)
        assert len(result["embedding_specificity_ratios"]) == len(ENTITIES)
        assert 0.0 <= result["fraction_above_1"] <= 1.0
        assert isinstance(result["mean_ratio"], float)
        assert isinstance(result["std_ratio"], float)
        assert isinstance(result["spearman_r"], float)

    def test_ratios_read_from_ipe_not_recomputed(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Values in result must match what was written to IPE, not be recomputed."""
        task = "people"
        method = "distil"
        epochs = 400
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=True)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})

        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
            n_permutations=50,
        )
        data = rt._compute_from_scratch()

        # _write_interference_per_entity writes ratio = 1.5 + i*0.1 for entity i
        expected_ratios = [1.5 + i * 0.1 for i in range(len(ENTITIES))]
        actual_ratios = data["result"]["embedding_specificity_ratios"]
        assert actual_ratios == expected_ratios, (
            f"Ratios should be read from IPE, not recomputed. "
            f"Expected {expected_ratios}, got {actual_ratios}"
        )

    def test_missing_ipe_raises(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """No IPE file at all → InterferencePerEntity.compute() raises (no local, no HF)."""
        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        import vision_unlearning.benchmarks.I_care.metadata as meta_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": 400, "uce": 0, "munba": 200}})
        # Patch HF check to return False so it falls through to compute_from_scratch
        monkeypatch.setattr(meta_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False)
        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task="people",
            unlearning_algorithm="distil",
            base_folder=str(tmp_path),
            save_outputs=False,
        )
        with pytest.raises(NotImplementedError):
            rt._compute_from_scratch()

    def test_missing_ratio_column_raises(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IPE present but no embedding_specificity_ratio column → FileNotFoundError."""
        task = "people"
        method = "distil"
        epochs = 400
        # Write IPE without ratio column
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=False)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})

        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
        )
        with pytest.raises(FileNotFoundError, match="embedding_specificity_ratio"):
            rt._compute_from_scratch()

    def test_permutation_p_value_in_range(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        task = "people"
        method = "distil"
        epochs = 400
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=True)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})
        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
            n_permutations=200,
        )
        data = rt._compute_from_scratch()
        p = data["result"]["permutation_pvalue"]
        assert 0.0 <= p <= 1.0, f"p-value out of range: {p}"

    def test_metadata_contains_references(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        task = "people"
        method = "distil"
        epochs = 400
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=True)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})
        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
            n_permutations=100,
        )
        data = rt._compute_from_scratch()
        meta = data["metadata"]
        assert "concealment_reference" in meta
        assert "pinpoint_reference" in meta
        assert "2409.05668" in meta["concealment_reference"]
        assert "ICCV 2025" in meta["pinpoint_reference"]


class TestEmbeddingForgettingEfficiencyPlot:
    def test_plot_runs_without_error(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        task = "people"
        method = "distil"
        epochs = 400
        _write_interference_per_entity(tmp_path, task, method, epochs, include_ratio=True)

        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": epochs, "uce": 0, "munba": 200}})
        rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
            task=task,
            unlearning_algorithm=method,
            base_folder=str(tmp_path),
            save_outputs=False,
            n_permutations=100,
        )
        data = rt._compute_from_scratch()

        result = vb.ResultTemplateEmbeddingForgettingEfficiency.plot(data, return_fig=True)
        assert result is not None
        fig, axes = result
        assert fig is not None
