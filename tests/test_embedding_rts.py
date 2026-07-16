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
# "No network" for every test in this file is now enforced suite-wide by the autouse
# `_no_real_huggingface_by_default` fixture in `tests/conftest.py` -- no per-file fixture
# needed here anymore.

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
                "prompt": f"An image of {entity}",
                "embedding": vec,
            })
    return records


def _write_baseline_file(tmp_path: Any, task: str, method: str, epochs: int) -> str:
    """Write the canonical, method-agnostic baseline embedding file.

    The baseline is produced by the original model with no unlearning, so its name carries
    no method or epoch (embeddings_{task}_original.json) — the only name the RTs read now
    that the per-method fallback has been removed.
    """
    rng = np.random.default_rng(0)
    fname = f"embeddings_{task}_original.json"
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


def _write_baseline_file_legacy(tmp_path: Any, task: str, method: str, epochs: int) -> str:
    """Write the OBSOLETE per-method baseline name.

    Retained solely to feed the negative test that proves the per-method fallback was
    removed: with only this file present (and no canonical file), the RT must fail to find
    a baseline.
    """
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
        # Dots are stripped from the entity slug: "George W. Bush" → "george_w_bush"
        assert "george_w_bush" in s.lower()
        assert "george_w._bush" not in s.lower()

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
        """With large shift on forgotten entity and an IPE file providing the ratio, ratio > 1."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)

        # The inline fallback was removed: ratio only comes from IPE now.
        # Write an IPE file with a ratio > 1 for the forgotten entity.
        col_ratio = f"metric_{method}_{epochs}_embedding_specificity_ratio (↑)"
        ipe_rows = [{"name": ent, col_ratio: 2.5 if ent == FORGOTTEN_ENTITY else 0.8}
                    for ent in ENTITIES]
        ipe_path = str(tmp_path / f"interference_per_entity_{task}.json")
        with open(ipe_path, "w", encoding="utf-8") as f:
            json.dump(ipe_rows, f)

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
        assert ratio > 1.0, f"Expected ratio > 1 (from IPE), got {ratio}"
        assert data["result"]["targeted"] is True
        assert data["result"]["ratio_source"] == "ipe"

    def test_ratio_source_not_available_when_no_ipe(self, tmp_path: Any) -> None:
        """When no IPE file is present, ratio_source must be 'not_available' and ratio NaN.

        The inline fallback was removed: it used a different denominator than the IPE value
        and gave a misleading label.  With no IPE, the ratio is genuinely unavailable.
        """
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)
        # No IPE file written

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

        assert data["result"]["ratio_source"] == "not_available", (
            "ratio_source must be 'not_available' when IPE file is absent (inline fallback removed)"
        )
        assert math.isnan(data["result"]["embedding_specificity_ratio"]), (
            "embedding_specificity_ratio must be NaN when IPE is absent"
        )
        assert data["result"]["targeted"] is False, (
            "targeted must be False when ratio is NaN"
        )

    def test_ratio_source_ipe_when_ipe_present(self, tmp_path: Any) -> None:
        """When IPE file has the ratio column, ratio_source must be 'ipe'."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)
        # Write IPE with ratio for FORGOTTEN_ENTITY
        col_ratio = f"metric_{method}_{epochs}_embedding_specificity_ratio (↑)"
        ipe_rows = [{"name": ent, col_ratio: 1.8 if ent == FORGOTTEN_ENTITY else 1.0}
                    for ent in ENTITIES]
        ipe_path = str(tmp_path / f"interference_per_entity_{task}.json")
        with open(ipe_path, "w", encoding="utf-8") as f:
            json.dump(ipe_rows, f)

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

        assert data["result"]["ratio_source"] == "ipe", (
            "ratio_source must be 'ipe' when IPE file has the ratio column for this entity"
        )
        assert math.isclose(data["result"]["embedding_specificity_ratio"], 1.8, rel_tol=1e-6), (
            "Ratio value must match IPE when source is 'ipe'"
        )

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

    def test_mp_clip_diffs_loaded_when_mp_file_present(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When an Mp file exists for this entity, mp_clip_diffs should have non-NaN values."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)

        # Write a fake Mp file for entity index 0 (Alice = index 0 in ENTITIES)
        mp_data = {ent.replace(" ", "_"): {"clip_diff": float(-1.0 - i), "brisque_diff": 0.0,
                                            "rmse": 0.0, "ssim": 0.0}
                   for i, ent in enumerate(ENTITIES) if ent != FORGOTTEN_ENTITY}
        mp_path = str(tmp_path / "datasets" / f"interferences_caused_by_{task}_0_{method}_{epochs}.json")
        os.makedirs(os.path.dirname(mp_path), exist_ok=True)
        with open(mp_path, "w", encoding="utf-8") as f:
            json.dump(mp_data, f)

        monkeypatch.setattr(
            _rt_mod, "unlearning_algorithm_to_epochs",
            {"people": {"distil": epochs, "uce": 0, "munba": 200}},
        )

        # Patch get_metadata_filtered to return ENTITIES in order (Alice = index 0)
        fake_meta = [{"name": ent.replace(" ", "_")} for ent in ENTITIES]
        with patch("vision_unlearning.benchmarks.I_care.result_templates.get_metadata_filtered",
                   return_value=fake_meta):
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            with patch.object(rt, "_resolve_hf_entity", return_value=FORGOTTEN_ENTITY):
                data = rt._compute_from_scratch()

        mp_diffs = data["result"]["mp_clip_diffs"]
        n_valid = sum(1 for v in mp_diffs if not math.isnan(v))
        assert n_valid > 0, "Expected at least one non-NaN mp_clip_diff when Mp file is present"

    def test_mp_clip_diffs_all_nan_when_mp_file_missing(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the Mp file does not exist, mp_clip_diffs must be all-NaN (not an error)."""
        task = "people"
        method = "distil"
        epochs = 400

        _write_baseline_file(tmp_path, task, method, epochs)
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)
        # No Mp file written

        monkeypatch.setattr(
            _rt_mod, "unlearning_algorithm_to_epochs",
            {"people": {"distil": epochs, "uce": 0, "munba": 200}},
        )

        fake_meta = [{"name": ent.replace(" ", "_")} for ent in ENTITIES]
        with patch("vision_unlearning.benchmarks.I_care.result_templates.get_metadata_filtered",
                   return_value=fake_meta):
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path),
                save_outputs=False,
            )
            with patch.object(rt, "_resolve_hf_entity", return_value=FORGOTTEN_ENTITY):
                data = rt._compute_from_scratch()

        mp_diffs = data["result"]["mp_clip_diffs"]
        assert all(math.isnan(v) for v in mp_diffs), (
            "All mp_clip_diffs must be NaN when the Mp file is missing"
        )

    def test_mismatched_entity_sets_raises(self, tmp_path: Any) -> None:
        """If the entity file is missing an entity that the baseline has, raise ValueError.

        The silent intersection was removed precisely to catch this: a different off_mat
        per entity produces a different PCA basis, making dot positions inconsistent.
        """
        import json as _json
        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        task = "people"
        method = "distil"
        epochs = 400
        _write_baseline_file(tmp_path, task, method, epochs)

        # Write an entity file that is missing one entity ('Eve') relative to the baseline.
        rng = np.random.default_rng(99)
        incomplete_entities = [e for e in ENTITIES if e != "Eve"]  # drop one
        fname = f"embeddings_{task}_{FORGOTTEN_ENTITY}_{method}_{epochs:03d}.json"
        path = str(tmp_path / "datasets" / fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "metadata": {"task": task, "forgotten_entity": FORGOTTEN_ENTITY,
                         "method": method, "num_train_epochs": epochs,
                         "lora_state": "on", "embedding_model": "dinov2_vits14",
                         "embedding_dim": DIM},
            "embeddings": _make_embedding_records(
                incomplete_entities, DIM, rng, "on", FORGOTTEN_ENTITY
            ),
        }
        with open(path, "w") as f:
            _json.dump(data, f)

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
            with pytest.raises(ValueError, match="mismatched entity sets"):
                rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs


class TestBaselinePerMethodObliteration:
    """The obsolete per-method baseline name is unrepresentable and no longer read.

    Guards the deliberate, authorized backward-compatibility break: the baseline embedding
    depends only on task/model/embedding-function, never on an unlearning method or epoch.
    """

    def test_baseline_embeddings_interface_has_no_method_or_epoch(self) -> None:
        """B1 — interface shape: BaselineEmbeddings cannot carry a method/epoch.

        This is what makes the wrong name *unrepresentable* rather than merely unused: the
        artifact has no field through which a method or epoch could enter the baseline path.
        """
        from vision_unlearning.benchmarks.I_care.metadata import BaselineEmbeddings
        fields = set(BaselineEmbeddings.model_fields)
        for forbidden in ("unlearning_algorithm", "method", "epochs", "num_train_epochs"):
            assert forbidden not in fields, (
                f"BaselineEmbeddings must not expose '{forbidden}' — the baseline is "
                f"method/epoch-agnostic."
            )

    def test_baseline_per_method_fallback_removed(self, tmp_path: Any) -> None:
        """B2 — negative test for the break: only the obsolete per-method file present.

        With the canonical embeddings_{task}_original.json absent and ONLY the old
        embeddings_{task}_original_{method}_{epochs}.json present, the RT must raise (it no
        longer falls back to the per-method name). Nothing else in the suite would catch a
        quiet re-introduction of that fallback.
        """
        task, method, epochs = "people", "distil", 400
        _write_baseline_file_legacy(tmp_path, task, method, epochs)  # old name only
        _write_entity_file(tmp_path, task, FORGOTTEN_ENTITY, method, epochs)
        import vision_unlearning.benchmarks.I_care.result_templates as rt_mod
        original_epochs = rt_mod.unlearning_algorithm_to_epochs
        rt_mod.unlearning_algorithm_to_epochs = {"people": {"distil": epochs, "uce": 0, "munba": 200}}
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task, unlearning_algorithm=method, entity=FORGOTTEN_ENTITY,
                base_folder=str(tmp_path), save_outputs=False,
            )
            with pytest.raises(FileNotFoundError, match="Baseline embedding file not found"):
                rt._compute_from_scratch()
        finally:
            rt_mod.unlearning_algorithm_to_epochs = original_epochs


class TestEmbeddingUnlearningProfileGrouping:
    """The embedding mean groups records by ``prompt``, never by ``prompted_entity`` (§6)."""

    def test_mean_embeddings_groups_by_prompt_not_prompted_entity(self) -> None:
        """C — divergent-prompt: grouping must follow ``prompt``, not ``prompted_entity``.

        Two records for the same entity share one prompt but carry DIFFERENT (inconsistent)
        ``prompted_entity`` strings. Grouping by prompt yields one bucket for that entity
        (correct); grouping by ``prompted_entity`` would split it into two — the §6 violation
        this fix removes. Without divergence, both grouping keys agree and the test cannot
        distinguish a correct fix from a broken one, which is why the fixture forces it.
        """
        raw = {
            "embeddings": [
                {"prompted_entity": "Alice", "seed": 0,
                 "prompt": "An image of Alice", "embedding": [1.0, 0.0]},
                # Same entity/prompt, deliberately inconsistent prompted_entity:
                {"prompted_entity": "a photo of Alice", "seed": 1,
                 "prompt": "An image of Alice", "embedding": [3.0, 0.0]},
                {"prompted_entity": "Bob", "seed": 0,
                 "prompt": "An image of Bob", "embedding": [0.0, 2.0]},
            ]
        }
        means = _rt_mod.ResultTemplateEmbeddingUnlearningProfile._mean_embeddings(raw)
        # Grouping by prompt → exactly two entities, keyed by the stripped canonical name.
        assert set(means.keys()) == {"Alice", "Bob"}
        # Alice's mean averages BOTH divergent-prompted_entity records: (1 + 3) / 2 = 2.
        # Under the old prompted_entity grouping this would be [1.0, 0.0] and the key set
        # would additionally contain "a photo of Alice".
        assert means["Alice"].tolist() == [2.0, 0.0]
        assert means["Bob"].tolist() == [0.0, 2.0]


class TestEmbeddingUnlearningProfileClipDiffNormalization:
    """Test that clip_diff is populated when IPE names use underscores but embedding labels use spaces."""

    def test_clip_diffs_loaded_when_ipe_name_has_underscores(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IPE stores 'Alice_Smith' (underscores) but embedding file records use 'Alice Smith' (spaces).
        The EUP must normalize so clip_diffs are non-NaN."""
        task = "people"
        method = "distil"
        epochs = 400
        # Entities: embedding files record them with SPACES, IPE stores them with UNDERSCORES.
        entities_with_spaces = ["Alice Smith", "Bob Jones", "Carol Doe", "Dave Lee", "Eve Fox"]
        forgotten_entity_space = "Alice Smith"
        forgotten_entity_underscore = "Alice_Smith"

        # Write baseline + entity embedding files using space-delimited entity names (as real files do).
        # Real filenames: embeddings_people_Andre Agassi_distil_400.json (spaces in name, not underscores).
        # "original" is the baseline; each entity file uses the space-delimited name directly.
        for fname_suffix in ["original", *entities_with_spaces]:
            rng2 = np.random.default_rng(hash(fname_suffix) % (2**31))
            if fname_suffix == "original":
                # Baseline is method-agnostic (no method/epoch in the name).
                fname = f"embeddings_{task}_original.json"
            else:
                fname = f"embeddings_{task}_{fname_suffix}_{method}_{epochs:03d}.json"
            path = str(tmp_path / "datasets" / fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            records = []
            for ent_space in entities_with_spaces:
                for seed in range(4):
                    vec = rng2.standard_normal(16).tolist()
                    # Large shift for the forgotten entity in its own file → ratio > 1
                    if fname_suffix == forgotten_entity_space and ent_space == forgotten_entity_space:
                        vec = [v * -10 for v in rng2.standard_normal(16).tolist()]
                    records.append({"prompted_entity": ent_space, "seed": seed, "prompt": f"An image of {ent_space}", "embedding": vec})
            with open(path, "w") as f:
                json.dump({"metadata": {"embedding_model": "dinov2_vits14"}, "embeddings": records}, f)

        # Write IPE with UNDERSCORE names and both ratio + clip_diff columns
        col_clip = f"metric_{method}_{epochs}_emitter_minus_receiver_average_clip_diff (↑)"
        col_ratio = f"metric_{method}_{epochs}_embedding_specificity_ratio (↑)"
        rows = []
        for i, ent_space in enumerate(entities_with_spaces):
            rows.append({
                "name": ent_space.replace(" ", "_"),   # underscore format in IPE
                col_clip: float(-1.0 - i),
                col_ratio: float(1.5 + i * 0.1),
            })
        ipe_path = str(tmp_path / f"interference_per_entity_{task}.json")
        with open(ipe_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)

        monkeypatch.setattr(
            _rt_mod, "unlearning_algorithm_to_epochs",
            {"people": {"distil": epochs, "uce": 0, "munba": 200}},
        )

        rt = vb.ResultTemplateEmbeddingUnlearningProfile(
            task=task,
            unlearning_algorithm=method,
            entity=forgotten_entity_space,   # entity name WITH spaces
            base_folder=str(tmp_path),
            save_outputs=False,
        )

        # Patch _resolve_hf_entity to return space-delimited name (matches embedding records)
        # This simulates how get_target_overwrite() returns the HF display name
        with patch.object(rt, "_resolve_hf_entity", return_value=forgotten_entity_space):
            data = rt._compute_from_scratch()

        clip_diffs = data["result"]["clip_diffs"]
        valid_count = sum(1 for x in clip_diffs if x is not None and not math.isnan(float(x)))
        # All 5 entities should have clip_diff populated (IPE has underscore names → normalized to spaces)
        assert valid_count == len(entities_with_spaces), (
            f"Expected {len(entities_with_spaces)} non-NaN clip_diffs but got {valid_count}. "
            f"IPE underscore-to-space normalization may have failed."
        )


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

    def test_n_valid_matches_non_nan_ratios(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """n_valid must equal the number of entities with non-NaN ratios."""
        task = "people"
        method = "distil"
        epochs = 400
        # Write IPE where only 3 of 5 entities have a non-NaN ratio
        col_clip = f"metric_{method}_{epochs}_emitter_average_clip_diff (↑)"
        col_ratio = f"metric_{method}_{epochs}_embedding_specificity_ratio (↑)"
        rows = []
        for i, ent in enumerate(ENTITIES):
            row: Dict[str, Any] = {"name": ent, col_clip: float(-5.0 - i)}
            if i < 3:
                row[col_ratio] = float(1.5 + i * 0.1)  # valid for first 3 entities
            # Last 2 entities have no ratio (simulates missing IPE data)
            rows.append(row)
        ipe_path = str(tmp_path / f"interference_per_entity_{task}.json")
        with open(ipe_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)

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
        result = data["result"]
        assert result["n_valid"] == 3, (
            f"Expected n_valid=3 (3 entities with valid ratio), got {result['n_valid']}"
        )
        assert result["n_entities"] == len(ENTITIES), (
            f"n_entities must count ALL entities, got {result['n_entities']}"
        )
        assert result["n_valid"] <= result["n_entities"]

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
        import vision_unlearning.artifact as artifact_mod
        monkeypatch.setattr(rt_mod, "unlearning_algorithm_to_epochs",
                            {"people": {"distil": 400, "uce": 0, "munba": 200}})
        # Patch the cascade's HF existence check to False so InterferencePerEntity.compute()
        # falls through to _compute_from_scratch (which raises).
        monkeypatch.setattr(artifact_mod, "huggingface_dataset_file_exists", lambda *a, **kw: False)
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


class TestEmbeddingForgettingEfficiencyNValid:
    def test_n_valid_equals_n_total_when_all_ratios_present(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """n_valid must equal n_entities when all entities have a non-NaN ratio."""
        task = "people"
        method = "distil"
        epochs = 400
        # All ENTITIES have a valid ratio (no missing values)
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
        result = data["result"]
        assert result["n_valid"] == result["n_entities"], (
            f"n_valid ({result['n_valid']}) must equal n_entities ({result['n_entities']}) "
            "when all ratios are present"
        )


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
