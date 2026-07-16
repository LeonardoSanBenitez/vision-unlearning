"""Tests for ResultTemplateImplicitAssociationTest (IAT).

All tests are CPU-only, require no GPU, no network access, and no real data files.
Fake embedding data is written to tmp_path to keep tests hermetic.

Coverage:
- Configuration (type_l, domain_l, GUI_TO_BACKEND updated correctly)
- B matrix computation: shape, known values, NaN for empty group
- ΔB sign: positive when original > unlearned
- Permutation test: p=1 when all ΔB=0; p close to 0 when shift is large
- _compute_from_scratch: correct keys, shape, per-entity data present
- plot: returns figure with 3 axes, runs without error
- _serialize_parameters: expected string format
- Validation errors: unknown attribute, missing file, unsupported embedding type
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

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


# Small entity set used across multiple tests.
# entity name must be >=3 chars so get_target_overwrite doesn't reject them.
# For task='people', get_target_overwrite returns name-with-underscores replaced by spaces.
ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
DIM = 8

METADATA = [
    {"name": "Alice",  "gender": "F", "occupation_simplified": "Artist"},
    {"name": "Bob",    "gender": "M", "occupation_simplified": "Politician"},
    {"name": "Carol",  "gender": "F", "occupation_simplified": "Politician"},
    {"name": "Dave",   "gender": "M", "occupation_simplified": "Artist"},
    {"name": "Eve",    "gender": "F", "occupation_simplified": "Athlete"},
    {"name": "Frank",  "gender": "M", "occupation_simplified": "Athlete"},
]


def _make_embedding_file(
    entities: List[str],
    dim: int = DIM,
    seed: int = 0,
    lora_state: str = "off",
    forgotten_entity: str = "",
) -> dict:
    """Build a minimal DINOv2-style embedding file for testing.

    If lora_state='on' and forgotten_entity is set, the forgotten entity's
    embeddings are flipped (negated) to simulate a large shift.
    """
    rng = np.random.default_rng(seed)
    embeddings = []
    for entity in entities:
        for s in range(4):
            vec: List[float] = rng.standard_normal(dim).tolist()
            if lora_state == "on" and entity == forgotten_entity:
                vec = [-v for v in vec]
            embeddings.append({
                "prompted_entity": entity,
                "seed": s,
                "prompt": f"An image of {entity}",
                "embedding": vec,
            })
    return {
        "metadata": {
            "embedding_model": "dinov2_vits14",
            "lora_state": lora_state,
        },
        "embeddings": embeddings,
    }


def _write_embedding_file(
    path: str,
    entities: List[str],
    dim: int = DIM,
    seed: int = 0,
    lora_state: str = "off",
    forgotten_entity: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            _make_embedding_file(entities, dim, seed, lora_state, forgotten_entity),
            f,
        )


def _write_baseline(tmp_path: Any, task: str = "people", method: str = "distil", epochs: int = 400) -> None:
    # Canonical, method-agnostic baseline name — the only name the RT reads now that the
    # per-method fallback has been removed.
    fname = f"embeddings_{task}_original.json"
    _write_embedding_file(
        str(tmp_path / "datasets" / fname),
        ENTITIES,
    )


def _write_baseline_legacy(tmp_path: Any, task: str = "people", method: str = "distil", epochs: int = 400) -> None:
    """Write the OBSOLETE per-method baseline name, for the negative test only."""
    fname = f"embeddings_{task}_original_{method}_{epochs:03d}.json"
    _write_embedding_file(
        str(tmp_path / "datasets" / fname),
        ENTITIES,
    )


def _write_entity_files(
    tmp_path: Any,
    task: str = "people",
    method: str = "distil",
    epochs: int = 400,
    lora_state: str = "on",
) -> None:
    """Write one per-entity embedding file for each entity in ENTITIES."""
    for i, entity in enumerate(ENTITIES):
        fname = f"embeddings_{task}_{entity}_{method}_{epochs:03d}.json"
        _write_embedding_file(
            str(tmp_path / "datasets" / fname),
            ENTITIES,
            seed=i + 1,
            lora_state=lora_state,
            forgotten_entity=entity,
        )


def _make_rt(
    tmp_path: Any,
    attribute_1: str = "gender",
    attribute_2: str = "occupation_simplified",
    method: str = "distil",
    task: str = "people",
) -> vb.ResultTemplateImplicitAssociationTest:
    return vb.ResultTemplateImplicitAssociationTest(
        task=task,
        unlearning_algorithm=method,
        attribute_1=attribute_1,
        attribute_2=attribute_2,
        latent_embedding="dino_embedding",
        save_outputs=False,
        base_folder=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Section 1 — Configuration updates
# ---------------------------------------------------------------------------

class TestConfigurationUpdates:
    def test_dino_embedding_in_type_l(self) -> None:
        assert "dino_embedding" in vb.type_l.__args__

    def test_clip_embedding_still_in_type_l(self) -> None:
        # Backwards compatibility: existing code referencing clip_embedding should not break.
        assert "clip_embedding" in vb.type_l.__args__

    def test_dino_embedding_in_domain_l(self) -> None:
        assert "DINOv2 Embedding" in vb.domain_l

    def test_gui_to_backend_dino_embedding(self) -> None:
        assert vb.GUI_TO_BACKEND["latent_embedding"]["DINOv2 Embedding"] == "dino_embedding"


# ---------------------------------------------------------------------------
# Section 2 — Registry checks
# ---------------------------------------------------------------------------

class TestIATRegistry:
    def test_in_rt_name_to_class(self) -> None:
        assert "ImplicitAssociationTest" in vb.rt_name_to_class

    def test_class_is_correct(self) -> None:
        assert (
            vb.rt_name_to_class["ImplicitAssociationTest"]
            is vb.ResultTemplateImplicitAssociationTest
        )

    def test_params_in_registry(self) -> None:
        expected = [
            "model", "task", "unlearning_algorithm",
            "attribute_1", "attribute_2", "latent_embedding",
        ]
        assert vb.rt_name_to_params["ImplicitAssociationTest"] == expected

    def test_registry_consistent(self) -> None:
        assert set(vb.rt_name_to_class) == set(vb.rt_name_to_params)


# ---------------------------------------------------------------------------
# Section 3 — _serialize_parameters
# ---------------------------------------------------------------------------

class TestSerializeParameters:
    def test_basic_slug(self, tmp_path: Any) -> None:
        rt = _make_rt(tmp_path)
        s = rt._serialize_parameters()
        assert "sd1.4" in s
        assert "people" in s
        assert "distil" in s
        assert "gender" in s
        assert "occupation_simplified" in s
        assert "dino_embedding" in s

    def test_spaces_replaced(self, tmp_path: Any) -> None:
        rt = vb.ResultTemplateImplicitAssociationTest(
            task="people",
            unlearning_algorithm="distil",
            attribute_1="hpi bin",   # hypothetical name with space
            attribute_2="occ name",
            latent_embedding="dino_embedding",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        s = rt._serialize_parameters()
        assert " " not in s


# ---------------------------------------------------------------------------
# Section 4 — _compute_B_matrix (unit tests, no file I/O)
# ---------------------------------------------------------------------------

class TestComputeBMatrix:
    """Unit-test the static B-matrix helper in isolation."""

    def _unit(self, vec: List[float]) -> np.ndarray:
        """Return unit-normalized vector."""
        a = np.array(vec, dtype=float)
        return a / np.linalg.norm(a)

    def test_shape(self) -> None:
        """B matrix should have shape (|vals_1|, |vals_2|)."""
        embs = {
            "Alice": self._unit([1, 0, 0, 0]),
            "Bob":   self._unit([0, 1, 0, 0]),
            "Carol": self._unit([0, 0, 1, 0]),
            "Dave":  self._unit([0, 0, 0, 1]),
        }
        group_1 = {"F": ["Alice", "Carol"], "M": ["Bob", "Dave"]}
        group_2 = {"Artist": ["Alice", "Bob"], "Politician": ["Carol", "Dave"]}
        vals_1 = ["F", "M"]
        vals_2 = ["Artist", "Politician"]
        B = _rt_mod.ResultTemplateImplicitAssociationTest._compute_B_matrix(
            embs, group_1, group_2, vals_1, vals_2
        )
        assert B.shape == (2, 2)

    def test_diagonal_is_one_when_group_contains_single_identical_entity(self) -> None:
        """When group_i and group_j contain the same single entity, cosine sim = 1."""
        v = self._unit([1.0, 2.0, 3.0, 4.0])
        embs = {"Solo": v}
        group_1 = {"A": ["Solo"]}
        group_2 = {"B": ["Solo"]}
        vals_1 = ["A"]
        vals_2 = ["B"]
        B = _rt_mod.ResultTemplateImplicitAssociationTest._compute_B_matrix(
            embs, group_1, group_2, vals_1, vals_2
        )
        assert B[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_groups_zero(self) -> None:
        """Groups with pairwise orthogonal unit vectors → B entry ≈ 0."""
        embs = {
            "Alice": self._unit([1, 0, 0, 0]),
            "Bob":   self._unit([0, 1, 0, 0]),
        }
        group_1 = {"G1": ["Alice"]}
        group_2 = {"G2": ["Bob"]}
        vals_1 = ["G1"]
        vals_2 = ["G2"]
        B = _rt_mod.ResultTemplateImplicitAssociationTest._compute_B_matrix(
            embs, group_1, group_2, vals_1, vals_2
        )
        assert B[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_nan_when_group_has_no_embeddings(self) -> None:
        """B entry is NaN when a group contains only entities absent from entity_embeddings."""
        embs = {"Alice": self._unit([1, 0, 0, 0])}
        group_1 = {"G1": ["Alice"]}
        group_2 = {"G2": ["MissingEntity"]}  # no embedding
        vals_1 = ["G1"]
        vals_2 = ["G2"]
        B = _rt_mod.ResultTemplateImplicitAssociationTest._compute_B_matrix(
            embs, group_1, group_2, vals_1, vals_2
        )
        assert math.isnan(B[0, 0])

    def test_known_values_2x2(self) -> None:
        """Verify B values with fully controlled embeddings."""
        # Group A1 = [Alice], Group A2 = [Bob]
        # Group B1 = [Carol], Group B2 = [Dave]
        # e_alice = [1,0], e_bob = [0,1], e_carol = [1,0], e_dave = [0,1]
        embs = {
            "Alice": np.array([1.0, 0.0]),
            "Bob":   np.array([0.0, 1.0]),
            "Carol": np.array([1.0, 0.0]),
            "Dave":  np.array([0.0, 1.0]),
        }
        group_1 = {"F": ["Alice", "Bob"]}
        group_2 = {"G1": ["Carol"], "G2": ["Dave"]}
        vals_1 = ["F"]
        vals_2 = ["G1", "G2"]
        B = _rt_mod.ResultTemplateImplicitAssociationTest._compute_B_matrix(
            embs, group_1, group_2, vals_1, vals_2
        )
        # B[0,0] = mean cosine([1,0],[1,0]) + cosine([0,1],[1,0]) / 2 = (1 + 0) / 2 = 0.5
        assert B[0, 0] == pytest.approx(0.5, abs=1e-6)
        # B[0,1] = mean cosine([1,0],[0,1]) + cosine([0,1],[0,1]) / 2 = (0 + 1) / 2 = 0.5
        assert B[0, 1] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Section 5 — _permutation_test_delta_B (unit tests)
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_p_equals_one_when_delta_B_is_zero(self) -> None:
        """When all ΔB entries are zero, every permutation gives the same stat → p = 1."""
        delta_B_arr = np.zeros((5, 2, 3))
        p = _rt_mod.ResultTemplateImplicitAssociationTest._permutation_test_delta_B(
            delta_B_arr, n_permutations=100
        )
        assert p == pytest.approx(1.0, abs=0.02)

    def test_p_small_when_large_consistent_shift(self) -> None:
        """When all ΔB have a large consistent positive shift, p should be small."""
        rng = np.random.default_rng(0)
        # All entities shifted in the same positive direction
        delta_B_arr = np.ones((20, 2, 3)) * 0.5 + rng.standard_normal((20, 2, 3)) * 0.01
        p = _rt_mod.ResultTemplateImplicitAssociationTest._permutation_test_delta_B(
            delta_B_arr, n_permutations=500, rng_seed=1
        )
        # p should be very small (consistent large shift is unlikely under H0)
        assert p < 0.05

    def test_output_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(99)
        delta_B_arr = rng.standard_normal((10, 2, 2))
        p = _rt_mod.ResultTemplateImplicitAssociationTest._permutation_test_delta_B(
            delta_B_arr, n_permutations=200
        )
        assert 0.0 < p <= 1.0

    def test_reproducible_with_same_seed(self) -> None:
        rng = np.random.default_rng(7)
        arr = rng.standard_normal((8, 2, 2))
        p1 = _rt_mod.ResultTemplateImplicitAssociationTest._permutation_test_delta_B(
            arr, n_permutations=200, rng_seed=42
        )
        p2 = _rt_mod.ResultTemplateImplicitAssociationTest._permutation_test_delta_B(
            arr, n_permutations=200, rng_seed=42
        )
        assert p1 == p2


# ---------------------------------------------------------------------------
# Section 6 — _compute_from_scratch integration tests
# ---------------------------------------------------------------------------

class TestComputeFromScratch:
    """Integration tests with mocked file I/O and metadata."""

    def _setup_mocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Patch get_metadata_filtered and huggingface_dataset_file_exists."""
        monkeypatch.setattr(
            vb, "huggingface_dataset_file_exists", lambda *a, **kw: False
        )
        monkeypatch.setattr(
            _rt_mod, "get_metadata_filtered",
            lambda task, **kw: METADATA,
        )
        monkeypatch.setattr(
            vb, "get_metadata_filtered",
            lambda task, **kw: METADATA,
        )

    def test_smoke(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """compute() returns expected top-level keys."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()

        assert "metadata" in data
        assert "result" in data

        meta = data["metadata"]
        assert meta["attribute_1"] == "gender"
        assert meta["attribute_2"] == "occupation_simplified"
        assert meta["latent_embedding"] == "dino_embedding"
        assert set(meta["attribute_1_values"]) == {"F", "M"}
        assert set(meta["attribute_2_values"]) == {"Artist", "Athlete", "Politician"}

        result = data["result"]
        assert "B_original" in result
        assert "B_unlearned_mean" in result
        assert "delta_B_mean" in result
        assert "delta_B_std" in result
        assert "delta_B_per_entity" in result
        assert "n_unlearned_entities" in result
        assert "iat_effect_size" in result
        assert "permutation_pvalue" in result
        assert "significant" in result

    def test_baseline_per_method_fallback_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Negative test for the authorized compat break: with ONLY the obsolete per-method
        baseline file present (no canonical embeddings_{task}_original.json), IAT must raise.
        Proves the per-method fallback was removed here too, not only in EUP."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline_legacy(tmp_path)  # old name only
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        with pytest.raises(FileNotFoundError, match="Baseline DINOv2 embedding file not found"):
            rt._compute_from_scratch()

    def test_b_matrix_values_are_in_minus1_to_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Cosine similarities must be in [-1, 1]."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()

        for key in ("B_original", "B_unlearned_mean"):
            for row in data["result"][key].values():
                for val in row.values():
                    if val is not None:
                        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6

    def test_n_entities_matches_files_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """n_unlearned_entities should equal the number of entity files written."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()
        assert data["result"]["n_unlearned_entities"] == len(ENTITIES)

    def test_per_entity_delta_keys_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """delta_B_per_entity should have one entry per entity file present."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()
        per_entity = data["result"]["delta_B_per_entity"]
        assert set(per_entity.keys()) == set(ENTITIES)

    def test_delta_B_sign(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """When B_original == B_unlearned, mean ΔB should be ~0."""
        self._setup_mocks(monkeypatch, tmp_path)
        # Write identical baseline and entity files (same seed → same embeddings)
        _write_baseline(tmp_path)
        # Entity files with same embeddings as baseline (lora_state='off', no flip)
        _write_entity_files(tmp_path, lora_state="off")

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()
        # All ΔB entries should be close to zero
        for row in data["result"]["delta_B_mean"].values():
            for val in row.values():
                if val is not None:
                    assert abs(val) < 0.5  # allow some noise from seed differences

    def test_attribute_value_sorted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Attribute values in metadata should be sorted alphabetically."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()
        assert data["metadata"]["attribute_1_values"] == sorted(
            data["metadata"]["attribute_1_values"]
        )
        assert data["metadata"]["attribute_2_values"] == sorted(
            data["metadata"]["attribute_2_values"]
        )

    def test_missing_entity_file_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """If some entity files are missing, they are silently skipped."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        # Write only 3 of 6 entity files
        for entity in ENTITIES[:3]:
            fname = f"embeddings_people_{entity}_distil_400.json"
            _write_embedding_file(
                str(tmp_path / "datasets" / fname),
                ENTITIES,
                lora_state="on",
                forgotten_entity=entity,
            )

        rt = _make_rt(tmp_path)
        data = rt._compute_from_scratch()
        assert data["result"]["n_unlearned_entities"] == 3

    def test_missing_baseline_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """FileNotFoundError when baseline embedding file is absent."""
        self._setup_mocks(monkeypatch, tmp_path)
        # Do NOT write baseline file
        _write_entity_files(tmp_path)

        rt = _make_rt(tmp_path)
        with pytest.raises(FileNotFoundError, match="Baseline DINOv2 embedding file not found"):
            rt._compute_from_scratch()

    def test_no_entity_files_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """RuntimeError when no per-entity embedding files are found."""
        self._setup_mocks(monkeypatch, tmp_path)
        _write_baseline(tmp_path)
        # Do NOT write any entity files

        rt = _make_rt(tmp_path)
        with pytest.raises(RuntimeError, match="No per-entity embedding files found"):
            rt._compute_from_scratch()

    def test_unsupported_embedding_type_raises(self, tmp_path: Any) -> None:
        """NotImplementedError when latent_embedding='clip_embedding'."""
        rt = vb.ResultTemplateImplicitAssociationTest(
            task="people",
            unlearning_algorithm="distil",
            attribute_1="gender",
            attribute_2="occupation_simplified",
            latent_embedding="clip_embedding",
            save_outputs=False,
            base_folder=str(tmp_path),
        )
        with pytest.raises(NotImplementedError, match="clip_embedding"):
            rt._compute_from_scratch()


# ---------------------------------------------------------------------------
# Section 7 — plot tests
# ---------------------------------------------------------------------------

class TestIATPlot:
    """Test plot() without any file I/O (data dict constructed directly)."""

    def _make_data(self) -> dict:
        """Build a minimal data dict matching the expected schema."""
        vals_1 = ["F", "M"]
        vals_2 = ["Artist", "Athlete", "Politician"]

        def _nested(val: float) -> Dict[str, Dict[str, float]]:
            return {r: {c: val for c in vals_2} for r in vals_1}

        return {
            "metadata": {
                "RT": "ResultTemplateImplicitAssociationTest",
                "model": "sd1.4",
                "task": "people",
                "unlearning_algorithm": "distil",
                "attribute_1": "gender",
                "attribute_2": "occupation_simplified",
                "attribute_1_values": vals_1,
                "attribute_2_values": vals_2,
                "latent_embedding": "dino_embedding",
                "n_entities_computed": 6,
                "group_sizes_attr1": {"F": 3, "M": 3},
                "group_sizes_attr2": {"Artist": 2, "Athlete": 2, "Politician": 2},
                "significance_threshold": 0.05,
            },
            "result": {
                "B_original": _nested(0.82),
                "B_unlearned_mean": _nested(0.80),
                "delta_B_mean": _nested(0.02),
                "delta_B_std": _nested(0.01),
                "delta_B_per_entity": {},
                "n_unlearned_entities": 6,
                "iat_effect_size": 0.02,
                "permutation_pvalue": 0.12,
                "significant": False,
            },
        }

    def test_returns_figure_and_axes(self) -> None:
        data = self._make_data()
        result = vb.ResultTemplateImplicitAssociationTest.plot(data, return_fig=True)
        assert result is not None
        fig, axes = result
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)

    def test_three_axes(self) -> None:
        data = self._make_data()
        result = vb.ResultTemplateImplicitAssociationTest.plot(data, return_fig=True)
        assert result is not None
        _, axes = result
        # axes is a 1D array of 3 Axes
        assert len(axes) == 3

    def test_returns_none_when_not_requested(self) -> None:
        data = self._make_data()
        result = vb.ResultTemplateImplicitAssociationTest.plot(data, return_fig=False)
        assert result is None

    def test_significant_in_title(self) -> None:
        data = self._make_data()
        # Force significant = True
        data["result"]["permutation_pvalue"] = 0.01
        data["result"]["significant"] = True
        result = vb.ResultTemplateImplicitAssociationTest.plot(data, return_fig=True)
        assert result is not None
        fig, axes = result
        title = axes[2].get_title()
        assert "significant" in title.lower()
