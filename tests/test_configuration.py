"""Tests for the declarative registries in `configuration.py`.

These lock the `type_*` Literals (the static, mypy-checked definitions) against the
declarative registries (`MP_REGISTRY`/`S_REGISTRY`/`L_REGISTRY`/`ALGORITHM_REGISTRY`) and the
combinatorial `generate_me_names()` -- the two representations must never drift, since a
missing/extra entry in either would previously fail silently (a `ValueError` at call time, not
an import-time error). All tests are CPU-only and require no GPU, no network, and no real
data files.
"""
from __future__ import annotations

from typing import get_args

import vision_unlearning.benchmarks.I_care.configuration as config


class TestRegistryCompletenessAgainstLiterals:
    """Every `type_*` Literal member must have exactly one registry entry, and vice versa."""

    def test_mp_registry_matches_type_mp(self) -> None:
        assert set(config.MP_REGISTRY.keys()) == set(get_args(config.type_mp))

    def test_s_registry_matches_type_s(self) -> None:
        assert set(config.S_REGISTRY.keys()) == set(get_args(config.type_s))

    def test_l_registry_matches_type_l(self) -> None:
        assert set(config.L_REGISTRY.keys()) == set(get_args(config.type_l))

    def test_algorithm_registry_matches_type_unlearning_algorithm(self) -> None:
        assert set(config.ALGORITHM_REGISTRY.keys()) == set(get_args(config.type_unlearning_algorithm))

    def test_mp_display_order_matches_registry_keys(self) -> None:
        assert set(config._MP_DISPLAY_ORDER) == set(config.MP_REGISTRY.keys())

    def test_s_display_order_is_a_subset_of_registry_keys_with_display_names(self) -> None:
        gui_visible = {k for k, v in config.S_REGISTRY.items() if v.name_pretty is not None}
        assert set(config._S_DISPLAY_ORDER) == gui_visible

    def test_l_display_order_matches_registry_keys(self) -> None:
        assert set(config._L_DISPLAY_ORDER) == set(config.L_REGISTRY.keys())

    def test_unlearning_algorithm_display_order_matches_registry_keys(self) -> None:
        assert set(config._UNLEARNING_ALGORITHM_DISPLAY_ORDER) == set(config.ALGORITHM_REGISTRY.keys())


class TestDerivedValuesMatchRegistries:
    """`domain_*` / `*_to_direction` / `GUI_TO_BACKEND` are derived -- spot-check the derivation
    logic itself (byte-identical-to-pre-refactor is verified separately, ad hoc, against git
    history; this test protects the derivation going forward)."""

    def test_mp_to_direction_matches_registry(self) -> None:
        assert config.mp_to_direction == {k: v.direction for k, v in config.MP_REGISTRY.items()}

    def test_s_to_direction_matches_registry(self) -> None:
        assert config.s_to_direction == {k: v.direction for k, v in config.S_REGISTRY.items()}

    def test_domain_mp_pretty_names_come_from_registry(self) -> None:
        for pretty_name in config.domain_mp:
            assert pretty_name in {v.name_pretty for v in config.MP_REGISTRY.values()}

    def test_gui_to_backend_unlearning_algorithm_is_inverse_of_registry_pretty_names(self) -> None:
        for backend_name, spec in config.ALGORITHM_REGISTRY.items():
            assert config.GUI_TO_BACKEND["unlearning_algorithm"][spec.name_pretty] == backend_name

    def test_weight_overlap_has_no_display_name_and_is_excluded_from_gui(self) -> None:
        assert config.S_REGISTRY["weight_overlap"].name_pretty is None
        assert "weight_overlap" not in config.GUI_TO_BACKEND["similarity_metric"].values()


class TestGenerateMeNames:
    """`generate_me_names()` must reproduce exactly the current `type_me` vocabulary."""

    def test_generated_set_matches_type_me_literal(self) -> None:
        assert set(config.generate_me_names()) == set(get_args(config.type_me))

    def test_no_duplicate_names_generated(self) -> None:
        names = config.generate_me_names()
        assert len(names) == len(set(names))

    def test_generated_set_matches_domain_me(self) -> None:
        assert set(config.generate_me_names()) == set(config.domain_me)

    def test_only_clip_diff_has_worse_than_zero_variant(self) -> None:
        names = config.generate_me_names()
        worse_than_zero = [n for n in names if "worse than zero" in n]
        # One per role (Emitter / Receiver / Emitter minus receiver), clip_diff only.
        assert len(worse_than_zero) == 3
        assert all(n.endswith("clip diff") for n in worse_than_zero)

    def test_dino_diff_has_no_worse_than_zero_variant(self) -> None:
        names = config.generate_me_names()
        assert "Emitter number of interfered worse than zero dino diff" not in names
