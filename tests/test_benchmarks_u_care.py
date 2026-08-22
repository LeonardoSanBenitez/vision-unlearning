"""Regression tests for the u-care benchmark configuration and artifacts."""
from __future__ import annotations

import os
from typing import Any, List

import pytest

import vision_unlearning.benchmarks.u_care.configuration as cfg
import vision_unlearning.benchmarks.u_care.generated_dataset as generated_dataset_mod
import vision_unlearning.benchmarks.u_care.metadata as metadata_mod
from vision_unlearning.benchmarks.u_care.pipeline_06_compute_interference_per_pair import (
    receiver_image_filenames,
)
from vision_unlearning.benchmarks.u_care.pipeline_07_compute_interference_per_entity import (
    aggregate_entity_metrics,
)


class TestConfigurationInvariants:
    def test_entity_lists_are_consistent_and_disjoint(self) -> None:
        assert len(cfg.STYLE_ENTITIES) == 51
        assert len(cfg.OBJECT_ENTITIES) == 20
        assert len(cfg.ENTITIES) == 71
        assert set(cfg.STYLE_ENTITIES).isdisjoint(cfg.OBJECT_ENTITIES)
        assert set(cfg.ENTITIES) == set(cfg.STYLE_ENTITIES + cfg.OBJECT_ENTITIES)

    def test_unlearnable_logic_only_excludes_seed_images(self) -> None:
        assert cfg.is_unlearnable("Cats") is True
        assert cfg.is_unlearnable("Seed_Images") is False
        assert len(cfg.UNLEARNABLE_ENTITIES) == 70
        assert set(cfg.UNLEARNABLE_ENTITIES) == set(cfg.ENTITIES) - {"Seed_Images"}

    def test_registry_keys_match_the_declared_literals(self) -> None:
        assert set(cfg.MP_REGISTRY.keys()) == {"accuracy", "accuracy_diff", "target_probability", "target_probability_diff"}
        assert set(cfg.ALGORITHM_REGISTRY.keys()) == {"ca", "ediff", "esd", "fmn", "salun", "seot", "shs", "spm", "uce"}

    def test_answer_set_prompt_and_model_segment(self) -> None:
        assert cfg.answer_set_prompt("Van_Gogh", "Cats") == "A Cats image in Van Gogh style."
        assert cfg.model_segment("sd_style50") == "_sd_style50"


class TestGeneratedDatasetValidator:
    def test_baseline_and_emitter_require_both_or_neither(self) -> None:
        generated_dataset_mod.GeneratedDataset(base_folder="assets", emitter=None, method=None)

        with pytest.raises(ValueError, match="both be set"):
            generated_dataset_mod.GeneratedDataset(base_folder="assets", emitter="Cats")

        with pytest.raises(ValueError, match="both be set"):
            generated_dataset_mod.GeneratedDataset(base_folder="assets", method="uce")


class TestStage2EvaluationLogic:
    def test_receiver_slices_have_expected_grid_sizes(self) -> None:
        assert len(receiver_image_filenames("Van_Gogh", [188], "off")) == 20
        assert len(receiver_image_filenames("Cats", [188], "off")) == 51
        assert receiver_image_filenames("Van_Gogh", [188], "off")[0].startswith(
            "off_188_A Architectures image in Van Gogh style..png"
        )

    def test_entity_aggregates_use_receiver_domains(self) -> None:
        pair_metrics = {
            receiver: {"accuracy": 1.0 if receiver != "Van_Gogh" else 0.25}
            for receiver in cfg.ENTITIES
        }
        result = aggregate_entity_metrics("Van_Gogh", pair_metrics)
        assert result["Unlearning accuracy"] == 0.75
        assert result["In domain retain accuracy"] == 1.0
        assert result["Cross domain retain accuracy"] == 1.0


class TestGeneratedDatasetLifecycle:
    def test_exists_detects_complete_dataset_folder(self, tmp_path: Any) -> None:
        dataset = generated_dataset_mod.GeneratedDataset(base_folder=str(tmp_path), emitter=None, method=None)
        dataset_folder = os.path.join(str(tmp_path), "datasets", "generated_baseline_sd_style50")
        os.makedirs(dataset_folder, exist_ok=True)
        for seed in [1, 2]:
            for prompt in ["Cats", "Dogs"]:
                with open(os.path.join(dataset_folder, f"off_{seed:02}_{prompt}.png"), "w", encoding="utf-8") as handle:
                    handle.write("x")
        with open(os.path.join(dataset_folder, "metadata.jsonl"), "w", encoding="utf-8") as handle:
            handle.write("{}\n")

        assert dataset.exists([1, 2], ["Cats", "Dogs"]) is True
        assert dataset.exists([1, 2], ["Cats", "Dogs", "Birds"]) is False

    def test_produce_from_scratch_raises_when_pipeline_is_unavailable(self, tmp_path: Any) -> None:
        dataset = generated_dataset_mod.GeneratedDataset(base_folder=str(tmp_path), emitter=None, method=None)
        dataset._pending_seeds = [1]
        dataset._pending_prompts = ["Cats"]
        with pytest.raises(generated_dataset_mod.ArtifactNotAvailableError, match="cannot be produced"):
            dataset._produce_from_scratch()


class TestMetadataArtifacts:
    def test_entity_metadata_uses_expected_remote_path(self) -> None:
        artifact = metadata_mod.EntityMetadata()
        assert artifact._get_data_path_remote() == "metadata_filtered.json"

    def test_interference_artifacts_use_entity_indexed_remote_paths(self) -> None:
        artifact = metadata_mod.InterferencePerPair(emitter="Cats", method="uce")
        assert artifact._get_data_path_remote() == "datasets/interferences_caused_by_55_uce_sd_style50.json"

    def test_baseline_accuracy_uses_expected_remote_path(self) -> None:
        artifact = metadata_mod.BaselineAccuracy()
        assert artifact._get_data_path_remote() == "datasets/accuracies_original.json"
