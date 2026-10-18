"""Regression tests for the u-care benchmark configuration and artifacts."""
from __future__ import annotations

import json
import os
from typing import Any, List

import pytest

import vision_unlearning.benchmarks.u_care.configuration as cfg
import vision_unlearning.benchmarks.u_care.generated_dataset as generated_dataset_mod
import vision_unlearning.benchmarks.u_care.metadata as metadata_mod
import vision_unlearning.benchmarks.u_care.result_templates as result_templates_mod
from vision_unlearning.benchmarks.u_care.pipeline_06_compute_interference_per_pair import (
    receiver_image_filenames,
)
from vision_unlearning.benchmarks.u_care.pipeline_07_compute_interference_per_entity import (
    aggregate_entity_metrics,
)
from vision_unlearning.benchmarks.u_care.pipeline_03_unlearn_model import (
    build_uce_command,
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

    def test_uce_command_uses_upstream_arguments(self) -> None:
        command = build_uce_command(
            upstream_script="train_erase.py",
            checkpoint="model",
            emitter="Van_Gogh",
            output_folder="output",
            erase_scale=0.05,
            lamb=1.0,
            guided_concept="An image in Photo style",
            python_executable="python",
        )
        assert command == [
            "python", "train_erase.py", "--ckpt", "model", "--theme", "Van_Gogh",
            "--output_dir", "output", "--erase_scale", "0.05", "--lamb", "1.0",
            "--guided_concepts", "An image in Photo style", "--add_prompts",
        ]


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


def _write_pair_file(folder: Any, emitter: str, accuracy_by_receiver: dict) -> None:
    """One emitter's per-pair file, in the layout pipeline_06 writes."""
    row = {
        receiver: {
            "accuracy": accuracy_by_receiver.get(receiver, 1.0),
            "accuracy_diff": 0.0,
            "target_probability": 0.5,
            "target_probability_diff": 0.0,
        }
        for receiver in cfg.ENTITIES
    }
    path = os.path.join(str(folder), f"interferences_caused_by_{emitter}_uce_sd_style50.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle)


def _write_per_entity_file(folder: Any, rows: List[dict]) -> None:
    """The per-entity file, in the layout pipeline_07 writes."""
    path = os.path.join(str(folder), "interference_per_entity_sd_style50.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def _per_entity_row(name: str, ua: float, ira: float, cra: float) -> dict:
    return {
        "name": name, "index": cfg.ENTITIES.index(name), "domain": cfg.entity_domain(name), "unlearnable": True,
        "metric_uce_Unlearning accuracy": ua,
        "metric_uce_In domain retain accuracy": ira,
        "metric_uce_Cross domain retain accuracy": cra,
    }


class TestResultTemplateInterferenceMatrix:
    def test_rows_are_all_emitters_and_columns_all_receivers(self, tmp_path: Any) -> None:
        _write_pair_file(tmp_path, "Van_Gogh", {"Van_Gogh": 0.0, "Monet": 0.8})
        rt = result_templates_mod.ResultTemplateInterferenceMatrix(
            base_folder=str(tmp_path), unlearning_algorithm="uce", interference_pair="accuracy")
        data = rt._compute_from_scratch()
        rt._validate(data)
        rows = data["result"]
        assert [row["emitter"] for row in rows] == cfg.UNLEARNABLE_ENTITIES
        assert all(list(row)[1:] == cfg.ENTITIES for row in rows)
        van_gogh = rows[cfg.UNLEARNABLE_ENTITIES.index("Van_Gogh")]
        assert van_gogh["Van_Gogh"] == 0.0
        assert van_gogh["Monet"] == 0.8
        assert van_gogh["Cats"] == 1.0

    def test_missing_emitter_is_an_empty_row_and_is_counted(self, tmp_path: Any) -> None:
        _write_pair_file(tmp_path, "Van_Gogh", {})
        _write_pair_file(tmp_path, "Bears", {})
        data = result_templates_mod.ResultTemplateInterferenceMatrix(
            base_folder=str(tmp_path), unlearning_algorithm="uce", interference_pair="accuracy",
        )._compute_from_scratch()
        assert data["metadata"]["n_emitters_present"] == 2
        assert data["metadata"]["n_emitters_total"] == 70
        cats = data["result"][cfg.UNLEARNABLE_ENTITIES.index("Cats")]
        assert all(cats[receiver] is None for receiver in cfg.ENTITIES)

    def test_plot_draws_only_the_present_rows(self, tmp_path: Any) -> None:
        _write_pair_file(tmp_path, "Van_Gogh", {})
        _write_pair_file(tmp_path, "Bears", {})
        data = result_templates_mod.ResultTemplateInterferenceMatrix(
            base_folder=str(tmp_path), unlearning_algorithm="uce", interference_pair="accuracy",
        )._compute_from_scratch()
        result = result_templates_mod.ResultTemplateInterferenceMatrix.plot(data, return_fig=True)
        assert result is not None
        fig, ax = result
        assert ax.images[0].get_array().shape == (2, 71)
        assert "n=2/70" in ax.get_title()

    def test_plot_colour_range_is_the_data_by_default_and_fixed_when_given(self, tmp_path: Any) -> None:
        _write_pair_file(tmp_path, "Van_Gogh", {})
        data = result_templates_mod.ResultTemplateInterferenceMatrix(
            base_folder=str(tmp_path), unlearning_algorithm="uce", interference_pair="accuracy",
        )._compute_from_scratch()
        plot = result_templates_mod.ResultTemplateInterferenceMatrix.plot
        default = plot(data, return_fig=True)
        fixed = plot(data, return_fig=True, vmin=-1.0, vmax=1.0)
        assert default is not None and fixed is not None
        assert default[1].images[0].get_clim() != (-1.0, 1.0)
        assert fixed[1].images[0].get_clim() == (-1.0, 1.0)

    def test_remote_path_names_the_parameters(self) -> None:
        rt = result_templates_mod.ResultTemplateInterferenceMatrix(unlearning_algorithm="uce", interference_pair="accuracy")
        assert rt._get_data_path_remote() == "results/InterferenceMatrix/sd_style50_unlearncanvas_uce_accuracy.json"


class TestResultTemplateBenchmarkSummary:
    def test_means_over_the_domain_emitters(self, tmp_path: Any) -> None:
        _write_per_entity_file(tmp_path, [
            _per_entity_row("Van_Gogh", 1.0, 0.9, 0.8),
            _per_entity_row("Monet", 0.5, 0.7, 0.6),
            _per_entity_row("Bears", 0.25, 0.5, 0.4),
        ])
        rt = result_templates_mod.ResultTemplateBenchmarkSummary(
            base_folder=str(tmp_path), unlearning_algorithm_list=["uce"], domain="style")
        data = rt._compute_from_scratch()
        rt._validate(data)
        (row,) = data["result"]
        assert row["method"] == "uce"
        assert row["n_emitters"] == 2
        assert row["Unlearning accuracy"] == pytest.approx(0.75)
        assert row["In domain retain accuracy"] == pytest.approx(0.8)
        assert row["Cross domain retain accuracy"] == pytest.approx(0.7)

    def test_a_metric_no_row_carries_stays_missing(self, tmp_path: Any) -> None:
        _write_per_entity_file(tmp_path, [_per_entity_row("Bears", 0.25, 0.5, 0.4)])
        data = result_templates_mod.ResultTemplateBenchmarkSummary(
            base_folder=str(tmp_path), unlearning_algorithm_list=["uce", "esd"], domain="object",
        )._compute_from_scratch()
        uce, esd = data["result"]
        assert uce["Frechet inception distance"] is None
        assert uce["Runtime seconds"] is None
        assert esd["n_emitters"] == 0
        assert esd["Unlearning accuracy"] is None

    def test_plot_renders_one_table_row_per_method(self, tmp_path: Any) -> None:
        _write_per_entity_file(tmp_path, [_per_entity_row("Bears", 0.25, 0.5, 0.4)])
        data = result_templates_mod.ResultTemplateBenchmarkSummary(
            base_folder=str(tmp_path), unlearning_algorithm_list=["uce", "esd"], domain="object",
        )._compute_from_scratch()
        result = result_templates_mod.ResultTemplateBenchmarkSummary.plot(data, return_fig=True)
        assert result is not None
        fig, ax = result
        texts = [cell.get_text().get_text() for cell in ax.tables[0].get_celld().values()]
        assert "0.25" in texts
        assert "UCE" in texts and "ESD" in texts

    def test_remote_path_names_the_parameters(self) -> None:
        rt = result_templates_mod.ResultTemplateBenchmarkSummary(unlearning_algorithm_list=["uce", "esd"], domain="style")
        assert rt._get_data_path_remote() == "results/BenchmarkSummary/sd_style50_unlearncanvas_uce-esd_style.json"
