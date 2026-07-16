"""Characterization tests for the base-model filename dimension.

These lock two properties of ``model_segment`` and every asset path/artifact threaded
with a ``model`` parameter:

1. **Byte-identical for ``sd1.4``.** The default (and only produced) base model adds no
   segment, so every existing asset name is unchanged. This is what guarantees the
   feasibility-demonstration assets are never renamed.
2. **Disambiguating for a non-default model.** A hypothetical second base model (``sdxl``
   here) inserts a ``_sdxl`` segment, so its assets never collide with the ``sd1.4`` ones.
   ``type_model`` currently has only ``sd1.4``; the non-default cases use ``cast`` to
   exercise the interface-ready branch without adding an unproduced model to the enum.

The activation-fingerprint files keep their own explicit-model convention
(``act_fingerprints_{task}_{model}.json``) and are intentionally NOT covered here.
"""
from typing import cast

from vision_unlearning.benchmarks.I_care.configuration import model_segment, type_model
from vision_unlearning.benchmarks.I_care.metadata import (
    BaselineEmbeddings,
    EntityEmbeddings,
    InterferencePerEntity,
    InterferencePerPair,
    _interference_per_pair_filename,
    get_embedding_hf_path,
    get_embedding_output_path,
    get_interference_per_entity_path,
)
from vision_unlearning.benchmarks.I_care.similarity import Similarity
from vision_unlearning.datasets.testbed import (
    get_generated_dataset_folder,
    get_shared_baseline_folder,
    get_unlearned_model_folder,
)

_SDXL = cast(type_model, "sdxl")  # interface-ready second model; not in type_model yet


# ---------------------------------------------------------------------------
# The segment helper itself
# ---------------------------------------------------------------------------

def test_model_segment_default_is_empty() -> None:
    assert model_segment("sd1.4") == ""


def test_model_segment_non_default_disambiguates() -> None:
    assert model_segment(_SDXL) == "_sdxl"


# ---------------------------------------------------------------------------
# sd1.4 is byte-identical (exact expected strings) — embeddings + interference
# ---------------------------------------------------------------------------

def test_embedding_paths_byte_identical_for_sd14() -> None:
    assert get_embedding_output_path("people", "brad pitt", "uce", 0).endswith(
        "embeddings_people_brad pitt_uce_000.json"
    )
    assert get_embedding_hf_path("people", "brad pitt", "uce", 0) == (
        "datasets/embeddings_people_brad pitt_uce_000.json"
    )
    assert BaselineEmbeddings(task="people")._get_data_path_remote() == (
        "datasets/embeddings_people_original.json"
    )
    # EntityEmbeddings epoch count comes from the executed-combinations table; assert the
    # sd1.4 name carries no model token rather than hardcoding the epoch value.
    entity_sd14 = EntityEmbeddings(
        task="people", hf_entity="brad pitt", unlearning_algorithm="uce"
    )._get_data_path_remote()
    assert entity_sd14.startswith("datasets/embeddings_people_brad pitt_uce_")
    assert "sdxl" not in entity_sd14 and "sd1.4" not in entity_sd14


def test_interference_paths_byte_identical_for_sd14() -> None:
    assert _interference_per_pair_filename("people", 5, "uce", 0) == (
        "interferences_caused_by_people_5_uce_0.json"
    )
    assert InterferencePerPair(
        task="people", index=5, method="uce", num_train_epochs=0
    )._get_data_path_remote() == "datasets/interferences_caused_by_people_5_uce_0.json"
    assert get_interference_per_entity_path("people").endswith("interference_per_entity_people.json")
    assert InterferencePerEntity(task="people")._get_data_path_remote() == (
        "interference_per_entity_people.json"
    )


def test_similarity_path_byte_identical_for_sd14() -> None:
    assert Similarity(task="people", similarity_metric="clip")._get_data_path_remote() == (
        "similarity_clip_people.json"
    )


def test_generated_and_model_folders_byte_identical_for_sd14() -> None:
    assert get_generated_dataset_folder("breeds", "uce", 0, "poodle").endswith(
        "generated_breeds_poodle_uce_000"
    )
    assert get_shared_baseline_folder("scenes").endswith("generated_scenes_baseline")
    assert get_unlearned_model_folder("people", "distil", 400, "brad pitt").endswith(
        "people_brad pitt_distil_400"
    )


# ---------------------------------------------------------------------------
# A non-default model inserts the disambiguating segment at the right place
# ---------------------------------------------------------------------------

def test_non_default_model_inserts_segment() -> None:
    # The plain path helpers do no runtime Literal validation, so they can exercise the
    # non-default branch directly. Files: segment lands before the extension; folders: it is
    # appended to the folder name.
    assert get_embedding_output_path("people", "brad pitt", "uce", 0, model=_SDXL).endswith(
        "embeddings_people_brad pitt_uce_000_sdxl.json"
    )
    assert get_interference_per_entity_path("people", model=_SDXL).endswith(
        "interference_per_entity_people_sdxl.json"
    )
    assert _interference_per_pair_filename("people", 5, "uce", 0, _SDXL) == (
        "interferences_caused_by_people_5_uce_0_sdxl.json"
    )
    assert get_generated_dataset_folder("breeds", "uce", 0, "poodle", model=_SDXL).endswith(
        "generated_breeds_poodle_uce_000_sdxl"
    )
    assert get_shared_baseline_folder("scenes", model=_SDXL).endswith("generated_scenes_baseline_sdxl")
    assert get_unlearned_model_folder("people", "distil", 400, "brad pitt", model=_SDXL).endswith(
        "people_brad pitt_distil_400_sdxl"
    )


def test_non_default_model_never_collides_with_sd14() -> None:
    # The whole point: an sdxl asset name is never equal to the sd1.4 one.
    assert (
        get_generated_dataset_folder("breeds", "uce", 0, "poodle", model=_SDXL)
        != get_generated_dataset_folder("breeds", "uce", 0, "poodle")
    )
    assert (
        get_embedding_output_path("people", "brad pitt", "uce", 0, model=_SDXL)
        != get_embedding_output_path("people", "brad pitt", "uce", 0)
    )


def test_artifact_validates_model_against_type_model() -> None:
    # The artifacts validate ``model`` against ``type_model`` at construction, so an
    # unproduced base model cannot be built until it is added to the enum. The naming is
    # ready for a second model, but the type stays honest — the segment branch for artifacts
    # becomes reachable only once ``type_model`` gains that value. (The cast above only
    # satisfies mypy; pydantic enforces the Literal at runtime.)
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BaselineEmbeddings(task="people", model=_SDXL)
