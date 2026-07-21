"""Tests for the I-CARE state-of-the-world census (``benchmarks/I_care/state.py``).

Hermetic and torch-free: the census answers presence/validity through the artifacts'
non-resolving audit API (``exists_local`` / ``is_in_listing`` / ``validate_local``), so no
test here touches the real network. The one networking entry point
(``refresh_remote_listing``) is exercised with a monkeypatched lister.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set
from typing import get_args

import pytest

import vision_unlearning.benchmarks.I_care.state as state
from vision_unlearning.benchmarks.I_care import configuration as cfg
from vision_unlearning.benchmarks.I_care.metadata import EntityEmbeddings
from vision_unlearning.datasets.testbed import MetadataFiltered


def _write_json(path: str, content: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle)


def _write_metadata(base_folder: str, task: str, names: List[str]) -> None:
    art = MetadataFiltered(task=task, base_folder=base_folder)  # type: ignore[arg-type]
    _write_json(art._get_data_path_local(), [{"name": n} for n in names])


# ---------------------------------------------------------------------------
# The grid is locked to the declared ontology (the "cannot drift" claim)
# ---------------------------------------------------------------------------
def test_constants_match_configuration() -> None:
    assert set(state.TASKS) == set(get_args(cfg.type_task))
    assert set(state.METHODS) == set(get_args(cfg.type_unlearning_algorithm))
    # SIMILARITY_METRICS is exactly the GUI-selectable similarity metrics: every type_s value
    # except the scenes/distil-only diagnostic weight_overlap.
    assert set(state.SIMILARITY_METRICS) == set(get_args(cfg.type_s)) - {"weight_overlap"}


def test_expected_grid_size_is_generated_from_metadata(tmp_path: Any) -> None:
    base = str(tmp_path)
    # 3 entities per task, all three tasks present -> the full grid is enumerable.
    for task in state.TASKS:
        _write_metadata(base, task, [f"{task}_entity_{i}" for i in range(3)])

    expected = state.expected_artifacts(base_folder=base)
    # per-task rows: metadata + baseline_embeddings + interference_per_entity + 4 similarity = 7
    per_task = 3 * 7
    # per-entity rows: (entity_embeddings + interference_per_pair) x 3 methods x 3 entities
    per_entity = 3 * (2 * 3 * 3)
    assert len(expected) == per_task + per_entity

    kinds = {e.kind for e in expected}
    assert kinds == {
        "metadata", "baseline_embeddings", "interference_per_entity", "similarity",
        "entity_embeddings", "interference_per_pair",
    }


def test_missing_metadata_skips_per_entity_grid_but_reports_metadata_gap(tmp_path: Any) -> None:
    base = str(tmp_path)
    _write_metadata(base, "people", ["a_person", "b_person"])
    # scenes/breeds metadata deliberately absent.
    expected = state.expected_artifacts(base_folder=base)

    per_entity_tasks = {e.task for e in expected if e.kind == "entity_embeddings"}
    assert per_entity_tasks == {"people"}  # only the task with metadata is enumerable
    # the per-task metadata rows still exist for every task, so the gap is visible
    metadata_tasks = {e.task for e in expected if e.kind == "metadata"}
    assert metadata_tasks == set(state.TASKS)


# ---------------------------------------------------------------------------
# Local / remote presence
# ---------------------------------------------------------------------------
def test_local_presence_reflects_files_on_disk(tmp_path: Any) -> None:
    base = str(tmp_path)
    _write_metadata(base, "people", ["a_person", "b_person"])
    rows = state.build_census(base_folder=base)

    # metadata_people is present (we wrote it); an entity embedding is not.
    metadata_row = next(r for r in rows if r.kind == "metadata" and r.task == "people")
    embedding_row = next(r for r in rows if r.kind == "entity_embeddings")
    assert metadata_row.present_local is True
    assert embedding_row.present_local is False
    assert metadata_row.present_remote is None  # no listing supplied

    # write the embedding file at its own declared path -> now present.
    _write_json(embedding_row.local_path, {"embeddings": []})
    rows2 = state.build_census(base_folder=base)
    embedding_row2 = next(r for r in rows2 if r.local_path == embedding_row.local_path)
    assert embedding_row2.present_local is True


def test_remote_presence_uses_the_listing_not_the_network(tmp_path: Any) -> None:
    base = str(tmp_path)
    _write_metadata(base, "people", ["a_person"])
    rows_paths = state.build_census(base_folder=base)
    # Claim that exactly the similarity remote paths exist on the remote.
    listing: Set[str] = {r.remote_path for r in rows_paths if r.kind == "similarity"}

    rows = state.build_census(base_folder=base, remote_listing=listing)
    for row in rows:
        if row.kind == "similarity":
            assert row.present_remote is True
        else:
            assert row.present_remote is False


# ---------------------------------------------------------------------------
# Validation — the only thing that distinguishes "present" from "valid"
# ---------------------------------------------------------------------------
def test_validate_flags_present_but_invalid_files(tmp_path: Any) -> None:
    base = str(tmp_path)
    _write_metadata(base, "people", ["a_person", "b_person"])
    rows = state.build_census(base_folder=base)
    embedding_rows = [r for r in rows if r.kind == "entity_embeddings"]
    good, bad = embedding_rows[0], embedding_rows[1]

    _write_json(good.local_path, {"embeddings": [{"prompt": "x", "embedding": [0.0]}]})
    _write_json(bad.local_path, {"not_embeddings": True})  # EntityEmbeddings._validate requires 'embeddings'

    validated = state.build_census(base_folder=base, validate=True)
    good_v = next(r for r in validated if r.local_path == good.local_path)
    bad_v = next(r for r in validated if r.local_path == bad.local_path)
    assert good_v.valid is True
    assert bad_v.valid is False
    assert bad_v.validation_error is not None
    # an absent file is neither valid nor invalid
    absent = next(r for r in validated if r.kind == "entity_embeddings" and not r.present_local)
    assert absent.valid is None


def test_audit_methods_are_bools_and_touch_no_network(tmp_path: Any) -> None:
    base = str(tmp_path)
    art = EntityEmbeddings(
        task="people", hf_entity="A Person", unlearning_algorithm="distil", base_folder=base,
    )
    assert art.exists_local() is False
    assert art.is_in_listing(set()) is False
    assert art.is_in_listing({art._get_data_path_remote()}) is True

    _write_json(art._get_data_path_local(), {"embeddings": []})
    assert art.exists_local() is True
    art.validate_local()  # valid: 'embeddings' present -> no raise

    _write_json(art._get_data_path_local(), {"nope": 1})
    with pytest.raises(AssertionError):
        art.validate_local()


# ---------------------------------------------------------------------------
# Remote listing cache
# ---------------------------------------------------------------------------
def test_refresh_and_load_remote_listing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    base = str(tmp_path)
    fake_files = ["datasets/embeddings_people_original.json", "similarity_clip_people.json"]
    monkeypatch.setattr(state, "huggingface_dataset_list_files", lambda repo, token: list(fake_files))

    cache_path = state.refresh_remote_listing(base_folder=base)
    assert os.path.exists(cache_path)

    listing, date = state.load_cached_listing(base_folder=base)
    assert listing == set(fake_files)
    assert date != "unknown"


def test_load_cached_listing_missing_raises(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError):
        state.load_cached_listing(base_folder=str(tmp_path))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _small_census(tmp_path: Any) -> List[state.ArtifactState]:
    base = str(tmp_path)
    _write_metadata(base, "people", ["a_person"])
    return state.build_census(base_folder=base)


def test_summarize_text_reports_totals(tmp_path: Any) -> None:
    text = state.summarize_text(_small_census(tmp_path), remote_known=False, validated=False)
    assert "TOTAL expected=" in text
    assert "kind" in text and "local" in text


def test_generate_state_md_has_pointers_and_gaps(tmp_path: Any) -> None:
    rows = _small_census(tmp_path)
    md = state.generate_state_md(
        rows, remote_known=False, validated=False, listing_date=None, base_folder="assets",
    )
    assert md.startswith("# I-CARE — state of the world")
    assert "Known gaps" in md
    assert "python state.py state-md" in md


def test_filter_rows_missing_local(tmp_path: Any) -> None:
    rows = _small_census(tmp_path)
    missing = state.filter_rows(rows, kind="entity_embeddings", missing_local=True)
    assert missing  # nothing was written for entity embeddings
    assert all(r.kind == "entity_embeddings" and not r.present_local for r in missing)
