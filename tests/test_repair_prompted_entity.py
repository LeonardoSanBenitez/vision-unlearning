"""Tests for the one-off repair of ``prompted_entity`` in the embedding files (no network, no torch).

The utility rewrites a single field in files that are not under version control and cost GPU-days to
reproduce, so the tests here are about the things that protect that payload rather than about the
string rule itself, which ``test_entity_names.py`` owns:

* the new value is derived from the record's own ``prompt``, never from a lookup;
* a semantic digest covers everything the repair must not touch, and notices when it moves;
* a validation failure leaves the original file byte-identical;
* running it twice changes nothing the second time;
* a run resumed from a partial manifest does not touch a file the manifest already validated.
"""
import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import repair_prompted_entity as rpe


_SCENES_PROMPTS = ["An image of an abbey scene", "An image of a badlands scene"]


def _record(prompt: str, prompted_entity: str, seed: int, embedding: List[float]) -> Dict[str, Any]:
    return {"prompted_entity": prompted_entity, "seed": seed, "prompt": prompt, "embedding": embedding}


def _broken_payload() -> Dict[str, Any]:
    """A miniature of a real scenes embedding file: the doubled article and the doubled suffix."""
    return {
        "metadata": {
            "task": "scenes",
            "forgotten_entity": "an abbey scene",
            "method": "distil",
            "num_train_epochs": 100,
            "lora_state": "on",
            "embedding_model": "dinov2_vits14",
            "embedding_dim": 384,
        },
        "embeddings": [
            _record(_SCENES_PROMPTS[0], "an an abbey scene scene", 42, [0.1, 0.2]),
            _record(_SCENES_PROMPTS[1], "an a badlands scene scene", 42, [0.3, 0.4]),
            _record(_SCENES_PROMPTS[0], "an an abbey scene scene", 43, [0.5, 0.6]),
        ],
    }


def _write(path: str, payload: Dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        result: Dict[str, Any] = json.load(handle)
    return result


class TestDerivation:
    def test_new_value_comes_from_the_records_own_prompt(self) -> None:
        assert rpe.prompted_entity_from_prompt("scenes", "An image of an abbey scene") == "an abbey scene"
        assert rpe.prompted_entity_from_prompt("breeds", "An image of a basenji dog") == "a basenji dog"
        assert rpe.prompted_entity_from_prompt("people", "An image of Colin Powell") == "Colin Powell"

    def test_a_prompt_that_is_not_the_generation_form_is_refused(self) -> None:
        """The repair may only run where the prompt is the string the corpus was generated with."""
        with pytest.raises(ValueError):
            rpe.prompted_entity_from_prompt("scenes", "a photo of an abbey scene")

    def test_a_prompt_whose_body_is_not_canonical_is_refused(self) -> None:
        """``An image of abbey`` is missing the article and the suffix: the file is not repairable,
        because the field cannot be derived from a prompt that is itself wrong."""
        with pytest.raises(ValueError):
            rpe.prompted_entity_from_prompt("scenes", "An image of abbey")


class TestSemanticDigest:
    def test_the_field_under_repair_is_outside_the_digest(self) -> None:
        payload = _broken_payload()
        before = rpe.semantic_digest(payload)
        for record in payload["embeddings"]:
            record["prompted_entity"] = "whatever the writer used to put here"
        assert rpe.semantic_digest(payload) == before

    def test_a_changed_vector_changes_the_digest(self) -> None:
        payload = _broken_payload()
        before = rpe.semantic_digest(payload)
        payload["embeddings"][1]["embedding"][0] += 1e-9
        assert rpe.semantic_digest(payload) != before

    def test_a_dropped_record_changes_the_digest(self) -> None:
        payload = _broken_payload()
        before = rpe.semantic_digest(payload)
        payload["embeddings"].pop()
        assert rpe.semantic_digest(payload) != before

    def test_a_reordered_record_changes_the_digest(self) -> None:
        payload = _broken_payload()
        before = rpe.semantic_digest(payload)
        payload["embeddings"].reverse()
        assert rpe.semantic_digest(payload) != before

    def test_changed_metadata_changes_the_digest(self) -> None:
        payload = _broken_payload()
        before = rpe.semantic_digest(payload)
        payload["metadata"]["num_train_epochs"] = 50
        assert rpe.semantic_digest(payload) != before


class TestRepairOneFile:
    def test_dry_run_reports_the_change_and_writes_nothing(self, tmp_path: Any) -> None:
        path = _write(str(tmp_path / "embeddings_scenes_x.json"), _broken_payload())
        before_bytes = open(path, "rb").read()
        manifest_path = str(tmp_path / "manifest.jsonl")

        outcome = rpe.repair_file(path, manifest_path=manifest_path, dry_run=True)

        assert outcome.status == "would-change"
        assert outcome.n_records == 3
        assert outcome.n_changed == 3
        assert open(path, "rb").read() == before_bytes
        assert not os.path.exists(manifest_path)

    def test_a_real_run_rewrites_only_the_repaired_field(self, tmp_path: Any) -> None:
        path = _write(str(tmp_path / "embeddings_scenes_x.json"), _broken_payload())
        manifest_path = str(tmp_path / "manifest.jsonl")

        outcome = rpe.repair_file(path, manifest_path=manifest_path, dry_run=False)

        assert outcome.status == "repaired"
        after = _read(path)
        assert [record["prompted_entity"] for record in after["embeddings"]] == [
            "an abbey scene", "a badlands scene", "an abbey scene",
        ]
        assert rpe.semantic_digest(after) == rpe.semantic_digest(_broken_payload())
        assert after["metadata"] == _broken_payload()["metadata"]
        assert [record["embedding"] for record in after["embeddings"]] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    def test_the_manifest_records_the_old_values_so_the_change_is_invertible(self, tmp_path: Any) -> None:
        path = _write(str(tmp_path / "embeddings_scenes_x.json"), _broken_payload())
        manifest_path = str(tmp_path / "manifest.jsonl")

        rpe.repair_file(path, manifest_path=manifest_path, dry_run=False)

        entries = rpe.read_manifest(manifest_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["old_prompted_entity"] == [
            "an an abbey scene scene", "an a badlands scene scene", "an an abbey scene scene",
        ]

        rpe.invert(manifest_path)
        assert _read(path) == _broken_payload()

    def test_a_file_that_is_already_correct_is_left_alone(self, tmp_path: Any) -> None:
        payload = _broken_payload()
        for record in payload["embeddings"]:
            record["prompted_entity"] = rpe.prompted_entity_from_prompt("scenes", record["prompt"])
        path = _write(str(tmp_path / "embeddings_scenes_ok.json"), payload)
        before_bytes = open(path, "rb").read()

        outcome = rpe.repair_file(path, manifest_path=str(tmp_path / "manifest.jsonl"), dry_run=False)

        assert outcome.status == "unchanged"
        assert outcome.n_changed == 0
        assert open(path, "rb").read() == before_bytes

    def test_a_second_pass_changes_nothing(self, tmp_path: Any) -> None:
        path = _write(str(tmp_path / "embeddings_scenes_x.json"), _broken_payload())
        manifest_path = str(tmp_path / "manifest.jsonl")

        rpe.repair_file(path, manifest_path=manifest_path, dry_run=False)
        after_first = open(path, "rb").read()
        second = rpe.repair_file(path, manifest_path=str(tmp_path / "manifest2.jsonl"), dry_run=False)

        assert second.status == "unchanged"
        assert second.n_changed == 0
        assert open(path, "rb").read() == after_first


class TestTheOriginalSurvivesAFailure:
    def test_validation_failure_leaves_the_original_byte_identical(self, tmp_path: Any, monkeypatch: Any) -> None:
        """The temporary file is validated after being reopened; if that validation raises, the
        original must never have been touched and no temporary file may be left behind."""
        path = _write(str(tmp_path / "embeddings_scenes_x.json"), _broken_payload())
        before_bytes = open(path, "rb").read()

        def _explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("validation of the rewritten file failed")

        monkeypatch.setattr(rpe, "_validate_rewritten_file", _explode)

        with pytest.raises(RuntimeError):
            rpe.repair_file(path, manifest_path=str(tmp_path / "manifest.jsonl"), dry_run=False)

        assert open(path, "rb").read() == before_bytes
        assert os.listdir(tmp_path) == ["embeddings_scenes_x.json"]

    def test_an_unrepairable_file_is_refused_without_being_touched(self, tmp_path: Any) -> None:
        payload = _broken_payload()
        payload["embeddings"][1]["prompt"] = "a photo of a badlands scene"
        path = _write(str(tmp_path / "embeddings_scenes_bad.json"), payload)
        before_bytes = open(path, "rb").read()

        outcome = rpe.repair_file(path, manifest_path=str(tmp_path / "manifest.jsonl"), dry_run=False)

        assert outcome.status == "refused"
        assert outcome.reason is not None
        assert open(path, "rb").read() == before_bytes


class TestResume:
    def test_a_file_already_in_the_manifest_is_not_reprocessed(self, tmp_path: Any, monkeypatch: Any) -> None:
        paths = [
            _write(str(tmp_path / "embeddings_scenes_a.json"), _broken_payload()),
            _write(str(tmp_path / "embeddings_scenes_b.json"), _broken_payload()),
        ]
        manifest_path = str(tmp_path / "manifest.jsonl")
        rpe.repair_file(paths[0], manifest_path=manifest_path, dry_run=False)

        touched: List[str] = []
        original_repair = rpe.repair_file

        def _spy(path: str, **kwargs: Any) -> Any:
            touched.append(path)
            return original_repair(path, **kwargs)

        monkeypatch.setattr(rpe, "repair_file", _spy)
        summary = rpe.repair_all(paths, manifest_path=manifest_path, dry_run=False)

        assert touched == [paths[1]]
        assert summary.n_skipped_already_done == 1
        assert summary.n_repaired == 1

    def test_the_summary_counts_every_file_it_was_given(self, tmp_path: Any) -> None:
        paths = [
            _write(str(tmp_path / "embeddings_scenes_a.json"), _broken_payload()),
            _write(str(tmp_path / "embeddings_scenes_b.json"), _broken_payload()),
        ]
        summary = rpe.repair_all(paths, manifest_path=str(tmp_path / "manifest.jsonl"), dry_run=False)

        assert summary.n_seen == 2
        assert summary.n_repaired == 2
        assert summary.n_refused == 0
        assert summary.n_seen == summary.n_repaired + summary.n_unchanged + summary.n_refused + summary.n_skipped_already_done
