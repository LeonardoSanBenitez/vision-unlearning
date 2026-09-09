"""Library code must resolve artifact-backed data through the Artifact classes.

Two gates, both of which the codebase previously failed while every other test was green:

1. **Architecture fitness** (this module's first half): library modules must not read
   artifact-backed data with raw filesystem calls, must not call the convenience functional
   wrappers, and must not reach through an artifact's private ``_get_data_path_local()`` to
   get a path and then open it by hand.

2. **Fresh clone** (second half): with an empty local cache and the data present only on
   HuggingFace, an RT's ``_compute_from_scratch`` must resolve rather than raise.

Why these are needed: a cascade that nothing calls is green on every other gate. The
artifacts were correct and the RTs bypassed them, so the RTs failed on a fresh clone while
resolving fine on a developer machine with a populated ``assets/``. Grepping for the class
name cannot catch this -- a caller can construct the artifact and still bypass the cascade by
taking a path out of it -- so the fitness test looks for the raw *operations*.
"""
from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, Iterator, List, Literal, Set

import pytest
from PIL import Image

import vision_unlearning.benchmarks.I_care.result_templates as _rt_mod
import vision_unlearning.artifact as _artifact_mod
import vision_unlearning.datasets.testbed as _testbed_mod
import vision_unlearning.integrations.huggingface as _hf_mod

_I_CARE_DIR = os.path.dirname(_rt_mod.__file__)
_VU_DIR = os.path.dirname(_artifact_mod.__file__)

# Library code: everything that is not a pipeline script, a report script, or user code.
# CONTRIBUTING_ICARE.md section 8 defines the split; pipeline_NN_* and reports/ are user
# code and may legitimately use the functional wrappers and raw paths.
_LIBRARY_MODULES = [
    os.path.join(_I_CARE_DIR, "result_templates.py"),
    os.path.join(_I_CARE_DIR, "similarity.py"),
    os.path.join(_I_CARE_DIR, "metadata.py"),
    os.path.join(_VU_DIR, "datasets", "testbed.py"),
]

# Functional wrappers exist for user code only; library code constructs the artifact.
_FORBIDDEN_WRAPPERS = {
    "get_metadata_filtered",
    "get_interference_per_pair",
    "get_interference_per_entity",
    "get_off_image_path",
}

# Path helpers that hand out a raw location for an artifact-backed file.
_FORBIDDEN_PATH_HELPERS = {
    "get_interference_per_entity_path",
    "get_interference_per_pair_path",
    "get_generated_dataset_folder",
    "get_shared_baseline_folder",
    "get_embedding_output_path",
}

# Explicit, justified exceptions. Listed here rather than silently excluded so that each one
# is a visible decision with a reason attached: if an entry stops being justified, it should
# be deleted and the code fixed. Keyed by module basename -> allowed callee names.
_ALLOWLIST: Dict[str, Set[str]] = {
    # get_interference_per_pair_inverse is deliberately local-only and NOT part of the
    # cascade. Its only callers are pipeline_07 and unlearning-analysis scripts, which run on
    # a machine that has just produced these files locally, and it is not reachable from
    # Forgety; pipeline_07's own guard is a raw local check too, so that loop is internally
    # consistent. Migrating it is a real behaviour change to a live m_e metric and is
    # deliberately out of scope. See metadata.py's note above the function.
    # get_interference_per_entity_path is additionally used by save_interference_per_entity,
    # which must have a path in order to write; a writer is not a reader bypassing a cascade.
    "metadata.py": {
        "get_metadata_filtered", "get_interference_per_pair_path",
        "get_interference_per_entity_path",
    },
    # testbed.py defines the path helpers below; GeneratedDataset.get_off_image_path is the
    # classmethod delegating to the module-level implementation, and the GeneratedDataset
    # artifact builds its own storage locations from get_generated_dataset_folder /
    # get_shared_baseline_folder. A module implementing the abstraction is not a module
    # bypassing it.
    "testbed.py": {
        "get_off_image_path", "get_generated_dataset_folder",
        "get_shared_baseline_folder",
    },
}


class _CallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.wrapper_calls: List[str] = []
        self.path_helper_calls: List[str] = []
        self.private_path_calls: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _FORBIDDEN_WRAPPERS:
                self.wrapper_calls.append(f"{func.id} (line {node.lineno})")
            elif func.id in _FORBIDDEN_PATH_HELPERS:
                self.path_helper_calls.append(f"{func.id} (line {node.lineno})")
        elif isinstance(func, ast.Attribute):
            if func.attr == "_get_data_path_local" and not isinstance(func.value, ast.Name):
                # obj._get_data_path_local() where obj is not a bare name -- e.g.
                # BaselineEmbeddings(...)._get_data_path_local(): an artifact constructed
                # purely to act as a filename calculator.
                self.private_path_calls.append(f"_get_data_path_local (line {node.lineno})")
            elif func.attr == "_get_data_path_local" and isinstance(func.value, ast.Name) \
                    and func.value.id != "self":
                self.private_path_calls.append(f"_get_data_path_local (line {node.lineno})")
        self.generic_visit(node)


def _visit(path: str) -> _CallVisitor:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    v = _CallVisitor()
    v.visit(tree)
    return v


@pytest.mark.parametrize("module_path", _LIBRARY_MODULES, ids=os.path.basename)
class TestArchitectureFitness:
    def test_no_functional_wrapper_calls(self, module_path: str) -> None:
        """Library code constructs the Artifact; only user code may use the wrappers."""
        allowed = _ALLOWLIST.get(os.path.basename(module_path), set())
        found = [c for c in _visit(module_path).wrapper_calls
                 if c.split(" ")[0] not in allowed]
        assert not found, (
            f"{os.path.basename(module_path)} calls functional wrapper(s) {found}. "
            "Library code must construct the Artifact class directly (CONTRIBUTING_ICARE "
            "section 8) so the local -> HuggingFace cascade is not bypassed."
        )

    def test_no_raw_path_helper_calls(self, module_path: str) -> None:
        """A raw path helper returns a location, which discards the cascade."""
        allowed = _ALLOWLIST.get(os.path.basename(module_path), set())
        found = [c for c in _visit(module_path).path_helper_calls
                 if c.split(" ")[0] not in allowed]
        assert not found, (
            f"{os.path.basename(module_path)} calls raw path helper(s) {found}. "
            "Resolve through the artifact's .compute()/.exists() instead."
        )

    def test_no_artifact_used_as_filename_calculator(self, module_path: str) -> None:
        """Constructing an artifact and taking its private path out is still a bypass.

        This is the shape that made the migration look complete while having no cascade:
        the class name greps green, but ``._get_data_path_local()`` throws the cascade away.
        """
        found = _visit(module_path).private_path_calls
        assert not found, (
            f"{os.path.basename(module_path)} takes a path out of an artifact via "
            f"{found}. Resolve via .compute()/.exists(); never take a path out of an "
            "artifact."
        )


# ---------------------------------------------------------------------------
# Fresh clone: empty local cache, data only on HuggingFace
# ---------------------------------------------------------------------------

_LABELS = [f"Person {chr(65 + i)}" for i in range(12)]


def _metadata() -> List[Dict[str, Any]]:
    # Two occupation groups: the categorical RT compares groups and needs at least two.
    return [
        {"name": n, "occupation_simplified": "politician" if i < 6 else "artist"}
        for i, n in enumerate(_LABELS)
    ]


def _interference_per_entity() -> List[Dict[str, Any]]:
    return [
        {
            "name": n,
            "occupation_simplified": "politician" if i < 6 else "artist",
            "hpi_bin": float(i),
            "metric_distil_400_emitter_average_rmse (↓)": float(i) * 1.5,
            "metric_distil_400_emitter_average_ssim (↑)": float(i) * 0.5,
        }
        for i, n in enumerate(_LABELS)
    ]


@pytest.fixture
def remote_has_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> Iterator[Set[str]]:
    """Simulate a fresh clone: nothing local, everything on HuggingFace.

    ``base_folder`` points at an empty tmp dir. The HF layer reports every file as present
    and "downloads" it by writing the fixture to the artifact's expected local path -- the
    same two steps the real cascade performs, so a caller that bypasses the cascade cannot
    accidentally pass by reading this machine's real ``./assets``.
    """
    downloaded: Set[str] = set()
    payloads = {
        "metadata_people_2_enriched_filtered.json": _metadata(),
        "interference_per_entity_people.json": _interference_per_entity(),
    }

    def fake_exists(repo: str, path: str, token: Any = None) -> bool:
        return os.path.basename(path) in payloads

    def fake_download(
        folder_datasets: str, dataset_repository: str, file_path: str, token: Any = None
    ) -> None:
        name = os.path.basename(file_path)
        if name not in payloads:
            raise FileNotFoundError(file_path)
        local = os.path.join(folder_datasets, file_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            json.dump(payloads[name], f)
        downloaded.add(name)

    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_exists", fake_exists)
    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_download", fake_download)
    monkeypatch.setattr(_artifact_mod, "get_hf_token_from_env", lambda: None)
    yield downloaded


class TestFreshCloneResolvesFromHuggingFace:
    """Each RT must resolve its inputs from HuggingFace when the local cache is empty.

    Before the artifact migration every one of these raised FileNotFoundError against an
    empty cache while the data sat on HuggingFace -- the exact "reports not-yet-precomputed
    for data that was already precomputed" outcome.
    """

    def test_significant_relationship_numerical(
        self, remote_has_everything: Set[str], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateSignificantRelationshipNumerical(
            task="people", unlearning_algorithm="distil",
            interference_entity="Emitter average rmse", attribute="hpi_bin",
            base_folder=str(tmp_path), save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert "pearson_pvalue" in data["result"]
        assert "interference_per_entity_people.json" in remote_has_everything

    def test_significant_relationship_categorical(
        self, remote_has_everything: Set[str], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateSignificantRelationshipCategorical(
            task="people", unlearning_algorithm="distil",
            interference_entity="Emitter average rmse",
            attribute="occupation_simplified",
            base_folder=str(tmp_path), save_outputs=False,
        )
        rt._compute_from_scratch()
        assert "interference_per_entity_people.json" in remote_has_everything

    def test_metric_metric_alignment(
        self, remote_has_everything: Set[str], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateMetricMetricAlignment(
            task="people", unlearning_algorithm="distil",
            interference_entity_1="Emitter average rmse",
            interference_entity_2="Emitter average ssim",
            base_folder=str(tmp_path), save_outputs=False,
        )
        rt._compute_from_scratch()
        assert "interference_per_entity_people.json" in remote_has_everything

    def test_reads_the_temp_base_folder_not_the_real_assets(
        self, remote_has_everything: Set[str], tmp_path: Any
    ) -> None:
        """base_folder must be honoured, not silently replaced by ./assets.

        Two independent bugs sat on the same line before the migration: no cascade, and the
        RT's base_folder dropped -- so the error named ``assets/...`` even when the RT was
        constructed with a temp folder.
        """
        rt = _rt_mod.ResultTemplateSignificantRelationshipNumerical(
            task="people", unlearning_algorithm="distil",
            interference_entity="Emitter average rmse", attribute="hpi_bin",
            base_folder=str(tmp_path), save_outputs=False,
        )
        rt._compute_from_scratch()
        assert os.path.exists(tmp_path / "interference_per_entity_people.json"), (
            "the artifact was downloaded somewhere other than the RT's base_folder"
        )


# ---------------------------------------------------------------------------
# Fresh clone: the embedding RT (ResultTemplateEmbeddingUnlearningProfile)
# ---------------------------------------------------------------------------

_EMBEDDING_ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve"]
_EMBEDDING_FORGOTTEN_ENTITY = "Alice"
_EMBEDDING_DIM = 16


def _embedding_records(lora_state: str, forgotten_entity: str) -> List[Dict[str, Any]]:
    """One record per (entity, seed), matching the shape read by
    ``ResultTemplateEmbeddingUnlearningProfile._mean_embeddings`` (keyed by ``prompt``,
    not ``prompted_entity`` -- see CONTRIBUTING_ICARE section 6).
    """
    import numpy as np
    rng = np.random.default_rng(0 if lora_state == "off" else 1)
    records = []
    for entity in _EMBEDDING_ENTITIES:
        for seed in range(4):
            vec = rng.standard_normal(_EMBEDDING_DIM).tolist()
            if lora_state == "on" and entity == forgotten_entity:
                vec = [-v for v in vec]
            records.append({
                "prompted_entity": entity,
                "seed": seed,
                "prompt": f"An image of {entity}",
                "embedding": vec,
            })
    return records


@pytest.fixture
def remote_has_embeddings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Set[str]]:
    """Simulate a fresh clone for the embedding RT: baseline + entity embedding files
    exist only on HuggingFace, nothing local.

    Mirrors ``remote_has_everything`` above but serves the two
    ``BaselineEmbeddings``/``EntityEmbeddings`` files (under the ``datasets/`` HF prefix)
    instead of ``MetadataFiltered``/``InterferencePerEntity``.
    """
    downloaded: Set[str] = set()
    payloads = {
        "datasets/embeddings_people_original.json": {
            "metadata": {"task": "people", "embedding_model": "dinov2_vits14"},
            "embeddings": _embedding_records("off", _EMBEDDING_FORGOTTEN_ENTITY),
        },
        "datasets/embeddings_people_Alice_uce_000.json": {
            "metadata": {"task": "people", "embedding_model": "dinov2_vits14"},
            "embeddings": _embedding_records("on", _EMBEDDING_FORGOTTEN_ENTITY),
        },
    }

    def fake_exists(repo: str, path: str, token: Any = None) -> bool:
        return path in payloads

    def fake_download(
        folder_datasets: str, dataset_repository: str, file_path: str, token: Any = None
    ) -> None:
        if file_path not in payloads:
            raise FileNotFoundError(file_path)
        local = os.path.join(folder_datasets, file_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            json.dump(payloads[file_path], f)
        downloaded.add(file_path)

    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_exists", fake_exists)
    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_download", fake_download)
    monkeypatch.setattr(_artifact_mod, "get_hf_token_from_env", lambda: None)
    yield downloaded


class TestFreshCloneResolvesEmbeddingsFromHuggingFace:
    """``ResultTemplateEmbeddingUnlearningProfile`` must resolve its embedding inputs from
    HuggingFace when the local cache is empty -- the same "fresh clone" property already
    proven for the metadata-based RTs above, extended to the embedding artifacts
    (``BaselineEmbeddings``/``EntityEmbeddings``) added in ``BenchmarkRefactor`` W2d.

    ``InterferencePerEntity`` and ``MetadataFiltered`` are deliberately absent from the
    fixture: the RT degrades gracefully (``ratio_source="not_available"``, Me-aggregate
    colouring skipped) rather than requiring them, so this test also exercises that
    documented degradation path on a genuinely empty cache.
    """

    def test_embedding_unlearning_profile(
        self, remote_has_embeddings: Set[str], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateEmbeddingUnlearningProfile(
            task="people",
            unlearning_algorithm="uce",
            entity=_EMBEDDING_FORGOTTEN_ENTITY,
            base_folder=str(tmp_path),
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert data["result"]["ratio_source"] == "not_available"
        assert "datasets/embeddings_people_original.json" in remote_has_embeddings
        assert "datasets/embeddings_people_Alice_uce_000.json" in remote_has_embeddings

    def test_reads_the_temp_base_folder_not_the_real_assets(
        self, remote_has_embeddings: Set[str], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateEmbeddingUnlearningProfile(
            task="people",
            unlearning_algorithm="uce",
            entity=_EMBEDDING_FORGOTTEN_ENTITY,
            base_folder=str(tmp_path),
            save_outputs=False,
        )
        rt._compute_from_scratch()
        assert os.path.exists(
            tmp_path / "datasets" / "embeddings_people_original.json"
        ), "the baseline embedding artifact was downloaded somewhere other than base_folder"


# ---------------------------------------------------------------------------
# Fresh clone: an image-loading RT (ResultTemplateInterferenceVisualSummary)
# ---------------------------------------------------------------------------

_IMAGE_LABELS = [f"Person {chr(65 + i)}" for i in range(12)]
_IMAGE_SEEDS = [42, 43, 44, 45]


def _image_metadata() -> List[Dict[str, Any]]:
    return [
        {"name": n, "occupation_simplified": "politician" if i < 6 else "artist"}
        for i, n in enumerate(_IMAGE_LABELS)
    ]


def _write_tiny_png(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (2, 2), color=(128, 128, 128)).save(path)


@pytest.fixture
def remote_has_image_dataset(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Simulate a fresh clone for an RT that loads generated images via ``GeneratedDataset``
    (a folder-shaped artifact, unlike every other fixture in this module).

    Serves ``MetadataFiltered``, ``InterferencePerPair`` (with its ``_validate`` loosened
    to this fixture's 12-entity set -- production defaults ``max_identities=100``), and
    both the shared baseline and per-entity image-dataset folders purely from HuggingFace:
    nothing exists under the RT's ``base_folder`` before ``_compute_from_scratch`` runs.
    """
    task: Literal["scenes", "objects", "breeds", "people"] = "people"
    method: Literal["munba", "uce", "distil"] = "uce"
    epochs = 0  # unlearning_algorithm_to_epochs['people']['uce']
    entity = _IMAGE_LABELS[0]

    metadata = _image_metadata()
    prompts = [f"An image of {m['name']}" for m in metadata]

    # Production InterferencePerPair._validate requires exactly max_identities (100)
    # entries; this fixture reuses the same 12-entity set as the rest of this module.
    monkeypatch.setattr(_rt_mod.InterferencePerPair, "_validate", lambda self, data: None)

    interference_per_pair = {
        m["name"]: {"clip_diff": float(i)} for i, m in enumerate(metadata)
    }
    json_payloads = {
        "metadata_people_2_enriched_filtered.json": metadata,
        "datasets/interferences_caused_by_people_0_uce_0.json": interference_per_pair,
    }

    baseline_folder_name = os.path.basename(
        _testbed_mod.get_shared_baseline_folder(task, "assets", "sd1.4")
    )
    entity_folder_name = os.path.basename(
        _testbed_mod.get_generated_dataset_folder(task, method, epochs, entity, "assets", "sd1.4")
    )
    image_folders = {baseline_folder_name, entity_folder_name}

    def fake_json_exists(repo: str, path: str, token: Any = None) -> bool:
        return path in json_payloads

    def fake_json_download(
        folder_datasets: str, dataset_repository: str, file_path: str, token: Any = None
    ) -> None:
        if file_path not in json_payloads:
            raise FileNotFoundError(file_path)
        local = os.path.join(folder_datasets, file_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            json.dump(json_payloads[file_path], f)

    def fake_dataset_exists(
        dataset_repository: str, dataset_config: str, token: Any = None,
        path_in_repo: Any = None,
    ) -> bool:
        return dataset_config in image_folders

    def fake_dataset_download(
        folder_datasets: str, dataset_repository: str, dataset_config: str,
        token: Any = None, clean: bool = False,
        folder_cache: str = "/tmp/huggingface_cache", clean_cache: bool = False,
        path_in_repo: Any = None, overwrite: bool = False,
    ) -> None:
        folder = os.path.join(folder_datasets, dataset_config)
        state: Literal["on", "off"] = "off" if dataset_config == baseline_folder_name else "on"
        for seed in _IMAGE_SEEDS:
            for prompt in prompts:
                _write_tiny_png(
                    os.path.join(
                        folder, _testbed_mod.get_generated_dataset_file(state, seed, prompt)
                    )
                )
        with open(os.path.join(folder, "metadata.jsonl"), "w", encoding="utf-8") as f:
            f.write("{}\n")

    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_exists", fake_json_exists)
    monkeypatch.setattr(_artifact_mod, "huggingface_dataset_file_download", fake_json_download)
    monkeypatch.setattr(_artifact_mod, "get_hf_token_from_env", lambda: None)
    monkeypatch.setattr(_hf_mod, "huggingface_dataset_exists", fake_dataset_exists)
    monkeypatch.setattr(_hf_mod, "huggingface_dataset_download", fake_dataset_download)
    return {"task": task, "method": method, "entity": entity}


class TestFreshCloneResolvesImagesFromHuggingFace:
    """An image-loading RT must resolve its ``GeneratedDataset`` folders (baseline and
    per-entity) from HuggingFace when the local cache is empty -- the folder-shaped
    counterpart to ``TestFreshCloneResolvesFromHuggingFace`` above. Exercises the exact RT
    Forgety's entity page calls (``pipeline_08``'s ``InterferenceVisualSummary`` runner).
    """

    def test_interference_visual_summary(
        self, remote_has_image_dataset: Dict[str, Any], tmp_path: Any
    ) -> None:
        rt = _rt_mod.ResultTemplateInterferenceVisualSummary(
            task=remote_has_image_dataset["task"],
            unlearning_algorithm=remote_has_image_dataset["method"],
            interference_pair="clip_diff",
            entity=remote_has_image_dataset["entity"],
            base_folder=str(tmp_path),
            save_outputs=False,
        )
        data = rt._compute_from_scratch()
        assert len(data["result"]["displayed_entities"]) == 9
        assert set(data["result"]["images"]["off"].keys()) == set(
            data["result"]["displayed_entities"]
        )
        assert set(data["result"]["images"]["on"].keys()) == set(
            data["result"]["displayed_entities"]
        )


# ---------------------------------------------------------------------------
# Allowlist entries must still be justified
# ---------------------------------------------------------------------------

class TestAllowlistEntriesStillHold:
    """``_ALLOWLIST`` entries are a visible, justified exception to the fitness gate above.

    An allow-entry naming a module or callee that no longer exists is dead weight that
    silently widens the gate's blind spot -- the low-footprint "reason still holds" check
    from the plan (W-D.3).
    """

    def test_allowlisted_modules_are_still_gated_modules(self) -> None:
        gated_basenames = {os.path.basename(p) for p in _LIBRARY_MODULES}
        for module_basename in _ALLOWLIST:
            assert module_basename in gated_basenames, (
                f"_ALLOWLIST has an entry for {module_basename!r}, which is not (or no "
                f"longer) one of _LIBRARY_MODULES. Delete the stale entry."
            )

    def test_allowlisted_callees_are_still_actually_called(self) -> None:
        for module_basename, allowed_callees in _ALLOWLIST.items():
            module_path = next(
                p for p in _LIBRARY_MODULES if os.path.basename(p) == module_basename
            )
            visitor = _visit(module_path)
            actually_called = {
                c.split(" ")[0]
                for c in (
                    visitor.wrapper_calls + visitor.path_helper_calls
                )
            }
            for callee in allowed_callees:
                assert callee in actually_called, (
                    f"_ALLOWLIST[{module_basename!r}] permits {callee!r}, but {module_basename} "
                    f"no longer calls it. Delete the stale allow-entry -- an unused exception "
                    f"is dead weight that silently widens the gate's blind spot."
                )
