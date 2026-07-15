from __future__ import annotations

from typing import Literal, Tuple, List, Dict, Optional, Any, cast
import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, linregress, pearsonr, spearmanr
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel

from vision_unlearning.utils.logger import get_logger
from vision_unlearning.artifact import SingleFileArtifact
from vision_unlearning.datasets.testbed import get_metadata_filtered, get_generated_dataset_folder, get_generated_dataset_file, get_target_overwrite
from vision_unlearning.integrations.huggingface import (
    get_hf_token_from_env,
    huggingface_dataset_file_exists,
    huggingface_dataset_file_download,
    huggingface_dataset_file_upload,
)
from vision_unlearning.benchmarks.I_care.configuration import (
    type_task,
    type_unlearning_algorithm,
    type_me,
    type_model,
    type_l,
    unlearning_algorithm_to_epochs,
)


logger = get_logger('I_care')


##########################################
# Metadata files - interference_per_pair
##########################################
def _interference_per_pair_filename(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
) -> str:
    return f'interferences_caused_by_{task}_{index}_{method}_{num_train_epochs}.json'


def get_interference_per_pair_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, 'datasets', _interference_per_pair_filename(task, index, method, num_train_epochs))


def get_interference_per_pair(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    max_identities: int = 100,
    base_folder: str = 'assets',
) -> Dict[str, Dict[str, float]]:
    # TODO: maybe this function should first check locally if the file exists, and if not, check in huggingface if the file exists there, and just then return an error if neighter?
    assert os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder)), "Caused interferences by this entity were not computed yet"
    with open(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder), 'r') as f:
        interference_per_pair = json.load(f)
    assert isinstance(interference_per_pair, dict)
    assert len(interference_per_pair) == max_identities
    return interference_per_pair


def exists_interference_per_pair(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> bool:
    return os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder))

def save_interference_per_pair(
    interference_per_pair: Dict[str, Dict[str, float]],
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> None:
    assert isinstance(interference_per_pair, dict)
    assert len(interference_per_pair) > 0, "interference_per_pair should not be empty"
    with open(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder), 'w') as f:
            json.dump(interference_per_pair, f)


def get_interference_per_pair_inverse(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    index_start: int = 0,
    max_identities: int = 100,
    base_folder: str = 'assets',
) -> Dict[str, Dict[str, float]]:
    metadata_filtered = get_metadata_filtered(task)
    target = metadata_filtered[index]['name']

    interference_per_pair_inverse = {}
    for idx_emitter in range(index_start, index_start + max_identities):
        path = get_interference_per_pair_path(task, idx_emitter, method, num_train_epochs, base_folder)
        if os.path.exists(path):  # Unlearning already performed
            with open(path, 'r') as f:
                interference_per_pair_temp = json.load(f)
            interference_per_pair_inverse[metadata_filtered[idx_emitter]['name']] = interference_per_pair_temp[target]

    assert isinstance(interference_per_pair_inverse, dict)
    assert len(interference_per_pair_inverse) <= max_identities
    return interference_per_pair_inverse


class InterferencePerPair(SingleFileArtifact):
    """Object-oriented interface over a single per-pair interference file.

    Wraps interferences_caused_by_{task}_{index}_{method}_{epochs}.json and adds the shared
    local -> HuggingFace -> (not-computed-on-demand) storage cascade. Complements the
    get_interference_per_pair / exists_interference_per_pair / save_interference_per_pair
    helpers, which remain the fast local-only path.
    """
    task: type_task = 'people'
    index: int
    method: type_unlearning_algorithm
    num_train_epochs: int
    max_identities: int = 100

    def _get_data_path_remote(self) -> str:
        return f"datasets/{_interference_per_pair_filename(self.task, self.index, self.method, self.num_train_epochs)}"

    def _compute_from_scratch(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError(
            "InterferencePerPair is produced by the interference pipeline, not computed on "
            "demand. Provide the local file or fetch it from HuggingFace."
        )

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict)
        assert len(data) == self.max_identities

    def compute(self) -> Dict[str, Dict[str, float]]:
        return cast(Dict[str, Dict[str, float]], self._resolve())


##########################################
# Metadata files - interference_per_entity
##########################################
def get_interference_per_entity_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, f"interference_per_entity_{task}.json")


def get_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    max_identities: int = 100,
    base_folder: str = 'assets',
) -> List[Dict[str, Any]]:
    assert os.path.exists(get_interference_per_entity_path(task, base_folder=base_folder))
    with open(get_interference_per_entity_path(task, base_folder=base_folder), "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    assert len(metadata_filtered) == max_identities
    return metadata_filtered


def save_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    metadata_filtered: List[Dict[str, Any]],
    base_folder: str = 'assets',
) -> None:
    with open(get_interference_per_entity_path(task, base_folder=base_folder), "w", encoding="utf-8") as f:
        json.dump(metadata_filtered, f, indent=4)



# InterferencePerEntity is stored as a single JSON file and shares the local -> HuggingFace
# -> from-scratch storage cascade with the Result Templates; both inherit that cascade from
# SingleFileArtifact. The functional helpers below (get_interference_per_entity, ...) remain
# available and coexist with this object-oriented interface.
class InterferencePerEntity(SingleFileArtifact):
    task: type_task = 'people'
    # This class deprecates: save_interference_per_entity, get_interference_per_entity_path

    def _get_data_path_remote(self) -> str:
        return f'interference_per_entity_{self.task}.json'

    def _compute_from_scratch(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "InterferencePerEntity._compute_from_scratch is not yet implemented. "
            "Provide a pre-computed file or fetch from HuggingFace."
        )

    def _validate(self, data: Any) -> None:
        assert type(data) == list, f"Expected a dict in the json file, but got {type(data)}"
        assert len(data) > 0  # == 100
        assert all(isinstance(item, dict) for item in data)

    def compute(self) -> List[Dict[str, Any]]:
        """Resolve the per-entity interference summary from local disk, HuggingFace, or from
        scratch. The storage cascade lives in SingleFileArtifact; this method only pins the
        return type."""
        return cast(List[Dict[str, Any]], self._resolve())




def choose_metric_column_interference_per_entity(
    unlearning_algorithm: type_unlearning_algorithm,
    interference_entity: type_me,
    metric_cols: List[str],
) -> str:
    """
    The columns of the interference per entity file are not named in a way that is easy to generate given `unlearning_algorithm` and `interference_entity`, so we need to search for the right one.
    We assume there is only one match, and we assert it. If there are no matches or more than one match, we raise an error.

    The names look like this:
        'metric_distil_400_emitter_minus_receiver_worst_interfered_ssim (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_brisque_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_clip_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_rmse (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_ssim (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_zero_clip_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_average_brisque_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_average_clip_diff (↑)',
       'metric_uce_000_emitter_minus_receiver_average_rmse (↓)',
       'metric_munba_100_emitter_minus_receiver_average_ssim (↑)',
    
    TODO: these names are defined in `4. Compute interference per entity.ipynb`. There should be a central way of defining them.
    """
    pattern = f"metric_{unlearning_algorithm}_[^_]*_{interference_entity.lower().replace(' ', '_')} .*"
    matching_cols = [col for col in metric_cols if re.match(pattern, col)]
    if len(matching_cols) == 0:
        raise ValueError(f'No metric column found for unlearning_algorithm={unlearning_algorithm} and interference_entity={interference_entity}')
    elif len(matching_cols) > 1:
        raise ValueError(f'Multiple metric columns found for unlearning_algorithm={unlearning_algorithm} and interference_entity={interference_entity}: {matching_cols}')
    return matching_cols[0]


##########################################
# Metadata files - embeddings
##########################################
def get_embedding_output_path(
    task: str,
    hf_entity: str,
    method: str,
    num_train_epochs: int,
    base_folder: str = "assets",
) -> str:
    """Local path for a per-entity (unlearned) embedding file.

    The method-agnostic baseline is addressed by :class:`BaselineEmbeddings`; it has no
    method or epoch, so it must never be built through this method/epoch-carrying interface.
    """
    filename = f"embeddings_{task}_{hf_entity}_{method}_{num_train_epochs:03d}.json"
    return os.path.join(base_folder, "datasets", filename)


def get_embedding_hf_path(
    task: str,
    hf_entity: str,
    method: str,
    num_train_epochs: int,
) -> str:
    """HuggingFace repo path (no leading slash) for a per-entity (unlearned) embedding file.

    The method-agnostic baseline is addressed by :class:`BaselineEmbeddings`.
    """
    return f"datasets/embeddings_{task}_{hf_entity}_{method}_{num_train_epochs:03d}.json"


def _embedding_function_suffix(embedding_function: type_l) -> str:
    """Filename segment for the embedding function.

    ``dino_embedding`` (the only function currently produced) keeps the historical
    method-agnostic name unchanged; any other function adds a disambiguating segment so
    files never collide. The non-``dino`` branch is interface-only: it makes a second
    embedding function representable without renaming any existing asset.
    """
    return "" if embedding_function == "dino_embedding" else f"_{embedding_function}"


# BaselineEmbeddings and EntityEmbeddings wrap the DINOv2 (or other embedding-function)
# embedding files, sharing the local -> HuggingFace -> from-scratch storage cascade with the
# Result Templates and the interference artifacts (all inherit it from SingleFileArtifact).
class BaselineEmbeddings(SingleFileArtifact):
    """Embeddings of the ORIGINAL-model generated images for one task.

    The baseline is produced by the base model with no unlearning, so it depends only on the
    task, the base model, and the embedding function — never on an unlearning method or its
    epoch count.
    """
    task: type_task = "people"
    model: type_model = "sd1.4"
    embedding_function: type_l = "dino_embedding"

    def _get_data_path_remote(self) -> str:
        suffix = _embedding_function_suffix(self.embedding_function)
        return f"datasets/embeddings_{self.task}_original{suffix}.json"

    def _compute_from_scratch(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "BaselineEmbeddings._compute_from_scratch requires a GPU (embedding the "
            "generated baseline images with DINOv2) and is not supported here. Provide the "
            "local file, fetch it from HuggingFace, or run pipeline_05 (compute embeddings) "
            "for the baseline."
        )

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict)
        assert "embeddings" in data

    def compute(self) -> Dict[str, Any]:
        """Resolve the baseline embeddings from local disk, HuggingFace, or from scratch."""
        return cast(Dict[str, Any], self._resolve())


class EntityEmbeddings(SingleFileArtifact):
    """Embeddings of the images generated by a model UNLEARNED on one entity.

    Unlike the baseline, these depend on the unlearning method and its epoch count. The
    epoch count is derived from the executed-combination table so callers only pass the
    method, mirroring the existing path helpers.
    """
    task: type_task = "people"
    hf_entity: str = ""
    unlearning_algorithm: type_unlearning_algorithm = "distil"
    model: type_model = "sd1.4"
    embedding_function: type_l = "dino_embedding"

    def _epochs(self) -> int:
        return unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

    def _get_data_path_remote(self) -> str:
        suffix = _embedding_function_suffix(self.embedding_function)
        return (
            f"datasets/embeddings_{self.task}_{self.hf_entity}"
            f"_{self.unlearning_algorithm}_{self._epochs():03d}{suffix}.json"
        )

    def _compute_from_scratch(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "EntityEmbeddings._compute_from_scratch requires a GPU (embedding the "
            "per-entity unlearned images with DINOv2) and is not supported here. Provide the "
            "local file, fetch it from HuggingFace, or run pipeline_05 (compute embeddings)."
        )

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict)
        assert "embeddings" in data

    def compute(self) -> Dict[str, Any]:
        """Resolve the per-entity embeddings from local disk, HuggingFace, or from scratch."""
        return cast(Dict[str, Any], self._resolve())



