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
)


logger = get_logger('I_care')


##########################################
# Metadata files - interference_per_pair
##########################################
def get_interference_per_pair_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, 'datasets', f'interferences_caused_by_{task}_{index}_{method}_{num_train_epochs}.json')


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
    target_preprocessed: str,
    method: str,
    num_train_epochs: int,
    base_folder: str = "assets",
) -> str:
    """Local path for the output JSON.

    Baseline embeddings (target_preprocessed == 'original') use a method-agnostic
    name because they embed generated_{task}_baseline/ which has no method or epoch.
    Per-entity embeddings include method and epoch.
    """
    if target_preprocessed == "original":
        filename = f"embeddings_{task}_original.json"
    else:
        filename = f"embeddings_{task}_{target_preprocessed}_{method}_{num_train_epochs:03d}.json"
    return os.path.join(base_folder, "datasets", filename)


def get_embedding_hf_path(
    task: str,
    target_preprocessed: str,
    method: str,
    num_train_epochs: int,
) -> str:
    """HF repo path (no leading slash) for the output JSON."""
    if target_preprocessed == "original":
        return f"datasets/embeddings_{task}_original.json"
    return f"datasets/embeddings_{task}_{target_preprocessed}_{method}_{num_train_epochs:03d}.json"



