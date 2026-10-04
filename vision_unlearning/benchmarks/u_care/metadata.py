import re
from typing import Any, Dict, List, cast
from pathlib import Path
from vision_unlearning.artifact import ArtifactNotAvailableError, SingleFileArtifact
from vision_unlearning.benchmarks.care import MetricEffectPerEntity, MetricEffectPerEntityPair
from vision_unlearning.benchmarks.u_care import configuration as cfg
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _me_column_fragment(inteference_entity: cfg.type_me) -> str:
    """The column fragment for a metric column, given the interference entity. E.g. 'accuracy_diff'
    or 'target_probability'."""
    return inteference_entity.lower().replace(' ', '_')

def choose_metric_column(method: cfg.type_unlearning_algorithm, interference_entity: cfg.type_me, metric_cols: List[str]) -> str:
    fragment = _me_column_fragment(interference_entity)
    pattern = f"metric_{method}_[^_]*_{fragment} .*"
    matching_cols = [col for col in metric_cols if re.match(pattern, col)]
    if len(matching_cols) == 0:
        raise ValueError(f'No metric column found for unlearning_algorithm={method} and interference_entity={interference_entity}')
    elif len(matching_cols) > 1:
        raise ValueError(f'Multiple metric columns found for unlearning_algorithm={method} and interference_entity={interference_entity}: {matching_cols}')
    return matching_cols[0]



class EntityMetadata(SingleFileArtifact):
    """The 71 entities. List[Dict] with keys name, index, domain ('style'|'object'),
    unlearnable (bool). Model-independent, so no model field."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME

    def _get_data_path_remote(self) -> str:
        return "metadata_filtered.json"

    def _compute_from_scratch(self) -> list:
        raise ArtifactNotAvailableError(
            "EntityMetadata is built by pipeline_01. Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, list) and len(data) == 71

    def compute(self) -> list:
        return cast(list, self._resolve())

class InterferencePerPair(MetricEffectPerEntityPair):
    """One emitter's row: {receiver: {accuracy, accuracy_diff, target_probability,
    target_probability_diff}}, 71 keys."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    emitter: str
    method: cfg.type_unlearning_algorithm
    base_folder: str  # Add base_folder as an attribute

    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like Path

    @property
    def local_path(self) -> Path:
        """Get the local path for the artifact."""
        return Path(self.base_folder) / f"interferences_caused_by_{self.emitter}_{self.method}_{self.model}.json"

    def exists(self) -> bool:
        """Check if the artifact exists locally or remotely."""
        # Check if the file exists locally
        if self.local_path.exists():
            logger.debug(f"Found local artifact for emitter {self.emitter}: {self.local_path}")
            return True
        # If not found locally, fallback to checking on Hugging Face
        logger.debug(f"Local artifact not found for emitter {self.emitter}, checking remote.")
        return self._exists_on_huggingface()

    def _exists_on_huggingface(self) -> bool:
        """Check if the artifact exists on Hugging Face."""
        try:
            # Assuming `_get_data_path_remote` provides the remote path
            remote_path = self._get_data_path_remote()
            logger.debug(f"Checking existence of remote artifact: {remote_path}")
            # Logic to check if the file exists on Hugging Face (e.g., using HTTP HEAD request)
            return False  # Replace with actual check
        except Exception as e:
            logger.warning(f"Failed to check remote existence for {self.emitter}: {e}")
            return False

    def _get_data_path_remote(self) -> str:
        """Get the remote path for the artifact."""
        index = cfg.ENTITIES.index(self.emitter)
        return f"datasets/interferences_caused_by_{index}_{self.method}{cfg.model_segment(self.model)}.json"

    def _compute_from_scratch(self) -> Dict[str, Dict[str, float]]:
        raise ArtifactNotAvailableError(
            "InterferencePerPair is produced by pipeline_06. Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict) and len(data) == 71

    def compute(self) -> Dict[str, Dict[str, float]]:
        return cast(Dict[str, Dict[str, float]], self._resolve())

class InterferencePerEntity(MetricEffectPerEntity):
    """The per-entity summary: each entity's metadata plus metric_{method}_{fragment} (arrow)
    columns. Produced by pipeline_07."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"

    def _get_data_path_remote(self) -> str:
        return f"interference_per_entity{cfg.model_segment(self.model)}.json"

    def _compute_from_scratch(self) -> list:
        raise ArtifactNotAvailableError(
            "InterferencePerEntity is produced by pipeline_07. Provide the local file or fetch it from HuggingFace.")

    def compute(self) -> list:
        return cast(list, self._resolve())


class BaselineAccuracy(SingleFileArtifact):
    """Un-unlearned classifier grid: {receiver: {accuracy, target_probability}}, 71 keys.
    No method and no emitter field, by construction — a baseline cannot depend on a method."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"

    def _get_data_path_remote(self) -> str:
        return "datasets/accuracies_original.json"

    def _compute_from_scratch(self) -> Dict[str, Dict[str, float]]:
        raise ArtifactNotAvailableError(
            "BaselineAccuracy is produced by pipeline_06 over the baseline answer set. "
            "Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict) and len(data) == 71

    def compute(self) -> Dict[str, Dict[str, float]]:
        return cast(Dict[str, Dict[str, float]], self._resolve())
