import re
from typing import Any, Dict, List, cast

from vision_unlearning.artifact import ArtifactNotAvailableError, SingleFileArtifact
from vision_unlearning.benchmarks.care import MetricEffectPerEntity, MetricEffectPerEntityPair
from vision_unlearning.benchmarks.u_care import configuration as cfg

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

    def _get_data_path_remote(self) -> str:
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
