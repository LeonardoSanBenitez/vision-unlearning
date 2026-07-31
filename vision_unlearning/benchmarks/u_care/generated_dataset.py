import os
from typing import Any, List, Optional, Literal, cast

from pydantic import model_validator

from vision_unlearning.artifact import Artifact, ArtifactNotAvailableError
from vision_unlearning.utils.logger import get_logger
from vision_unlearning.benchmarks.u_care import configuration as cfg

logger = get_logger('generated_dataset')


class GeneratedDataset(Artifact):
    """One answer-set folder: the full 51 x 20 prompt grid, over a set of seeds.
    Baseline (emitter=None, method=None):  assets/datasets/generated_baseline_sd_style50/
    Per emitter:                            assets/datasets/generated_{emitter}_{method}_sd_style50/
    Image files: {on|off}_{seed:02}_{prompt}.png   ('off' baseline, 'on' unlearned)
    """
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    emitter: Optional[str] = None
    method: Optional[cfg.type_unlearning_algorithm] = None

    # Runtime arguments for one compute() call, stashed so the nullary storage hooks
    # can still read them during the Artifact lifecycle.
    _pending_seeds: List[int] = []
    _pending_prompts: List[str] = []
    _pending_batch_size: int = 16

    @model_validator(mode="after")
    def _both_or_neither(self) -> "GeneratedDataset":
        if (self.emitter is None) != (self.method is None):
            raise ValueError(
                "emitter and method must both be set (per-emitter answer set) or both be None "
                f"(baseline). Got emitter={self.emitter!r}, method={self.method!r}.")
        return self

    @property
    def folder_path(self) -> str:
        if self.emitter is None and self.method is None:
            return os.path.join(self.base_folder, "datasets", f"generated_baseline_{self.model}")
        return os.path.join(
            self.base_folder,
            "datasets",
            f"generated_{self.emitter}_{self.method}_{self.model}",
        )

    @property
    def hf_config_name(self) -> str:
        return os.path.basename(self.folder_path)

    @property
    def hf_path_in_repo(self) -> str:
        return f"datasets/{self.hf_config_name}"

    def file_path(self, unlearning_state: Literal["on", "off"], seed: int, prompt: str) -> str:
        if self.emitter is None and self.method is None and unlearning_state == "on":
            raise ValueError("Baseline dataset has no 'on' (unlearned) images.")
        return os.path.join(self.folder_path, f"{unlearning_state}_{seed:02}_{prompt}.png")

    def exists(self, seeds: List[int], prompts: List[str]) -> bool:
        if not os.path.exists(self.folder_path):
            return False

        all_files = [
            f for f in os.listdir(self.folder_path)
            if f != '.ipynb_checkpoints'
        ]

        has_metadata = any(f.endswith(".jsonl") for f in all_files)
        expected_count = len(seeds) * len(prompts)

        if self.emitter is None and self.method is None:
            image_files = [f for f in all_files if f.startswith("off_") and f.endswith(".png")]
        else:
            image_files = [f for f in all_files if f.startswith("on_") and f.endswith(".png")]

        return len(image_files) == expected_count and has_metadata

    def compute(self, seeds: List[int], prompts: List[str], batch_size: int = 16) -> str:
        self._pending_seeds = seeds
        self._pending_prompts = prompts
        self._pending_batch_size = batch_size
        return cast(str, self._resolve())

    def _exists_local(self) -> bool:
        return self.exists(self._pending_seeds, self._pending_prompts)

    def _exists_remote(self, hf_token: Optional[str]) -> bool:
        from vision_unlearning.integrations.huggingface import (  # noqa: PLC0415
            huggingface_dataset_exists,
        )

        return huggingface_dataset_exists(
            self.remote_repository_name,
            self.hf_config_name,
            token=hf_token,
            path_in_repo=self.hf_path_in_repo,
        )

    def _pull_remote(self, hf_token: Optional[str]) -> None:
        from vision_unlearning.integrations.huggingface import (  # noqa: PLC0415
            huggingface_dataset_download,
        )

        huggingface_dataset_download(
            folder_datasets=os.path.join(self.base_folder, "datasets"),
            dataset_repository=self.remote_repository_name,
            dataset_config=self.hf_config_name,
            token=hf_token,
            path_in_repo=self.hf_path_in_repo,
        )

    def _load_local(self) -> str:
        return self.folder_path

    def _compute_from_scratch(
        self,
        seeds: List[int],
        prompts: List[str],
        batch_size: int = 16,
    ) -> str:
        raise ArtifactNotAvailableError(
            "GeneratedDataset cannot be produced on demand by this class. "
            "Run the u_care dataset-generation pipeline to create this folder."
        )

    def _produce_from_scratch(self) -> str:
        result = self._compute_from_scratch(
            self._pending_seeds,
            self._pending_prompts,
            batch_size=self._pending_batch_size,
        )
        assert result == self.folder_path, (
            "_compute_from_scratch() must return self.folder_path"
        )
        assert self.exists(self._pending_seeds, self._pending_prompts), (
            f"_compute_from_scratch() completed but dataset is still incomplete: "
            f"{self.folder_path}"
        )
        return result

    def _persist_local(self, data: Any) -> None:
        return None

    def _push_remote(self, hf_token: Optional[str]) -> None:
        assert hf_token is not None
        from vision_unlearning.integrations.huggingface import (  # noqa: PLC0415
            huggingface_dataset_upload,
        )
        logger.info(
            "Uploading recomputed dataset to HF: %s -> %s",
            self.hf_config_name,
            self.hf_path_in_repo,
        )
        huggingface_dataset_upload(
            folder_datasets=os.path.join(self.base_folder, 'datasets'),
            dataset_repository=self.remote_repository_name,
            dataset_config=self.hf_config_name,
            token=hf_token,
            path_in_repo=self.hf_path_in_repo,
        )
        logger.info("Upload complete: %s", self.hf_path_in_repo)


