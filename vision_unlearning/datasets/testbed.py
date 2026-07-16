from __future__ import annotations

from typing import Literal, Tuple, List, Dict, Optional, Any, cast
import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, model_validator, PrivateAttr

from vision_unlearning.utils.logger import get_logger
from vision_unlearning.artifact import Artifact, SingleFileArtifact


logger = get_logger('testbed')

# Type aliases used by GeneratedDataset — imported lazily to avoid a hard
# dependency on the benchmarks sub-package for callers that only need the
# lower-level helper functions.
_type_task = Literal['breeds', 'scenes', 'people']
_type_method = Literal['distil', 'munba', 'uce']
_type_model = Literal['sd1.4']


def _model_segment(model: _type_model) -> str:
    """Filename/folder segment for the base model.

    ``sd1.4`` (the only produced model) returns an empty segment so every existing
    dataset/model folder name is unchanged; any other base model adds a disambiguating
    ``_{model}`` segment. Kept local — mirroring ``configuration.model_segment`` — so this
    module stays decoupled from the ``benchmarks.I_care`` sub-package, exactly as the task
    and method aliases above are. The two must stay in sync if a real second model is added.
    """
    return '' if model == 'sd1.4' else f'_{model}'


##########################################
# Target handling
##########################################
def get_target_preprocessed(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    target: str,
) -> str:
    target_preprocessed: str
    # TODO THIS SHOULD FOLLOW THE RULES CURRENTLY CODEDE AT get_target_overwrite!!!!!!!!!!
    if task == 'people':
        target_preprocessed = target  # TODO
    elif task == 'breeds':
        target_preprocessed = target  # TODO
    elif task == 'scenes':
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target_preprocessed = f"{article} {target} scene"
    else:
        raise NotImplementedError()

    return target_preprocessed


def get_target_overwrite(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    target: str,
) -> Tuple[str, str]:
    '''
    @return preprocessed target, target_overwrite
    '''
    # TODO THIS SHOULD USE  get_target_preprocessed FOR THE TARGET!!!!
    if task == 'people':
        # target does NOT need to have an article,for example: picture of brad pitt

        # target_race = metadata_filtered[index]['race'].replace('indian_middleEastern_latinoHispanic', 'middle eastern') # enum: white, asian, black, indian_middleEastern_latinoHispanic
        # target_gender = 'male' if metadata_filtered[index]['gender']=='M' else 'female'  # Enum[M, F]
        # article = 'an' if (target_race[0].lower() in 'aeiou') else 'a'
        # target_overwrite = f"{article} {target_race} {target_gender}"  # For munba this is only the retain concept for final evaluation, there is no overwriting
        target_overwrite = 'a child'
    elif task == 'breeds':
        # target does needs to have an article,for example: picture of a poodle
        target_overwrite = 'a cat'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        #target = re.sub(r'\bdog\b', '', target, flags=re.IGNORECASE)
        target = f"{article} {target}"
    elif task == 'scenes':
        # target does needs to have an article,for example: picture of a phone_booth
        target_overwrite = 'the moon'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target = f"{article} {target} scene"

    else:
        raise NotImplementedError()

    target = target.replace('_', ' ')
    target = re.sub(r'\s+', ' ', target).strip()

    assert isinstance(target_overwrite, str)
    assert isinstance(target, str)
    assert len(target) >= 3

    return target, target_overwrite


##########################################
# Metadata filtered
##########################################
def _metadata_filtered_filename(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
) -> str:
    return f"metadata_{task}_2_enriched_filtered.json"


def get_metadata_filtered_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, _metadata_filtered_filename(task))


class MetadataFiltered(SingleFileArtifact):
    """Object-oriented interface over the filtered task metadata file.

    Complements the get_metadata_filtered / save_metadata_filtered helpers by adding the
    shared local -> HuggingFace -> (not-computed-on-demand) storage cascade, so a caller can
    fetch the metadata from HuggingFace when it is absent locally.
    """
    task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'

    def _get_data_path_remote(self) -> str:
        return _metadata_filtered_filename(self.task)

    def _compute_from_scratch(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "MetadataFiltered is produced by the data-preparation pipeline, not computed on "
            "demand. Provide the local file or fetch it from HuggingFace."
        )

    def _validate(self, data: Any) -> None:
        assert isinstance(data, list)

    def compute(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self._resolve())


def get_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets'
) -> List[Dict[str, Any]]:
    return MetadataFiltered(task=task, base_folder=base_folder).compute()


def save_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    metadata_filtered: List[Dict[str, Any]],
    base_folder: str = 'assets',
):
    assert isinstance(metadata_filtered, list)
    assert all(isinstance(item, dict) for item in metadata_filtered)
    assert len(metadata_filtered) > 0, "metadata_filtered should not be empty"
    with open(get_metadata_filtered_path(task, base_folder=base_folder), "w", encoding="utf-8") as f:
        json.dump(metadata_filtered, f, indent=4)


def exists_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> bool:
    return os.path.exists(get_metadata_filtered_path(task, base_folder=base_folder))


def get_attribute_for_entity(
    metadata_filtered: List[Dict[str, Any]],
    entity_name: str, 
    attribute: str,
) -> Any:
    for item in metadata_filtered:
        if item['name'] == entity_name:
            return item[attribute]
    return None

##########################################
# Training dataset
##########################################
task_to_dataset_map: Dict[Literal['scenes', 'objects', 'breeds', 'people'], str] = {  # Do not include base_folder
    'scenes': 'datasets/SUN_splits_filtered',
    'breeds': 'datasets/taras_breeds_splits_filtered',
    'people': 'datasets/lfw_splits_filtered',
}


##########################################
# Unlearned model
##########################################
def get_unlearned_model_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
    base_folder: str = 'assets',
    model: _type_model = 'sd1.4',
) -> str:
    # By convention, I'm passing here the NON preprocessed target
    return os.path.join(base_folder, 'models', f"{task}_{target}_{method}_{num_train_epochs:03d}{_model_segment(model)}")


def exists_unlearned_model(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
    base_folder: str = 'assets',
    model: _type_model = 'sd1.4',
) -> bool:
    model_path = get_unlearned_model_folder(task, method, num_train_epochs, target, base_folder=base_folder, model=model)
    if method == 'uce':
        return os.path.exists(os.path.join(model_path, 'uce_sd_weights.safetensors'))
    else:
        return os.path.exists(os.path.join(model_path, 'pytorch_lora_weights.safetensors'))


##########################################
# Generated data
##########################################
def get_generated_dataset_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
    base_folder: str = 'assets',
    model: _type_model = 'sd1.4',
) -> str:
    # By convention, I'm passing here the preprocessed target... TODO change?
    return os.path.join(base_folder, "datasets", f"generated_{task}_{target}_{method}_{num_train_epochs:03d}{_model_segment(model)}")


def get_generated_dataset_file(
    lora_state: Literal['on', 'off'],
    seed: int,
    prompt: str,
) -> str:
    return f'{lora_state}_{seed:02}_{prompt}.png'


def exists_unlearned_dataset(
    generated_dataset_output_path: str,
    generate_dataset_seeds: List[int],
    prompts: List[str],
) -> bool:
    """Return True if the entity dataset folder contains all expected on_* images.

    Only on_* (unlearned model) images are counted. off_* files that may exist in
    legacy entity folders (pre-baseline-refactor data) are ignored so that old datasets
    remain valid without requiring a re-generation pass.

    Baseline lora_state='off' images live in the shared baseline folder; see
    get_shared_baseline_folder() and get_off_image_path().

    Expected: len(seeds) * len(prompts) on_*.png files + 1 metadata.jsonl.
    """
    if not os.path.exists(generated_dataset_output_path):
        return False

    all_files = os.listdir(generated_dataset_output_path)
    all_files = [f for f in all_files if f != '.ipynb_checkpoints']

    # Count only on_*.png images; off_* files (legacy) are intentionally ignored.
    on_images = [f for f in all_files if f.startswith('on_') and f.endswith('.png')]
    has_metadata = any(f.endswith('.jsonl') for f in all_files)

    expected_on_count = len(generate_dataset_seeds) * len(prompts)
    return len(on_images) == expected_on_count and has_metadata


def get_shared_baseline_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
    model: _type_model = 'sd1.4',
) -> str:
    """Return the task-level shared baseline folder path.

    A single shared folder per task holds ALL method-agnostic baseline images
    (generated by 0_generate_dataset_original.py, run once per task, with no LoRA).
    Images are independent of which entity is being forgotten, so one folder serves
    all entities and all methods.

    Convention: assets/datasets/generated_{task}_baseline/
    """
    return os.path.join(base_folder, "datasets", f"generated_{task}_baseline{_model_segment(model)}")


def get_off_image_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    target: str,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    seed: int,
    prompt: str,
    base_folder: str = 'assets',
    seeds: Optional[List[int]] = None,
    prompts: Optional[List[str]] = None,
    model: _type_model = 'sd1.4',
) -> str:
    """Return the path to a baseline (lora_state='off') image for a given entity/seed/prompt.

    .. note::
        This module-level function and ``GeneratedDataset.get_off_image_path()``
        (classmethod) provide identical functionality.  Both exist because the
        module-level version predates the ``GeneratedDataset`` class; the classmethod
        delegates to this function.  New code should prefer the classmethod for
        consistency with the OO abstraction, but the module-level function is NOT
        vestigial — it is used by legacy callers and remains the implementation
        backing both entry points.

    Fallback / download cascade:
    1. If the shared task-level baseline folder exists locally, use it (preferred).
    2. If ``seeds`` and ``prompts`` (the *full* task-level lists) are provided and the
       baseline folder is absent locally, attempt to download it from HuggingFace via
       ``GeneratedDataset(task, method=None).compute(seeds, prompts)``.  This mirrors
       the OO cascade: local → HF → scratch.  If HF has the data it is downloaded; if
       not, ``_compute_from_scratch`` is called (which requires the base SD pipeline).
    3. Otherwise fall back to the legacy entity folder (get_generated_dataset_folder),
       which was the pre-refactor location for both on_* and off_* images.

    Parameters
    ----------
    task, target, method, num_train_epochs:
        Used for the legacy entity-folder fallback (step 3) and to identify the baseline.
    seed, prompt:
        Identify the specific image file to return.
    base_folder:
        Root assets directory.
    seeds, prompts:
        Full task-level seed and prompt lists — required for ``exists()`` and
        ``compute()`` on the shared baseline.  When provided the function will
        attempt an HF download if the baseline folder is missing locally (step 2).
        If omitted, the function skips the download attempt and falls back directly
        to the entity folder (backward-compatible).
    """
    # 1. Shared task-level baseline already present locally.
    shared_folder = get_shared_baseline_folder(task, base_folder, model)
    if os.path.exists(shared_folder):
        path = os.path.join(shared_folder, get_generated_dataset_file('off', seed, prompt))
        if not os.path.exists(path):
            raise ValueError(
                f"Baseline image not found in shared folder: {path}. "
                f"The baseline folder exists at {shared_folder} but does not contain "
                f"an image for seed={seed!r}, prompt={prompt!r}. "
                f"This usually means the baseline was computed with a different "
                f"seed list or prompt list than what you are requesting. "
                f"Re-run '0_generate_dataset_original.py' with the correct seeds and prompts."
            )
        return path

    # 2. Baseline absent locally — attempt HF download when caller provides full seed/prompt lists.
    # Only supported for tasks known to GeneratedDataset (_type_task).  'objects' is excluded
    # because GeneratedDataset does not yet cover that task variant.
    if seeds is not None and prompts is not None and task in ('breeds', 'scenes', 'people'):
        # Use the class directly since it is in the same module.
        # cast: task is narrowed to _type_task by the guard above.
        from typing import cast as _cast  # noqa: PLC0415
        baseline_ds = GeneratedDataset(task=_cast(_type_task, task), base_folder=base_folder, model=model)
        baseline_ds.compute(seeds, prompts)
        # After compute() the shared folder should now exist locally.
        if os.path.exists(shared_folder):
            return os.path.join(shared_folder, get_generated_dataset_file('off', seed, prompt))

    # 3. Entity folder fallback (pre-refactor mixed on_* + off_* folder).
    entity_folder = get_generated_dataset_folder(task, method, num_train_epochs, target, base_folder, model)
    return os.path.join(entity_folder, get_generated_dataset_file('off', seed, prompt))


##########################################
# GeneratedDataset — OO abstraction over all image-dataset folder types
##########################################
# TODO: In the future, GeneratedDataset, InterferencePerEntity, all RTs, and
# any other artifact (unlearned models, metadata) should inherit from a common
# base class ``Artifact`` that standardises the local/remote storage contract
# and the compute() lifecycle.  For now both the OO class and the legacy
# helper functions coexist; prefer GeneratedDataset for new code.


class GeneratedDataset(Artifact):
    """Abstraction over generated image dataset folders.

    Represents exactly one dataset folder — either the shared task-level
    baseline or a method-specific entity dataset.

    Folder conventions
    ------------------
    - **Shared baseline** (``method=None``):
        ``assets/datasets/generated_{task}_baseline/``
        All method-agnostic off-images for the whole task live here.
        Generated once per task by 0_generate_dataset_original.py.
    - **Entity dataset** (``method=<str>``, ``target=<str>``):
        ``assets/datasets/generated_{task}_{target}_{method}_{epochs:03d}/``
        Contains ``on_*`` unlearned images (and possibly legacy ``off_*``).

    ``compute()`` resolves data in priority order:
    1. Already complete locally → return immediately.
    2. Present in HuggingFace → download, then return.
    3. Neither → call ``_compute_from_scratch()``, which generates images
       from scratch using the Stable Diffusion pipeline.

    After ``_compute_from_scratch()`` completes, if ``upload_if_recomputed``
    is True the dataset folder is uploaded to HuggingFace.

    The ``get_off_image_path`` class method encapsulates the full fallback
    chain for a baseline image: shared baseline → entity folder.
    """

    task: _type_task
    target: Optional[str] = None          # None → shared baseline (task-level)
    method: Optional[_type_method] = None  # None → baseline dataset
    num_train_epochs: Optional[int] = None
    model: _type_model = 'sd1.4'          # base image-generating model; sd1.4 → unchanged names
    # base_folder, remote_repository_name, recompute_if_exists, upload_if_recomputed and
    # save_outputs are inherited from Artifact. save_outputs is unused here: image
    # generation writes the folder directly, so _persist_local is a no-op.

    # Runtime arguments for one compute() call, stashed so the (nullary) storage hooks can
    # read them.
    _pending_seeds: List[int] = PrivateAttr(default_factory=list)
    _pending_prompts: List[str] = PrivateAttr(default_factory=list)
    _pending_batch_size: int = PrivateAttr(default=16)

    @model_validator(mode='after')
    def _validate_consistency(self) -> 'GeneratedDataset':
        if self.method is not None:
            assert self.target is not None, (
                "target must be set when method is specified (entity dataset)."
            )
            assert self.num_train_epochs is not None, (
                "num_train_epochs must be set when method is specified (entity dataset)."
            )
        if self.method is None and self.target is not None:
            raise ValueError(
                "target must be None for a shared baseline dataset (method=None). "
                "The per-entity baseline concept does not exist in I-CARE: baselines are "
                "shared across all entities. Use target=None for a baseline, or provide "
                "both target and method for an entity dataset."
            )
        return self

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    @property
    def is_baseline(self) -> bool:
        """True when this dataset holds baseline (lora-off) images."""
        return self.method is None

    @property
    def folder_path(self) -> str:
        """Local path to the dataset folder.

        Replaces:
          - get_shared_baseline_folder()
          - get_generated_dataset_folder()
        """
        if self.is_baseline:
            return get_shared_baseline_folder(self.task, self.base_folder, self.model)
        # Entity dataset
        assert self.target is not None
        assert self.num_train_epochs is not None
        assert self.method is not None  # guaranteed by _validate_consistency
        method: _type_method = self.method
        return get_generated_dataset_folder(
            self.task, method, self.num_train_epochs, self.target, self.base_folder, self.model
        )

    @property
    def hf_config_name(self) -> str:
        """HuggingFace config / folder name (basename of folder_path).

        This is the bare folder name used for local path computation.
        Use ``hf_path_in_repo`` when you need the full HF-side path.
        """
        return os.path.basename(self.folder_path)

    @property
    def hf_path_in_repo(self) -> str:
        """Full path inside the HuggingFace repository where this dataset lives.

        All generated datasets (baseline and entity) live under the ``datasets/``
        prefix in the HF repo, matching the convention used by the legacy
        synchronisation notebook (0b. Synchronize.ipynb).

        Example: ``"datasets/generated_breeds_baseline"``
        """
        return f"datasets/{self.hf_config_name}"

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def file_path(
        self,
        lora_state: Literal['on', 'off'],
        seed: int,
        prompt: str,
    ) -> str:
        """Full path to one image file inside this dataset folder.

        Replaces get_generated_dataset_file() when used together with a
        GeneratedDataset instance.

        Note: lora_state='on' is only valid for entity datasets (method set).
              lora_state='off' is valid for all dataset types.
        """
        if lora_state == 'on' and self.is_baseline:
            raise ValueError("Baseline datasets do not contain on_* (unlearned) images.")
        return os.path.join(self.folder_path, get_generated_dataset_file(lora_state, seed, prompt))

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, seeds: List[int], prompts: List[str]) -> bool:
        """Return True if all expected images and metadata are present locally.

        For entity datasets, only on_* images are counted (off_* legacy files
        are ignored — same contract as exists_unlearned_dataset()).
        For baseline datasets, only off_* images are counted.

        Replaces exists_unlearned_dataset() for entity datasets and provides
        the equivalent for baseline folders.

        WARNING — shared baseline: The shared baseline folder contains images for
        ALL entities in the task (N_entities * len(seeds) images total), not just
        the entities in the ``prompts`` argument.  This method counts existing
        off_* files and compares against ``len(seeds) * len(prompts)``.

        If ``prompts`` is a partial (subset) list of the full task prompts,
        ``exists()`` will count more images than expected and incorrectly return
        False, triggering a full re-generation.  Always pass the COMPLETE prompt
        list for the task when calling ``exists()`` on a shared baseline dataset.

        For entity datasets this restriction does not apply because the entity
        folder contains only the images for that specific entity.
        """
        if not os.path.exists(self.folder_path):
            return False

        all_files = [
            f for f in os.listdir(self.folder_path)
            if f != '.ipynb_checkpoints'
        ]
        has_metadata = any(f.endswith('.jsonl') for f in all_files)

        expected_count = len(seeds) * len(prompts)

        if self.is_baseline:
            image_files = [f for f in all_files if f.startswith('off_') and f.endswith('.png')]
        else:
            # Entity dataset: count only on_* images (off_* may exist in legacy folders).
            image_files = [f for f in all_files if f.startswith('on_') and f.endswith('.png')]

        return len(image_files) == expected_count and has_metadata

    # ------------------------------------------------------------------
    # Compute (local → HF → scratch)
    # ------------------------------------------------------------------

    def _compute_from_scratch(
        self,
        seeds: List[int],
        prompts: List[str],
        batch_size: int = 16,
    ) -> str:
        """Generate images from scratch and return the folder path.

        For the shared baseline (method=None): loads the base SD pipeline once
        and generates all off-images for all (seed, prompt) pairs, storing them
        in folder_path with the ``off_{seed}_{prompt}.png`` filename convention.

        For entity datasets (method set): loads the already-trained unlearned
        model identified by (task, target, method, num_train_epochs) and generates
        on-images.  Raises FileNotFoundError if the trained model does not exist on
        disk — the caller must run 1_unlearn_from_metadata.py first to produce the
        model weights before calling compute().

        In both cases the method returns self.folder_path after generation.

        Note on metadata.jsonl (entity datasets): ``generate_dataset()`` writes
        ``metadata.jsonl`` to ``self.folder_path`` as its last step.  This is
        verified end-to-end for the shared baseline path.  For entity datasets,
        ``generate_dataset()`` itself writes the file in both the LoRA and UCE
        paths (see vision_unlearning/utils/data_generation.py line 165), but the
        unit tests for this method mock ``generate_dataset`` and therefore do not
        exercise the actual file write.  If the ``generate_dataset`` implementation
        changes and stops writing ``metadata.jsonl``, the entity path here would
        silently produce an incomplete dataset.

        Parameters
        ----------
        seeds : list of int
            Generation seeds.
        prompts : list of str
            Text prompts — one per image template, excluding seed variation.
        batch_size : int
            Number of prompts per pipeline call.  Default 16 (optimal for 8–12 GB
            VRAM on this hardware; see batch-size profiling in the project notes).
        """
        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from vision_unlearning.utils.data_generation import generate_dataset  # noqa: PLC0415

        model_base_name = "CompVis/stable-diffusion-v1-4"
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.is_baseline:
            # ------------------------------------------------------------------
            # Shared baseline: generate off-images using the base SD pipeline.
            # ------------------------------------------------------------------
            from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

            logger.info("_compute_from_scratch: loading base pipeline %s on %s", model_base_name, device)
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_base_name,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(device)
            logger.info("_compute_from_scratch: pipeline loaded; generating %d images", len(seeds) * len(prompts))

            filenames = [
                f'off_{seed}_{prompt}.png'
                for seed in seeds
                for prompt in prompts
            ]

            generate_dataset(
                model_base_name=None,
                lora_name=None,
                model_pipeline=pipeline,
                prompts=prompts,
                output_path=self.folder_path,
                seeds=seeds,
                filenames=filenames,
                batch_size=batch_size,
            )

        else:
            # ------------------------------------------------------------------
            # Entity dataset: load the already-trained unlearned model and
            # generate on-images.  Raise FileNotFoundError if model is missing.
            # ------------------------------------------------------------------
            assert self.target is not None
            assert self.method is not None
            assert self.num_train_epochs is not None

            model_folder = get_unlearned_model_folder(
                self.task, self.method, self.num_train_epochs, self.target, self.base_folder, self.model
            )
            if not exists_unlearned_model(
                self.task, self.method, self.num_train_epochs, self.target, self.base_folder, self.model
            ):
                raise FileNotFoundError(
                    f"Trained unlearned model not found at '{model_folder}'. "
                    f"Run 1_unlearn_from_metadata.py first to produce the model weights "
                    f"for task={self.task!r}, target={self.target!r}, "
                    f"method={self.method!r}, epochs={self.num_train_epochs}."
                )

            filenames = [
                f'on_{seed}_{prompt}.png'
                for seed in seeds
                for prompt in prompts
            ]

            if self.method == 'uce':
                from vision_unlearning.unlearner.uce_sd_erase import UCE  # noqa: PLC0415
                logger.info(
                    "_compute_from_scratch: loading UCE pipeline from %s on %s",
                    model_folder, device,
                )
                pipeline = UCE.get_pipeline_from_modified_weights(
                    pretrained_model_name_or_path=model_base_name,
                    device=device,
                    output_dir=model_folder,
                )
                logger.info(
                    "_compute_from_scratch: UCE pipeline loaded; generating %d images",
                    len(seeds) * len(prompts),
                )
                generate_dataset(
                    model_base_name=None,
                    lora_name=None,
                    model_pipeline=pipeline,
                    prompts=prompts,
                    output_path=self.folder_path,
                    seeds=seeds,
                    filenames=filenames,
                    batch_size=batch_size,
                )
            else:
                # distil / munba — LoRA-based methods.
                logger.info(
                    "_compute_from_scratch: loading %s LoRA from %s on %s",
                    self.method, model_folder, device,
                )
                generate_dataset(
                    model_base_name=model_base_name,
                    lora_name=model_folder,
                    model_pipeline=None,
                    prompts=prompts,
                    output_path=self.folder_path,
                    seeds=seeds,
                    filenames=filenames,
                    lora_requires_inversion=(self.method == 'munba'),
                    batch_size=batch_size,
                )

        logger.info("_compute_from_scratch: done — folder %s", self.folder_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return self.folder_path

    def compute(self, seeds: List[int], prompts: List[str], batch_size: int = 16) -> str:
        """Ensure the dataset is available locally and return its folder path.

        Resolution order:
        1. Already complete locally → return immediately.
        2. Present in HuggingFace → download, return.
        3. Neither → call ``_compute_from_scratch()``.
           After generation completes, if ``upload_if_recomputed=True``, upload
           the folder to HuggingFace.

        Parameters
        ----------
        seeds : list of int
            Generation seeds.
        prompts : list of str
            Text prompts.  For shared baseline datasets, this MUST be the complete
            prompt list for the task (all entities).  Passing a partial list will
            cause ``exists()`` to return False and trigger unnecessary re-generation.
            See ``exists()`` docstring for details.
        batch_size : int
            Prompts per pipeline call, forwarded to ``_compute_from_scratch()``.
            Ignored if the data is already available locally or on HuggingFace.
            Default 16 (optimal for 8–12 GB VRAM; see batch-size profiling in
            the project notes).

        Returns
        -------
        str
            The local folder path to the (now complete) dataset.
        """
        self._pending_seeds = seeds
        self._pending_prompts = prompts
        self._pending_batch_size = batch_size
        return cast(str, self._resolve())

    # ------------------------------------------------------------------
    # Artifact storage hooks (folder-of-images shape)
    # ------------------------------------------------------------------
    def _exists_local(self) -> bool:
        return self.exists(self._pending_seeds, self._pending_prompts)

    def _exists_remote(self, hf_token: Optional[str]) -> bool:
        # Unguarded on purpose: unlike a cheap single-file recompute, regenerating a whole
        # image dataset is expensive, so a transient network error should surface rather than
        # silently trigger a from-scratch regeneration.
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
            folder_datasets=os.path.join(self.base_folder, 'datasets'),
            dataset_repository=self.remote_repository_name,
            dataset_config=self.hf_config_name,
            # None (not "") when unset: an empty string becomes an illegal
            # 'Authorization: Bearer ' header and breaks unauthenticated
            # downloads from public repositories.
            token=hf_token,
            path_in_repo=self.hf_path_in_repo,
        )

    def _load_local(self) -> str:
        return self.folder_path

    def _produce_from_scratch(self) -> str:
        result = self._compute_from_scratch(
            self._pending_seeds, self._pending_prompts, batch_size=self._pending_batch_size
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
        return None  # image generation writes the folder directly; nothing to persist

    def _push_remote(self, hf_token: Optional[str]) -> None:
        # _resolve only calls _push_remote after asserting a token is present.
        assert hf_token is not None
        from vision_unlearning.integrations.huggingface import (  # noqa: PLC0415
            huggingface_dataset_upload,
        )
        logger.info(
            "Uploading recomputed dataset to HF: %s -> %s",
            self.hf_config_name, self.hf_path_in_repo,
        )
        huggingface_dataset_upload(
            folder_datasets=os.path.join(self.base_folder, 'datasets'),
            dataset_repository=self.remote_repository_name,
            dataset_config=self.hf_config_name,
            token=hf_token,
            path_in_repo=self.hf_path_in_repo,
        )
        logger.info("Upload complete: %s", self.hf_path_in_repo)

    # ------------------------------------------------------------------
    # Off-image path with fallback (class-level utility)
    # ------------------------------------------------------------------

    @classmethod
    def get_off_image_path(
        cls,
        task: _type_task,
        target: str,
        method: _type_method,
        num_train_epochs: int,
        seed: int,
        prompt: str,
        base_folder: str = 'assets',
        seeds: Optional[List[int]] = None,
        prompts: Optional[List[str]] = None,
        model: _type_model = 'sd1.4',
    ) -> str:
        """Return the path to a baseline (lora_state='off') image.

        Fallback / download cascade:
        1. Shared task-level baseline folder present locally (preferred).
        2. If ``seeds`` and ``prompts`` (full task-level lists) are provided and the
           baseline folder is absent, download it from HuggingFace via
           ``GeneratedDataset(task, method=None).compute(seeds, prompts)``.
        3. Legacy entity folder (pre-refactor mixed on_* + off_* format).

        This class method delegates to the module-level get_off_image_path() which
        implements the same cascade.  Both exist; prefer this classmethod for new
        code using GeneratedDataset.

        Parameters
        ----------
        task, target, method, num_train_epochs:
            Used for the legacy entity-folder fallback (step 3).
        seed, prompt:
            Identify the specific image file.
        base_folder:
            Root assets directory.
        seeds, prompts:
            Full task-level seed and prompt lists.  Required to enable the HF
            download cascade (step 2).  When omitted the function falls back
            directly to the entity folder (backward-compatible).
        """
        return get_off_image_path(
            task=task,
            target=target,
            method=method,
            num_train_epochs=num_train_epochs,
            seed=seed,
            prompt=prompt,
            base_folder=base_folder,
            seeds=seeds,
            prompts=prompts,
            model=model,
        )


##########################################
# Similarity
##########################################
def get_similarity_clip_path(  # deprecated, use ResultTemplateSimilarityMatrix._get_path
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, f"similarity_clip_{task}.json")  # TODO: this should be in the results folder


def get_similarity_clip_df(  # deprecated, use ResultTemplateSimilarityMatrix._get_similarity_df
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> pd.DataFrame:
    df_similarities_clip = pd.read_json(get_similarity_clip_path(task, base_folder=base_folder), orient='records')
    df_similarities_clip.set_index('emitter', inplace=True)
    return df_similarities_clip


def calculate_similarity_clip(  # deprecated, use ResultTemplateSimilarityMatrix._calculate
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    labels: List[str],
    base_folder: str = 'assets',
) -> pd.DataFrame:
    # Lazy import: torchmetrics/torch only needed when this function is called.
    from vision_unlearning.metrics.text_and_text import MetricTextTextSimilarity  # noqa: PLC0415
    clip_text_metric = MetricTextTextSimilarity(metrics=['clip_text'])

    # Load existing
    if os.path.exists(get_similarity_clip_path(task, base_folder=base_folder)):
        df_similarities_clip = get_similarity_clip_df(task, base_folder=base_folder)
        assert df_similarities_clip.index.to_list() == labels
    else:
        df_similarities_clip = pd.DataFrame(index=labels, columns=labels)

    # Calculate
    for entity_emitter, row_emitter in df_similarities_clip.iterrows():
        # break
        print(f'Analying similarities for entity_emitter={entity_emitter}')
        for entity_receiver in row_emitter.index:
            if pd.isna(df_similarities_clip.loc[entity_emitter, entity_receiver]):  # type: ignore
                similarity: float = clip_text_metric.score(
                    get_target_preprocessed(task, str(entity_emitter)),
                    get_target_preprocessed(task, str(entity_receiver)),
                )['clip_text']
                df_similarities_clip.loc[entity_emitter, entity_receiver] = similarity

        # Save at the end of each row
        df_similarities_clip.reset_index(names='emitter').to_json(get_similarity_clip_path(task, base_folder=base_folder), orient='records')

    return df_similarities_clip


def plot_heatmap(df, figsize=None, cmap="viridis", title="Heatmap"):  # deprecated, use ResultTemplateSimilarityMatrix._plot_heatmap
    """
    Plot a heatmap for a square DataFrame with all labels visible.

    Parameters
    ----------
    df : pd.DataFrame
        A square DataFrame with same string labels for index and columns.
    figsize : tuple
        Figure size (width, height). Increase if labels overlap.
    cmap : str
        Colormap name for matplotlib.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square (same number of rows and columns).")
    if not np.all(df.index == df.columns):
        # logger.warning("Index and columns differ; continuing but axis labels may mismatch.")
        raise ValueError("Index and columns must be the same")

    df2 = df.dropna()
    if figsize is None:
        figsize = (int(0.2 * df2.shape[1]), int(0.18 * df2.shape[0]))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df2.values, cmap=cmap, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(np.arange(df2.shape[1]))
    ax.set_yticks(np.arange(df2.shape[0]))
    ax.set_xticklabels(df2.columns.to_list(), rotation=90, fontsize=5)
    ax.set_yticklabels(df2.index.to_list(), fontsize=5)

    ax.set_xlabel("Columns", fontsize=8)
    ax.set_ylabel("Index", fontsize=8)
    ax.set_title(title, fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.show()
