from __future__ import annotations

from typing import Literal, Tuple, List, Dict, Optional, Any
import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, model_validator

from vision_unlearning.utils.logger import get_logger


logger = get_logger('testbed')

# Type aliases used by GeneratedDataset — imported lazily to avoid a hard
# dependency on the benchmarks sub-package for callers that only need the
# lower-level helper functions.
_type_task = Literal['breeds', 'scenes', 'people']
_type_method = Literal['distil', 'munba', 'uce']


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
def get_metadata_filtered_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, f"metadata_{task}_2_enriched_filtered.json")


def get_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets'
) -> List[Dict[str, Any]]:
    with open(get_metadata_filtered_path(task, base_folder=base_folder), "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    return metadata_filtered


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
) -> str:
    # By convention, I'm passing here the NON preprocessed target
    return os.path.join(base_folder, 'models', f"{task}_{target}_{method}_{num_train_epochs:03d}")


def exists_unlearned_model(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
    base_folder: str = 'assets',
) -> bool:
    model_path = get_unlearned_model_folder(task, method, num_train_epochs, target, base_folder=base_folder)
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
) -> str:
    # By convention, I'm passing here the preprocessed target... TODO change?
    return os.path.join(base_folder, "datasets", f"generated_{task}_{target}_{method}_{num_train_epochs:03d}")


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
) -> str:
    """Return the task-level shared baseline folder path.

    A single shared folder per task holds ALL method-agnostic baseline images
    (generated by 0_generate_dataset_original.py, run once per task, with no LoRA).
    Images are independent of which entity is being forgotten, so one folder serves
    all entities and all methods.

    Convention: assets/datasets/generated_{task}_baseline/
    """
    return os.path.join(base_folder, "datasets", f"generated_{task}_baseline")


def get_off_image_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    target: str,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    seed: int,
    prompt: str,
    base_folder: str = 'assets',
) -> str:
    """Return the path to a baseline (lora_state='off') image for a given entity/seed/prompt.

    Fallback logic:
    1. If the shared task-level baseline folder exists, use it (preferred — generated once
       per task by 0_generate_dataset_original.py).
    2. Otherwise fall back to the legacy entity folder (get_generated_dataset_folder),
       which was the pre-refactor location for both on_* and off_* images.
    """
    # 1. Shared task-level baseline (preferred).
    shared_folder = get_shared_baseline_folder(task, base_folder)
    if os.path.exists(shared_folder):
        return os.path.join(shared_folder, get_generated_dataset_file('off', seed, prompt))
    # 2. Entity folder fallback (pre-refactor mixed on_* + off_* folder).
    entity_folder = get_generated_dataset_folder(task, method, num_train_epochs, target, base_folder)
    return os.path.join(entity_folder, get_generated_dataset_file('off', seed, prompt))


##########################################
# GeneratedDataset — OO abstraction over all image-dataset folder types
##########################################
# TODO: In the future, GeneratedDataset, InterferencePerEntity, all RTs, and
# any other artifact (unlearned models, metadata) should inherit from a common
# base class ``Artifact`` that standardises the local/remote storage contract
# and the compute() lifecycle.  For now both the OO class and the legacy
# helper functions coexist; prefer GeneratedDataset for new code.


class GeneratedDataset(BaseModel):
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
    base_folder: str = 'assets'
    remote_repository_name: str = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'
    recompute_if_exists: bool = False
    upload_if_recomputed: bool = False

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
            return get_shared_baseline_folder(self.task, self.base_folder)
        # Entity dataset
        assert self.target is not None
        assert self.num_train_epochs is not None
        assert self.method is not None  # guaranteed by _validate_consistency
        method: _type_method = self.method
        return get_generated_dataset_folder(
            self.task, method, self.num_train_epochs, self.target, self.base_folder
        )

    @property
    def hf_config_name(self) -> str:
        """HuggingFace config / folder name (basename of folder_path)."""
        return os.path.basename(self.folder_path)

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
    ) -> str:
        """Generate images from scratch and return the folder path.

        For the shared baseline (method=None): loads the base SD pipeline once
        and generates all off-images for all (seed, prompt) pairs, storing them
        in folder_path with the ``off_{seed:02d}_{prompt}.png`` filename convention.

        For entity datasets (method set): not implemented — use the standalone
        script 1_unlearn_from_metadata.py which handles LoRA training + generation.
        """
        if not self.is_baseline:
            raise NotImplementedError(
                "GeneratedDataset._compute_from_scratch for entity datasets is not supported "
                "in-process. Use 1_unlearn_from_metadata.py to train and generate entity datasets."
            )

        # Shared baseline generation: load base SD pipeline, generate all images.
        # This replicates the logic in 0_generate_dataset_original.py.
        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
        from vision_unlearning.utils.data_generation import generate_dataset  # noqa: PLC0415

        model_base_name = "CompVis/stable-diffusion-v1-4"
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        )

        logger.info("_compute_from_scratch: done — folder %s", self.folder_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return self.folder_path

    def compute(self, seeds: List[int], prompts: List[str]) -> str:
        """Ensure the dataset is available locally and return its folder path.

        Resolution order:
        1. Already complete locally → return immediately.
        2. Present in HuggingFace → download, return.
        3. Neither → call ``_compute_from_scratch()``.
           After generation completes, if ``upload_if_recomputed=True``, upload
           the folder to HuggingFace.

        Returns
        -------
        str
            The local folder path to the (now complete) dataset.
        """
        hf_token: Optional[str] = os.getenv('HF_TOKEN')

        # 1. Local
        if not self.recompute_if_exists and self.exists(seeds, prompts):
            return self.folder_path

        # 2. Remote (HuggingFace)
        from vision_unlearning.integrations.huggingface import (  # noqa: PLC0415
            huggingface_dataset_exists,
            huggingface_dataset_download,
            huggingface_dataset_upload,
        )
        if not self.recompute_if_exists and huggingface_dataset_exists(
            self.remote_repository_name,
            self.hf_config_name,
            token=hf_token,
        ):
            huggingface_dataset_download(
                folder_datasets=os.path.join(self.base_folder, 'datasets'),
                dataset_repository=self.remote_repository_name,
                dataset_config=self.hf_config_name,
                token=hf_token or '',
            )
            assert self.exists(seeds, prompts), (
                f"HuggingFace download completed but dataset is still incomplete: "
                f"{self.folder_path}"
            )
            return self.folder_path

        # 3. Compute from scratch
        result = self._compute_from_scratch(seeds, prompts)
        assert result == self.folder_path, (
            "_compute_from_scratch() must return self.folder_path"
        )
        assert self.exists(seeds, prompts), (
            f"_compute_from_scratch() completed but dataset is still incomplete: "
            f"{self.folder_path}"
        )

        # Upload to HF if requested
        if self.upload_if_recomputed:
            assert hf_token, (
                "upload_if_recomputed=True but HF_TOKEN is not set. "
                "Set HF_TOKEN environment variable before calling compute()."
            )
            logger.info("Uploading recomputed dataset to HF: %s", self.hf_config_name)
            huggingface_dataset_upload(
                folder_datasets=os.path.join(self.base_folder, 'datasets'),
                dataset_repository=self.remote_repository_name,
                dataset_config=self.hf_config_name,
                token=hf_token,
            )
            logger.info("Upload complete: %s", self.hf_config_name)

        return self.folder_path

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
    ) -> str:
        """Return the path to a baseline (lora_state='off') image.

        Fallback chain:
        1. Shared task-level baseline folder (preferred).
        2. Legacy entity folder (pre-refactor mixed on_* + off_* format).

        This class method delegates to the module-level get_off_image_path().
        Both exist; prefer this one for new code using GeneratedDataset.

        Parameters
        ----------
        task, target, method, num_train_epochs:
            Used only for the legacy entity-folder fallback (step 2).
        seed, prompt:
            Identify the specific image file.
        base_folder:
            Root assets directory.
        """
        return get_off_image_path(
            task=task,
            target=target,
            method=method,
            num_train_epochs=num_train_epochs,
            seed=seed,
            prompt=prompt,
            base_folder=base_folder,
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
