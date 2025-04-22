import os
import shutil
from typing import Dict, List, Optional
import requests
from PIL import Image, ImageFile
from io import BytesIO
from huggingface_hub import hf_api, HfApi, snapshot_download
from vision_unlearning.utils.logger import get_logger


logger = get_logger('integrations')


def huggingface_model_upload(
    folder_models: str,
    model_repository: str,
    model_config: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    '''
    Upload an entire folder or specific model config in one single commit
    When model_config is None, uploads entire contents of folder_models
    Supposes that the folder exists in `folder_models`, and that it contains the model files
    '''
    folder_model = folder_models if model_config is None else os.path.join(folder_models, model_config)
    # TODO: merge this func with the upload dataset
    # TODO: each config/version should be immutable... should this be ensured here?
    assert os.path.exists(folder_model)

    # TODO: upload_large_folder is better, but don't allow to set the path_in_repo
    # This can be solved by creating tje folder locally (with the path_in_repo inside), and then uploading the contents
    api = HfApi()
    api.upload_folder(
        folder_path=folder_model,
        repo_id=model_repository,
        path_in_repo=model_config,
        repo_type='model',
        token=token,
    )


def huggingface_model_download(
    folder_models: str,
    model_repository: str,
    model_config: Optional[str] = None,
    token: Optional[str] = None,
    clean: bool = False,
) -> None:
    '''
    Download a model or specific model config from Hugging Face Hub.

    Args:
        folder_models: Local directory to save the model
        model_repository: Hugging Face repository ID
        model_config: Specific model config to download (None for entire repository)
        token: Hugging Face authentication token
        clean: If True, the folder will be deleted before downloading
    '''
    folder_model = os.path.join(folder_models, model_config) if model_config else folder_models
    if clean and os.path.exists(folder_model):
        shutil.rmtree(folder_model)
    if os.path.exists(folder_model):
        logger.info('Model already exists locally, skipping download')
        return
    os.makedirs(folder_model, exist_ok=True)

    # Download to cache
    folder_cache = '/tmp/huggingface_cache'
    folder_cache_model = os.path.join(folder_cache, model_repository)
    if model_config:
        folder_cache_model = os.path.join(folder_cache_model, model_config)
    os.makedirs(folder_cache_model, exist_ok=True)

    if model_config:
        repo_path = snapshot_download(
            repo_id=model_repository,
            repo_type="model",
            token=token,
            allow_patterns=f"{model_config}/*",
            cache_dir=folder_cache,
        )
    else:
        repo_path = snapshot_download(
            repo_id=model_repository,
            repo_type="model",
            token=token,
            cache_dir=folder_cache,
        )

    # Copy from cache to final folder
    source_path = repo_path if not model_config else os.path.join(repo_path, model_config)
    for root, _, files in os.walk(source_path):
        for file in files:
            file_source_path = os.path.join(root, file)
            if os.path.islink(file_source_path):
                file_source_path = os.path.join(root, os.readlink(file_source_path))
            rel_path = os.path.relpath(os.path.join(root, file), start=source_path)
            target_path = os.path.join(folder_model, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(file_source_path, target_path)


def huggingface_dataset_upload(
    folder_datasets: str,
    dataset_repository: str,
    dataset_config: str,
    token: str,
):
    '''
    Supposes that a folder `dataset_config` exists in `folder_datasets`, and that it contains the dataset files
    '''
    folder_dataset = os.path.join(folder_datasets, dataset_config)
    assert os.path.exists(folder_dataset)

    # TODO: each config/version should be immutable... should this be ensured here?
    # TODO: upload_large_folder is better, but don't allow to set the path_in_repo
    # This can be solved by creating tje folder locally (with the path_in_repo inside), and then uploading the contents
    api = HfApi()
    api.upload_folder(
        folder_path=folder_dataset,
        repo_id=dataset_repository,
        path_in_repo=dataset_config,
        repo_type='dataset',
        token=token,
    )


def huggingface_dataset_download(
    folder_datasets: str,
    dataset_repository: str,
    dataset_config: str,
    token: str,
    clean: bool = False,
    folder_cache: str = '/tmp/huggingface_cache',
    clean_cache: bool = False,
):
    '''
    @param clean: If True, the folder will be deleted before downloading
    '''
    folder_dataset = os.path.join(folder_datasets, dataset_config)
    if clean:
        if os.path.exists(folder_dataset):
            shutil.rmtree(folder_dataset)
    if os.path.exists(folder_dataset):
        logger.info('Dataset already exists locally, skipping download')
        return
    os.makedirs(folder_dataset)

    folder_cache_dataset = os.path.join(folder_cache, dataset_repository, dataset_config)
    os.makedirs(folder_cache_dataset, exist_ok=True)

    # Download to cache
    repo_path = snapshot_download(
        repo_id=dataset_repository,
        repo_type="dataset",
        token=token,
        allow_patterns=f"{dataset_config}/*",
        cache_dir=folder_cache,
    )

    # Copy from cache to final folder
    for root, _, files in os.walk(os.path.join(repo_path, dataset_config)):
        for file in files:
            source_path = os.path.join(root, file)
            if os.path.islink(source_path):
                source_path = os.path.join(root, os.readlink(source_path))
            target_path = os.path.join(folder_dataset, os.path.relpath(os.path.join(root, file), start=os.path.join(repo_path, dataset_config)))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    # Remove cache
    if clean_cache:
        shutil.rmtree(repo_path)


def huggingface_get_model_metrics(model_id: str) -> Dict[str, float | int | bool]:
    '''
    Supposes that the credentials are properly configured
    '''
    api = hf_api.HfApi()
    name_to_value = {}
    model_info = api.model_info(model_id)
    if model_info.cardData and model_info.cardData.eval_results:
        for result in model_info.cardData.eval_results:
            name_to_value[str(result.metric_name)] = result.metric_value
    else:
        logger.info(f"No metrics found for {model_id}")
    return name_to_value


def huggingface_get_model_images(model_id, prefix: str = '') -> List[ImageFile]:
    '''
    Searches in anything starting with `prefix`
    '''
    images: List[ImageFile] = []
    api = hf_api.HfApi()
    model_info = api.model_info(model_id)
    if model_info.siblings:
        for sibling in model_info.siblings:
            if sibling.rfilename.endswith(('.png', '.jpg', '.jpeg', '.gif')) and sibling.rfilename.startswith(prefix):
                logger.info(f"Image: {sibling.rfilename}")
                response = requests.get(f"https://huggingface.co/{model_id}/resolve/main/{sibling.rfilename}")
                images.append(Image.open(BytesIO(response.content)))
    else:
        logger.info(f"No files found in the repository {model_id}")
    return images
