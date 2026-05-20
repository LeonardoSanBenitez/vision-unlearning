import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import requests
from PIL import Image, ImageFile
from io import BytesIO
from huggingface_hub import hf_api, HfApi, hf_hub_url, snapshot_download, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
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


def huggingface_dataset_exists(
    dataset_repository: str,
    dataset_config: str,
    token: Optional[str],
) -> bool:
    """
    Checks whether a folder exists in a Hugging Face dataset repository.

    Example:
        dataset_repository="username/my_dataset"
        dataset_config="configs/en"

    Works without listing the whole repository.
    """

    url = (
        f"https://huggingface.co/api/datasets/"
        f"{dataset_repository}/tree/main/{dataset_config.replace(os.sep, '/')}"
    )

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 404:
        return False

    response.raise_for_status()

    # Existing folders return a JSON array of entries.
    return isinstance(response.json(), list)

def huggingface_dataset_file_exists(
    dataset_repository: str,
    dataset_path: str,
    token: Optional[str],
) -> bool:
    """
    Checks if a specific file exists in a Hugging Face dataset repository.

    :param dataset_repository: e.g. "username/dataset_name"
    :param dataset_path: full path in repo (e.g. "config/file.jsonl")
    :param token: HF token (can be None for public repos)
    :return: True if file exists, False otherwise
    Efficiently checks if a file exists in a Hugging Face dataset repo without listing the entire repository.
    Could be done more efficiently if we use a new version of the lib, see https://chatgpt.com/share/69edd525-d008-832d-8a0c-ec4560a4fe3b

    """
    url = hf_hub_url(
        repo_id=dataset_repository,
        filename=dataset_path,
        repo_type="dataset",
    )
    #print('url:', url, flush=True)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.head(url, headers=headers, timeout=30)
    return response.status_code in (200, 302, 303, 307)


def huggingface_dataset_file_upload(
    file_path: str,
    dataset_repository: str,
    dataset_path: str,
    token: str,
):
    '''
    Upload a single file to a specific dataset config in Hugging Face Hub.
    @param dataset_path: full name of the file in the repository, including the config folder (e.g., "my_config/my_file.jsonl")
    '''
    assert os.path.exists(file_path)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=dataset_path,
        repo_id=dataset_repository,
        repo_type='dataset',
        token=token,
    )


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



def huggingface_dataset_file_download(
    folder_datasets: str,
    dataset_repository: str,
    file_path: str,
    token: Optional[str],
    folder_cache: str = '/tmp/huggingface_cache',
) -> None:
    '''
    Download a single file from a dataset in Hugging Face Hub.
    
    Args:
        folder_datasets: Local directory where datasets are stored.
        dataset_repository: Hugging Face dataset repository ID
        file_path: Full path of the file within the repository (e.g., "config/data.jsonl")
        token: Hugging Face authentication token
        folder_cache: Cache directory for downloads
    
    The file will be saved at os.path.join(folder_datasets, file_path)
    '''
    os.makedirs(folder_datasets, exist_ok=True)
    os.makedirs(folder_cache, exist_ok=True)
    
    # Download to cache
    cached_path = hf_hub_download(
        repo_id=dataset_repository,
        filename=file_path,
        repo_type="dataset",
        token=token,
        cache_dir=folder_cache,
    )

    # Copy from cache to final folder
    target_path = os.path.join(folder_datasets, file_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(cached_path, target_path)


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


def huggingface_get_model_images(model_id, prefix: str = '') -> List[ImageFile.ImageFile]:
    '''
    Searches in anything starting with `prefix`
    '''
    images: List[ImageFile.ImageFile] = []
    api = hf_api.HfApi()
    model_info = api.model_info(model_id)
    if model_info.siblings:
        for sibling in model_info.siblings:
            if sibling.rfilename.endswith(('.png', '.jpg', '.jpeg', '.gif')) and sibling.rfilename.startswith(prefix):
                logger.info(f"Image: {sibling.rfilename}")
                response = requests.get(f"https://huggingface.co/{model_id}/resolve/main/{sibling.rfilename}", timeout=60)
                images.append(Image.open(BytesIO(response.content)))
    else:
        logger.info(f"No files found in the repository {model_id}")
    return images


def _huggingface_download_one_file(
    entry: dict,  # type: ignore[type-arg]
    folder_dataset: str,
    dataset_repository: str,
    headers: dict,  # type: ignore[type-arg]
) -> bool:
    """Download a single file from HF via HTTP.  Returns True on success."""
    entry_path = entry.get("path", "")
    filename = os.path.basename(entry_path)
    local_path = os.path.join(folder_dataset, filename)
    if os.path.exists(local_path):
        return True  # already present — treated as success

    dl_url = (
        f"https://huggingface.co/datasets/{dataset_repository}"
        f"/resolve/main/{entry_path}"
    )
    for attempt in range(3):
        try:
            resp = requests.get(
                dl_url,
                headers=headers,
                allow_redirects=True,
                timeout=60,
            )
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                fh.write(resp.content)
            return True
        except Exception as exc:
            logger.warning(
                "Attempt %d/3 failed for %s: %s", attempt + 1, filename, exc
            )
    return False


def huggingface_dataset_download_parallel(
    folder_datasets: str,
    dataset_repository: str,
    dataset_config: str,
    token: str,
    clean: bool = False,
    folder_cache: str = "/tmp/huggingface_cache",
    hf_prefix: str = "datasets",
    max_workers: int = 12,
) -> None:
    """Download a dataset config folder from HF using parallel HTTP requests.

    Faster alternative to huggingface_dataset_download() for large folders.
    Uses ThreadPoolExecutor(max_workers) for concurrent file downloads; reduces
    per-entity download time from ~6 minutes (sequential snapshot_download) to
    ~35 s at max_workers=12 (measured on HF for 801 PNG files, 2026-05-20).

    Args:
        folder_datasets: Local parent directory (e.g. "assets/datasets").
        dataset_repository: HF dataset repo ID.
        dataset_config: Folder name within folder_datasets AND within hf_prefix
                        on HF (e.g. "generated_people_George W Bush_uce_000").
        token: HF auth token.
        clean: If True, delete local folder before downloading.
        folder_cache: Unused — kept for signature compatibility with huggingface_dataset_download().
        hf_prefix: Prefix path within the HF repo (default "datasets").
        max_workers: Thread pool size for concurrent HTTP downloads.
                     Benchmark (2026-05-20, 801 files): 1=349s, 4=91s, 8=48s, 12=35s.
                     12 is the recommended default; do not exceed 16 (HF rate limits).
    """
    folder_dataset = os.path.join(folder_datasets, dataset_config)
    if clean and os.path.exists(folder_dataset):
        shutil.rmtree(folder_dataset)
    if os.path.exists(folder_dataset) and len(os.listdir(folder_dataset)) > 0:
        logger.info('Dataset already exists locally, skipping download: %s', folder_dataset)
        return
    os.makedirs(folder_dataset, exist_ok=True)

    hf_path = f"{hf_prefix}/{dataset_config}" if hf_prefix else dataset_config
    headers = {"Authorization": f"Bearer {token}"}

    # List files via HF tree API
    tree_url = (
        f"https://huggingface.co/api/datasets/{dataset_repository}"
        f"/tree/main/{hf_path}"
    )
    logger.info("Fetching file list: %s", tree_url)
    r = requests.get(tree_url, headers=headers, timeout=30)
    r.raise_for_status()
    entries = r.json()
    logger.info("Files in HF folder: %d", len(entries))

    file_entries: List[dict] = [  # type: ignore[type-arg]
        e
        for e in entries
        if e.get("type", "file") != "directory" and os.path.basename(e.get("path", ""))
    ]

    # Parallel download
    failed = 0
    done = 0
    total = len(file_entries)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_huggingface_download_one_file, entry, folder_dataset, dataset_repository, headers): entry
            for entry in file_entries
        }
        for future in as_completed(futures):
            success = future.result()
            if success:
                done += 1
            else:
                failed += 1
            completed = done + failed
            if completed % 100 == 0 or completed == total:
                logger.info("Download progress: %d/%d (failed: %d)", completed, total, failed)

    logger.info(
        "Download complete: %d downloaded, %d failed -> %s",
        done, failed, folder_dataset,
    )
    fail_rate = failed / max(total, 1)
    if fail_rate > 0.01:
        raise RuntimeError(
            f"huggingface_dataset_download_parallel: {failed}/{total} files failed "
            f"({fail_rate:.1%}) for {dataset_config}"
        )
    if failed > 0:
        logger.warning(
            "Tolerating %d failed file(s) for %s (below 1%% threshold)",
            failed, dataset_config,
        )
