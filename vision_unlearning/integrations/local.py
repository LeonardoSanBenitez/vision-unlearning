import os
from typing import Dict, List, Optional
from PIL import Image, ImageFile
import yaml
from pathlib import Path


def local_get_model_metrics(file_path) -> Dict[str, float | int | bool]:
    name_to_value = {}
    content = Path(file_path).read_text()
    yaml_content = content.split('---')[1].strip()
    data = yaml.safe_load(yaml_content)
    metrics = data['model-index'][0]['results'][0]['metrics']
    for metric in metrics:
        name_to_value[metric['name']] = metric['value']
    return name_to_value


def local_get_model_images(folder_path, prefix: str = '') -> List[ImageFile]:
    '''
    Searches only in folder `prefix`
    TODO: make it more flexible
    '''
    if not Path(folder_path):
        raise RuntimeError(f"No directory found for model: {folder_path}")

    images: List[ImageFile] = []
    for file_path in Path(os.path.join(folder_path, prefix)).iterdir():
        if file_path.is_file() and file_path.suffix in {'.png', '.jpg', '.jpeg', '.gif'}:
            images.append(Image.open(file_path))

    return images
