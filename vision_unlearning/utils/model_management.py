import os
import json
from typing import List, Dict
from PIL import Image

from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub.repocard_data import EvalResult, ModelCardData

from libs.utils.logger import get_logger


logger = get_logger('utils')

def save_model_card(
    repo_id: str,
    images: Dict[str, Image.Image] = {},
    base_model: str = None,
    dataset_forget_name: str = None,
    dataset_retain_name: str = None,
    repo_folder: str = None,
    eval_results: List[EvalResult] = [],  # whenever possible, should have this names: https://huggingface.co/metrics
    tags: List[str] = [],
    hyperparameters: dict = {},
    similarities_gr: List[float] = [],
    similarities_gf: List[float] = [],
):
    '''
    the resulting file looks like this: https://github.com/huggingface/hub-docs/blob/main/modelcard.md

    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    os.makedirs(os.path.join(repo_folder, "images"), exist_ok=True)

    img_str = ""
    for name, image in images.items():
        path_relative = os.path.join("images", f"{name}.png")
        image.save(os.path.join(repo_folder, path_relative))
        img_str += f"![img]({path_relative})\n"

    # TODO: this description is not appearing in the model card
    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned for forgetting {dataset_forget_name} dataset, while retaining {dataset_retain_name}. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    model_card = populate_model_card(model_card, tags=tags)

    model_card.data = ModelCardData(
        model_name = repo_id,
        eval_results=eval_results,
        hyperparameters=hyperparameters,  # goes as kwargs
    )

    model_card.save(os.path.join(repo_folder, "README.md"))
    logger.info(f"Model card saved to {repo_folder}")

    if len(similarities_gf)>0 or len(similarities_gr)>0:
        logger.info('Saving gradient conflicts')
        with open(os.path.join(repo_folder, "gradient_conflicts.json"), "w") as f:
            json.dump({"forget": similarities_gf, "retain": similarities_gr}, f)
