import os
import json
from typing import List, Dict, Optional
from PIL import Image

import torch
from diffusers import AutoPipelineForText2Image


def generate_dataset(model_base_name: str, lora_name: Optional[str], prompts: List[str], output_path: str) -> List[Dict[str, str]]:
    '''
    @return metadata: list of dictionaries with keys "file_name" and "text"; follows this schema: Follows this schema: https://huggingface.co/docs/datasets/v2.4.0/en/image_load#image-captioning
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Load model
    pipeline = AutoPipelineForText2Image.from_pretrained(model_base_name, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    if lora_name:
        pipeline.load_lora_weights(lora_name, weight_name="pytorch_lora_weights.safetensors")

    # Generate
    images: List[Image] = []
    for prompt in prompts:
        images.append(pipeline(prompt).images[0])
    assert len(images) == len(prompts)

    # Save
    # TODO: refactor this to leverage paralelism
    metadata: List[Dict[str, str]] = []
    for i in range(len(prompts)):
        image_name = f"{i}.png"
        image_prompt = prompts[i]
        metadata.append({"file_name": image_name, "text": image_prompt})
        images[i].save(os.path.join(output_path, image_name), "PNG")

    with open(os.path.join(output_path, "metadata.jsonl"), "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    
    return metadata
