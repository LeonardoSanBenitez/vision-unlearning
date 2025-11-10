import os
from typing import List, Dict, Optional, Union
import torch
from diffusers import AutoPipelineForText2Image
from vision_unlearning.datasets.others import jsonl_dump
from vision_unlearning.unlearner.lora import unlearn_lora


def generate_dataset(
    model_base_name: str,
    lora_name: Optional[str],
    prompts: List[str],
    output_path: str,
    filenames: Optional[List[str]] = None,
    batch_size: int = 4,
    device: Union[int, str, torch.device] = 'cuda',
    lora_requires_inversion: bool = False,
) -> List[Dict[str, str]]:
    '''
    @param filenames: you must pass extension; Only png accepted (TODO: make configurable?)
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Load model
    if lora_name:
        _, _, pipeline = unlearn_lora(
            model_base_name,
            lora_name,
            device=str(device),
            weight_name="pytorch_lora_weights.safetensors",
            requires_inversion=lora_requires_inversion,
            return_original=False,
            return_learned=False,
        )
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_base_name,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

    # Filenames validation if provided
    if filenames is not None:
        assert len(filenames) == len(prompts), "filenames must have the same length as prompts"
        assert all(isinstance(fn, str) for fn in filenames), "all filenames must be strings"
        assert all(fn.lower().endswith('.png') for fn in filenames), "all filenames must end with .png"

    # Save metadata incrementally
    metadata: List[Dict[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_outputs = pipeline(batch_prompts).images  # type: ignore

        for i, image in enumerate(batch_outputs):
            idx = start + i
            image_name = filenames[idx] if filenames is not None else f"{idx}.png"
            image_prompt = prompts[idx]

            # Save image immediately
            image.save(os.path.join(output_path, image_name), "PNG")

            metadata.append({"file_name": image_name, "text": image_prompt})

    # Save metadata at the end
    # TODO: use the existing function to generate metadata.jsonl
    jsonl_dump(metadata, os.path.join(output_path, "metadata.jsonl"))

    return metadata
