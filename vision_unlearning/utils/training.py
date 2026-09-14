import random
from typing import Optional, Tuple
import numpy as np
import torch
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator

from vision_unlearning.utils.logger import get_logger


logger = get_logger('training')


def compute_crop_top_left(original_size: Tuple[int, int], resolution: int, center_crop: bool) -> Tuple[int, int]:
    '''
    The (top, left) offset, in the resized image's coordinates, of the crop that
    `Resize(resolution)` followed by `CenterCrop(resolution)` or `RandomCrop(resolution)` takes out
    of an image whose original size is `original_size` = (height, width).

    Stable Diffusion XL is conditioned on this offset, so that it can attribute the framing of a
    cropped training image to the cropping rather than to the concept being trained. Stable
    Diffusion 1.x has no such conditioning and ignores the value.

    `Resize` with a single int scales the shorter edge to `resolution` and keeps the aspect ratio,
    so a square image comes out exactly resolution x resolution and only one crop is possible,
    (0, 0). For a non-square image the center crop offset is computed exactly here; a random crop's
    offset is drawn inside the transform from the global generator, and recovering it would mean
    drawing from that generator a second time and changing the training stream, so it is reported
    as (0, 0) with a warning instead.
    '''
    height, width = original_size
    if height <= 0 or width <= 0:
        raise ValueError(f"original_size must be a positive (height, width), got {original_size}")

    # Mirrors torchvision.transforms.functional._compute_resized_output_size for an int size
    short, long = (height, width) if height <= width else (width, height)
    new_short, new_long = resolution, int(resolution * long / short)
    resized_height, resized_width = (new_short, new_long) if height <= width else (new_long, new_short)

    if (resized_height, resized_width) == (resolution, resolution):
        return (0, 0)
    if center_crop:
        return (int(round((resized_height - resolution) / 2.0)), int(round((resized_width - resolution) / 2.0)))
    logger.warning(
        f"Random crop of a non-square image (original size {original_size}, resized to "
        f"{(resized_height, resized_width)}): the crop offset is not recoverable, reporting (0, 0). "
        f"This only affects the Stable Diffusion XL micro-conditioning."
    )
    return (0, 0)


def tokenize_captions(examples, tokenizer, caption_column, is_train=True):
    '''
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def unwrap_model(model, accelerator):
    '''
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def forget_tokens(examples, tokenizer, caption_column, forget_prompt: str):
    length = len(examples[caption_column])
    captions = [forget_prompt] * length
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples, tokenizer, caption_column, image_column, train_transforms, overwrite_column: Optional[str] = None, concept_overwrite: Optional[str] = None):
    '''
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

    concept_overwrite: concept to be used for overwriting, described as an textual string (used to modify the prompt).

    TODO: this handling of concept_overwrite is weird... I wish this were somewhat more structured/organized/clear.
    For example, the overwriting string may need a more complex prompt than just "an image of f{concept_overwrite}", or with a different article
    '''
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column)
    if overwrite_column is not None:
        # get tokens from caption_overwrite_column
        examples["forget_ids"] = tokenize_captions(examples, tokenizer, overwrite_column)
    elif concept_overwrite is not None:
        # get tokens from hardcoded example with class
        examples["forget_ids"] = forget_tokens(examples, tokenizer, caption_column, f"An image of {concept_overwrite}")

    return examples


def collate_fn(examples):
    '''
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    result = {"pixel_values": pixel_values, "input_ids": input_ids}
    if "forget_ids" in examples[0]:
        # This happens when `preprocess_train` was called with a non-none `concept_overwrite`
        result["forget_ids"] = torch.stack([example["forget_ids"] for example in examples])
    return result


def launch_accelerated_training(unlearner: 'Unlearner'):  # type: ignore
    '''
    Wrap your training function with the accelerator
    '''
    accelerator = Accelerator(mixed_precision="fp16", dynamo_backend="no")
    with accelerator.local_main_process_first():
        if accelerator.is_local_main_process:
            unlearner.train()

    accelerator.wait_for_everyone()  # Wait for all processes to finish
