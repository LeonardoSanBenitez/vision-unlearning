import os
import time
from typing import List, Dict, Tuple
from pydantic import BaseModel, ConfigDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from huggingface_hub.repocard_data import EvalResult

from libs.metrics import MetricImageTextSimilarity, MetricPaintingStyle
from libs.utils.logger import get_logger
if is_wandb_available():
    from libs.integrations.wandb import wandb_log_image
from libs.integrations.tensorboard import tensorboard_log_image


logger = get_logger('evaluation')


class EvaluatorTextToImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pipeline_original: StableDiffusionPipeline
    pipeline_learned: StableDiffusionPipeline
    pipeline_unlearned: StableDiffusionPipeline
    prompts_forget: List[str]
    prompts_retain: List[str]
    metric_clip: MetricImageTextSimilarity
    compute_runtimes: bool = True

    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        eval_results = []
        images = {}

        metric_common_attributes = {
            "dataset_type": "inline-prompts",
            "task_type": "text-to-image",
        }

        for scope, prompts in {'forget': self.prompts_forget, 'retain': self.prompts_retain}.items():
            metric_common_attributes["dataset_name"] = scope.capitalize() + " set"
            scores_original: List[float] = []
            scores_learned: List[float] = []
            scores_unlearned: List[float] = []
            scores_difference_learned_unlearned: List[float] = []
            scores_difference_original_unlearned: List[float] = []
            latencies: List[float] = []

            for prompt in prompts:
                t0 = time.time()
                image_original = self.pipeline_original(prompt).images[0]  # type: ignore
                image_learned = self.pipeline_learned(prompt).images[0]  # type: ignore
                image_unlearned = self.pipeline_unlearned(prompt).images[0]  # type: ignore
                latencies.append((time.time() - t0) / 3)

                score_original = self.metric_clip.score(image_original, prompt)['clip']
                score_learned = self.metric_clip.score(image_learned, prompt)['clip']
                score_unlearned = self.metric_clip.score(image_unlearned, prompt)['clip']
                scores_original.append(score_original)
                scores_learned.append(score_learned)
                scores_unlearned.append(score_unlearned)
                scores_difference_learned_unlearned.append(score_learned - score_unlearned)
                scores_difference_original_unlearned.append(score_original - score_unlearned)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_original)
                axes[0].set_title(f"Original\nClip Score={score_original:.2f}")
                axes[0].axis("off")
                axes[1].imshow(image_learned)
                axes[1].set_title(f"Learned\nClip Score={score_learned:.2f}")
                axes[1].axis("off")
                axes[2].imshow(image_unlearned)
                axes[2].set_title(f"Unlearned\nClip Score={score_unlearned:.2f}")
                axes[2].axis("off")
                fig.suptitle(prompt, fontsize=16)
                fig.canvas.draw()
                images[prompt] = Image.fromarray(np.uint8(np.array(fig.canvas.buffer_rgba())))  # type: ignore
                plt.show()

            # Assemble metrics object
            # EvalResult: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repocard_data.py#L13
            # card_data_class: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repocard_data.py#L248
            # Some info about the fields:
            #   - task_type: str, https://hf.co/tasks
            #   - dataset_type: str, hub ID, as searchable in https://hf.co/datasets, or at least satisfying the pattern `/^(?:[\w-]+\/)?[\w-.]+$/`
            #   - dataset_name: str, pretty name
            #   - metric_type: str, whenever possible should have these names: https://hf.co/metrics
            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of original model mean (~↑)',
                metric_value=float(np.mean(scores_original)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of original model std (~↓)',
                metric_value=float(np.std(scores_original)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of learned model mean ({"~↑" if scope == "forget" else "~↓"})',
                metric_value=float(np.mean(scores_learned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of learned model std (~↓)',
                metric_value=float(np.std(scores_learned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of unlearned model mean ({"↓" if scope == "forget" else "↑"})',
                metric_value=float(np.mean(scores_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score of unlearned model std (~↓)',
                metric_value=float(np.std(scores_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between learned and unlearned mean ({"↑" if scope == "forget" else "↓"})',
                metric_value=float(np.mean(scores_difference_learned_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between learned and unlearned std (~↓)',
                metric_value=float(np.std(scores_difference_learned_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between original and unlearned mean ({"↑" if scope == "forget" else "↓"})',
                metric_value=float(np.mean(scores_difference_original_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

            eval_results.append(EvalResult(
                metric_type='clip',
                metric_name=f'{scope.capitalize()}Set clip score difference between original and unlearned std (~↓)',
                metric_value=float(np.std(scores_difference_original_unlearned)),
                **metric_common_attributes,  # type: ignore
            ))

        if self.compute_runtimes:
            metric_common_attributes["dataset_name"] = "Forget and Retain sets"
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name='Inference latency seconds mean(↓)',
                metric_value=float(np.mean(latencies)),
                **metric_common_attributes,  # type: ignore
            ))
    
            eval_results.append(EvalResult(
                metric_type='runtime',
                metric_name='Inference latency seconds std(~↓)',
                metric_value=float(np.std(latencies)),
                **metric_common_attributes,  # type: ignore
            ))

        return eval_results, images


def evaluate_painting_style(metadata: List[Dict[str, str]], metric_painting_style: MetricPaintingStyle, dataset_path: str, device: str) -> dict:
    '''
    @param metadata: list of dictionaries with keys "file_name" and "text"; follows this schema: Follows this schema: https://huggingface.co/docs/datasets/v2.4.0/en/image_load#image-captioning
    @return metrics (as float, not yet as EvalResult)
    Compute metrics from already generated images
    '''
    metrics = {
        'per_image': [],
    }

    # TODO: refactor this to leverage paralelism
    for result in metadata:
        image = Image.open(os.path.join(dataset_path, result['file_name']))
        result.update(metric_painting_style.score(image))
        metrics['per_image'].append(result)

    metrics['overall'] = {
        'is_desired_style_mean': float(np.mean([result['is_desired_style'] for result in metrics['per_image']])),
        'desired_style_confidence_mean': float(np.mean([result['desired_style_confidence'] for result in metrics['per_image']])),
        'desired_style_confidence_std': float(np.std([result['desired_style_confidence'] for result in metrics['per_image']])),
    }

    return metrics


def log_validation(
    pipeline,
    accelerator,
    epoch,
    num_validation_images,
    validation_prompt,
    seed,
    is_final_validation=False,
) -> Dict[str, Image.Image]:
    '''
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    images: Dict[str, Image.Image] = {}
    logger.info(
        f"Running validation... \n Generating {num_validation_images} images with prompt:"
        f" {validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for i in range(num_validation_images):
            images[f"val_prompt_{i+1:02d}"] = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            tensorboard_log_image(tracker, phase_name, validation_prompt, epoch, images)
        if tracker.name == "wandb":
            wandb_log_image(tracker, phase_name, validation_prompt, epoch, images)

    return images


def plot_gradient_conflict_hist(similarities: List[float], title: str, color: str) -> Image.Image:
    fig = plt.figure(figsize=(8, 5))
    plt.hist(similarities, bins=50, color=color, alpha=0.75, label="Values")
    plt.axvline(np.mean(similarities), color=color, linestyle='-.', linewidth=2, label="Avgerage")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(title)
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    fig.canvas.draw()
    return Image.fromarray(np.uint8(np.array(fig.canvas.buffer_rgba())))
