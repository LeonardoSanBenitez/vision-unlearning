import os
from typing import Any, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from datasets import Image
from vision_unlearning.utils.logger import get_logger


logger = get_logger('utils')


# TODO: maybe instead of receiving model_name_or_path, receive the already loaded model somehow?
class ParameterAttributionMethod(BaseModel, ABC):
    @abstractmethod
    def attribute(
        self,
        noise_scheduler: Any,  # DDPMScheduler
        text_encoder: Any,  # CLIPTextModel
        vae: Any,  # AutoencoderKL
        unet: Any,  # UNet2DConditionModel
        dataloader: DataLoader,
        device: str,
        weight_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        pass


class ParameterAttributionMethodSaliency(ParameterAttributionMethod):
    def attribute(
        self,
        noise_scheduler: Any,  # DDPMScheduler
        text_encoder: Any,  # CLIPTextModel
        vae: Any,  # AutoencoderKL
        unet: Any,  # UNet2DConditionModel
        dataloader: DataLoader,
        device: str,
        weight_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        '''
        @return saliency: keys like "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight", values are tensors of same shape as the parameter, containing the accumulated saliency values.
        Tensor are of type torch.float32.

        Expected characteristics of the dataloader:
        * Batch size and number of workers are set according to the training arguments.
        * shuffled
        * collate behavior: Batches are created by stacking the per-example tensors (pixel_values stacked into contiguous FloatTensor)
        * Fields
            * pixel_values: preprocessed images, ready to be fed to the vae (i.e. resized, cropped, normalized...). Shape=[batch size, 3, resolution, resolution]
            * input_ids: tokenized captions, ready to be fed to the text encoder. Shape=[batch size, sequence length]

        Expected characteristics of the model (scheduler, text encoder, vae, unet):
        * unet should have anabled gradients
        * All loaded in the same device (the one specified in the arguments)
        * All loaded in the same dtype (the one specified in the arguments)
        * Loadede from the same base model / working together coherently
        '''
        logger.debug("Initializing saliency storage...")
        saliency = {name: torch.zeros_like(param, device=device)
                    for name, param in unet.named_parameters()}

        logger.debug("Starting saliency loop over batches...")
        for i, batch in enumerate(dataloader):
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device=device), return_dict=False)[0]

            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device=device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            # Backward + accumulate
            loss = F.mse_loss(model_pred, noise)
            unet.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        saliency[name] += param.grad.abs()

            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i+1}/{len(dataloader)} batches.")

        logger.debug("Finished accumulating saliency.")

        return {name: tensor.clone().detach() for name, tensor in saliency.items()}
