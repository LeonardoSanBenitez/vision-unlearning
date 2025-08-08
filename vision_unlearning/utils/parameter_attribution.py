import os
from typing import Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from datasets import load_dataset, Image
from vision_unlearning.utils.logger import get_logger



logger = get_logger('utils')


class ParameterAttributionMethod(BaseModel, ABC):
    @abstractmethod
    def attribute(self, model_name_or_path: str, dataset_name: str, device: str) -> Dict[str, torch.Tensor]:
        pass


class ParameterAttributionMethodSaliency(ParameterAttributionMethod):
    q: float  # increasing q selects more parameters; between 0 and 1

    def attribute(self, model_name_or_path: str, dataset_name: str, device: str, image_column: str = 'image', caption_column: str = 'text', batch_size: int = 1) -> Dict[str, torch.Tensor]:
        '''
        @return mask_dict: keys like "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight"
        '''
        logger.debug("Loading scheduler and models...")
        sched = DDPMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer")
        text_enc = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder").to(device)
        vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(model_name_or_path, subfolder="unet").to(device)

        unet.requires_grad_(True)
        text_enc.eval()
        vae.eval()
        logger.debug("Models loaded")
    
        ##################
        # Prepare dataset 
        logger.debug("Loading dataset and casting image column...")
        ds = load_dataset(dataset_name, split="train")  # TODO: add support for other splits
        ds = ds.cast_column(image_column, Image())

        # define your image transforms pipeline
        pipe = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # map to tensors
        def preprocess(batch):
            batch["pixel_values"] = [pipe(img) for img in batch[image_column]]
            return batch

        logger.debug("Applying transforms to dataset...")
        ds = ds.map(preprocess, batched=True, remove_columns=[image_column])
        ds.set_format(type="torch", columns=["pixel_values", caption_column])
        logger.debug(f"Dataset ready: {len(ds)} examples.")

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)
        logger.debug("DataLoader created.")

        ##################
        # Accumulate saliency
        logger.debug("Initializing saliency storage...")
        saliency = {name: torch.zeros_like(param, device=device)
                    for name, param in unet.named_parameters()}

        logger.debug("Starting saliency loop over batches...")
        for i, batch in enumerate(loader):
            # Text → CLIP embeddings
            toks = tokenizer(batch[caption_column], padding=True, return_tensors="pt").to(device)
            txt_emb = text_enc(**toks).last_hidden_state

            # Image → latents
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            t = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy = sched.add_noise(latents, noise, t)

            # UNet prediction & loss
            pred = unet(noisy, t, encoder_hidden_states=txt_emb).sample
            loss = F.mse_loss(pred, noise)

            # Backward + accumulate
            unet.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        saliency[name] += param.grad.abs()

            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i+1}/{len(loader)} batches.")

        logger.debug("Finished accumulating saliency.")

        ##################
        # Flatten & threshold → mask
        logger.debug("Flattening saliency values to compute threshold...")
        all_vals = torch.cat([v.flatten() for v in saliency.values()])

        k = int(len(all_vals) * self.q)
        # find the value that’s the cutoff for the top-q
        # since we want the largest values, we take the (N-k+1)-th smallest
        cutoff = torch.kthvalue(all_vals, len(all_vals) - k + 1).values.item()
        logger.debug(f"Threshold for top {self.q*100:.0f}% saliency is {cutoff:.6f}")

        logger.debug("Building binary mask dict...")
        mask_dict = {}
        for name, tensor in saliency.items():
            mask_dict[name] = (tensor >= cutoff).to(torch.uint8)

        return mask_dict
