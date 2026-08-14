'''
Implementation of FADE (in its three variants: non-sparse, sparse-per-module, sparse-per-weight).
Please cite the following paper if you use this code:
@misc{kelsch2026fadeselectiveforgettingsparse,
      title={FADE: Selective Forgetting via Sparse LoRA and Self-Distillation},
      author={Carolina R. Kelsch and Leonardo S. B. Pereira and Natnael Mola and Luis H. Arribas and Juan C. S. M. Avedillo},
      year={2026},
      eprint={2602.07058},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.07058},
}

'''
import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal
from pydantic import Field
import numpy as np

import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from safetensors.torch import load_file
from datasets import load_dataset
from datasets import Image as DatasetImage
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionPipeline
import peft
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import accelerate
from accelerate import Accelerator

from vision_unlearning.unlearner import UnlearnerLora, logger
from vision_unlearning.utils.parameter_attribution import ParameterAttributionMethod
from vision_unlearning.utils.training import unwrap_model
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple


########################################
# Non-sparse version
########################################
class UnlearnerLoraDistillation(UnlearnerLora):
    '''
    SparsePEFT is not active in this trainer
    '''
    overwritting_concept: Optional[str] = Field(default=None, description="The concept to which the forgotten concept will be mapped to. Can be specified either in `overwritting_concept` (single fixed concept) or `overwrite_column`+`json_metafile` (arbitrary set of concepts); The latter have priority.")  # noqa
    overwrite_column: str = "overwrite"
    json_metafile: Optional[str] = Field(default=None, description="Path to a JSON file that contains a mapping from each training example to the concept that will overwrite the forgotten one. The overwrite concept can be specified either in `overwritting_concept` (single fixed concept) or `overwrite_column`+`json_metafile` (arbitrary set of concepts); The latter have priority. The datasets can be specified either in `dataset_forget_name`+`dataset_retain_name` (single-folder datasets) or `json_metafile` (arbitrary paths); The latter have priority.")  # noqa
    is_lora_negated: bool = Field(default=False, description="If Lora is trained to be good at the task (as suggestion by Zhang2023). If true, the trained model should be inverted using `unlearn_lora` before usage")  # noqa

    # This class accept to be configured via a json, which takes precedence over the configurations of (1) the datasets and (2) the overwrite concept
    # TODO: json configuration should be accepted by all unlearners
    # This json looks like the following:
    '''
    {
        "forget": [
            {
            "file_name": "images/Architectures-Warm_Smear-19.jpg",
            "text": "An image of Architectures in Warm Smear style.",
            "overwrite": "An image of Architectures in Photo style."
            },
            {
            "file_name": "images/Architectures-Warm_Smear-8.jpg",
            "text": "An image of Architectures in Warm Smear style.",
            "overwrite": "An image of Architectures in Photo style."
            },
            {
            "file_name": "images/Architectures-Warm_Smear-12.jpg",
            "text": "An image of Architectures in Warm Smear style.",
            "overwrite": "An image of Architectures in Photo style."
            }
        ],
        "retain": [
            {
            "file_name": "images/Architectures-Abstractionism-6.jpg",
            "text": "An image of Architectures in Abstractionism style.",
            "overwrite": ""
            },
            {
            "file_name": "images/Architectures-Abstractionism-4.jpg",
            "text": "An image of Architectures in Abstractionism style.",
            "overwrite": ""
            },
            {
            "file_name": "images/Architectures-Abstractionism-2.jpg",
            "text": "An image of Architectures in Abstractionism style.",
            "overwrite": ""
            }
        ]
    }
    '''
    def _pre_checks(self) -> None:
        super()._pre_checks()
        if not isinstance(self.gradient_weighting_method, GradientWeightingMethodSimple):
            raise ValueError(f"FADE does not support more advanced gradient weighting methods. Please set `gradient_weighting_method` to `GradientWeightingMethodSimple`.")
        if self.compute_gradient_conflict:
            raise ValueError("FADE does not support calculating gradient conflict. Please set `compute_gradient_conflict` to False.")

    def _prepare_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        '''
        Get the datasets: you can either provide your own training and evaluation files
        or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        In distributed training, the load_dataset function guarantees that only one local process can concurrently
        download the dataset.
        Downloading and loading a dataset from the hub.

        Characteristics of the returned dataloaders:
        * Batch size and number of workers are set according to the training arguments.
        * shuffled
        * collate behavior: Batches are created by stacking the per-example tensors (pixel_values stacked into contiguous FloatTensor)
        * Fields
            * pixel_values: preprocessed images, ready to be fed to the vae (i.e. resized, cropped, normalized...). Shape=[batch size, 3, resolution, resolution]
            * input_ids: tokenized captions, ready to be fed to the text encoder. Shape=[batch size, sequence length]
            * forget_ids: tokenized overwrite captions, ready to be fed to the text encoder; Returned just be forget dataloader. Shape=[batch size, sequence length]
        '''
        if (self.overwritting_concept is None) and (self.json_metafile is None or self.overwrite_column is None):
            raise ValueError("You need to specify either `overwritting_concept` (single fixed concept) or `overwrite_column`+`json_metafile` (arbitrary set of concepts). The latter have priority.")
        assert self._tokenizer is not None
        assert self._accelerator is not None

        # Load datasets
        if (self.json_metafile is None or self.overwrite_column is None):
            # Case 1: loading datasets using dataset names
            assert self.dataset_forget_name is not None
            assert self.dataset_retain_name is not None
            dataset_forget = load_dataset(
                self.dataset_forget_name,
                self.dataset_forget_config_name,
                cache_dir=self.cache_dir,
                data_dir=None,
            )
            dataset_retain = load_dataset(
                self.dataset_retain_name,
                self.dataset_retain_config_name,
                cache_dir=self.cache_dir,
                data_dir=None,
            )

        else:
            # Case 2: loading datasets using a json metafile
            dataset_forget = load_dataset(
                "json",
                data_files=self.json_metafile,
                field="forget",
                cache_dir=self.cache_dir,
            )
            dataset_retain = load_dataset(
                "json",
                data_files=self.json_metafile,
                field="retain",
                cache_dir=self.cache_dir,
            )

        # load_dataset function creates a 'train' split by default, even if the original dataset doesn't have splits.
        # We will use this 'train' split for both forgetting and retaining datasets.
        assert "train" in dataset_forget, f"Expecting a 'train' split in the dataset for forgetting, but got {dataset_forget.keys()}"
        assert "train" in dataset_retain, f"Expecting a 'train' split in the dataset for retaining, but got {dataset_retain.keys()}"

        dataset_forget_field = "train"
        dataset_retain_field = "train"

        assert type(dataset_forget) == type(dataset_retain), f"Expecting the same type for both datasets, but got {type(dataset_forget)} and {type(dataset_retain)}"  # noqa
        logger.info(f'Retain dataset: {dataset_retain}\nForget dataset: {dataset_forget}')

        column_names_forget = dataset_forget[dataset_forget_field].column_names
        column_names_retain = dataset_retain[dataset_retain_field].column_names
        assert type(column_names_forget) == type(column_names_retain) == list, f"Expecting column names to be a list, but got {type(column_names_forget)} and {type(column_names_retain)}"  # noqa
        if self.image_column not in column_names_forget:
            raise ValueError(f"image_column value '{self.image_column}' needs to be one of: {', '.join(column_names_forget)}")
        if self.caption_column not in column_names_forget:
            raise ValueError(f"caption_column' value '{self.caption_column}' needs to be one of: {', '.join(column_names_forget)}")
        if (self.json_metafile is not None and self.overwrite_column is not None) and (self.overwrite_column not in column_names_forget):
            raise ValueError(f"overwrite_column value '{self.overwrite_column}' needs to be one of: {', '.join(column_names_forget)}")
        logger.info(f"Forget Dataset config - Image column: {self.image_column}, caption column: {self.caption_column}, overwrite column: {self.overwrite_column if self.json_metafile is not None and self.overwrite_column is not None else 'N/A'}")

        if self.image_column not in column_names_retain:
            raise ValueError(f"image_column value '{self.image_column}' needs to be one of: {', '.join(column_names_retain)}")
        if self.caption_column not in column_names_retain:
            raise ValueError(f"caption_column' value '{self.caption_column}' needs to be one of: {', '.join(column_names_retain)}")
        logger.info(f"Retain Dataset config - Image column: {self.image_column}, caption column: {self.caption_column}")

        dataset_forget[dataset_forget_field] = dataset_forget[dataset_forget_field].cast_column(self.image_column, DatasetImage())
        dataset_retain[dataset_retain_field] = dataset_retain[dataset_retain_field].cast_column(self.image_column, DatasetImage())

        # Transformations
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        # ''' Ugly version (but works)
        from vision_unlearning.utils.training import tokenize_captions, preprocess_train, unwrap_model, forget_tokens, compute_crop_top_left  # noqa  # type: ignore

        def add_micro_conditioning(examples, images):  # type: ignore
            '''
            Record what Stable Diffusion XL's micro-conditioning is told about each image: its size
            before any resizing, and the offset of the crop the transforms took out of it (D6 of
            PLAN-TASK-2026-08-12-SDXL). Recorded for both base models -- Stable Diffusion simply
            never reads them -- so that a batch has one shape whichever checkpoint is loaded.
            '''
            examples["original_sizes"] = [(image.height, image.width) for image in images]
            examples["crop_top_lefts"] = [
                compute_crop_top_left((image.height, image.width), self.resolution, self.center_crop)
                for image in images
            ]
            return examples

        def tokenize_for_second_encoder(examples, column):  # type: ignore
            '''
            The same captions, tokenized with Stable Diffusion XL's second tokenizer.

            A caption column holding several captions per example is rejected rather than sampled:
            tokenize_captions draws one at random, so a second call would be free to pick a
            different caption than the first encoder saw, and the two halves of the conditioning
            would then describe two different prompts.
            '''
            if any(not isinstance(caption, str) for caption in examples[column]):
                raise NotImplementedError(
                    f"Column '{column}' holds more than one caption per example, which the Stable "
                    f"Diffusion XL path does not support: both text encoders must be given the same caption."
                )
            return tokenize_captions(examples, self._tokenizer_2, column)

        def preprocess_train(examples):  # type: ignore
            images = [image.convert("RGB") for image in examples[self.image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples, self._tokenizer, self.caption_column)
            if self._is_xl:
                examples["input_ids_2"] = tokenize_for_second_encoder(examples, self.caption_column)
            return add_micro_conditioning(examples, images)

        def preprocess_forget(examples):
            images = [image.convert("RGB") for image in examples[self.image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples, self._tokenizer, self.caption_column)
            if self._is_xl:
                examples["input_ids_2"] = tokenize_for_second_encoder(examples, self.caption_column)
            add_micro_conditioning(examples, images)
            if (self.json_metafile is None or self.overwrite_column is None):
                # Case 1: use fixed overwriting concept for all examples
                assert self.overwritting_concept is not None, "When not using json_metafile+overwrite_column, overwritting_concept cannot be None"
                examples["forget_ids"] = forget_tokens(examples, self._tokenizer, self.caption_column, f'An image of {self.overwritting_concept}')  # TODO: hardcoded tenmplate
                if self._is_xl:
                    examples["forget_ids_2"] = forget_tokens(examples, self._tokenizer_2, self.caption_column, f'An image of {self.overwritting_concept}')
            else:
                # Case 2: loading datasets using a json metafile
                examples["forget_ids"] = tokenize_captions(examples, self._tokenizer, self.overwrite_column)
                if self._is_xl:
                    examples["forget_ids_2"] = tokenize_for_second_encoder(examples, self.overwrite_column)
            return examples

        # Set the training transforms
        with self._accelerator.main_process_first():
            if self.max_train_samples is not None:
                dataset_forget[dataset_forget_field] = dataset_forget[dataset_forget_field].shuffle(seed=self.seed).select(range(self.max_train_samples))  # TODO: why only the dataset forget gets clipped
            train_dataset_forget = dataset_forget[dataset_forget_field].with_transform(preprocess_forget)  # applies a callable lazily at read time
            train_dataset_retain = dataset_retain[dataset_retain_field].with_transform(preprocess_train)

        # ''' End of ugly version

        ''' Clean version (but does not work)
        from vision_unlearning.utils.training import preprocess_train
        with self._accelerator.main_process_first():
            if self.max_train_samples is not None:
                dataset_forget[dataset_forget_field] = dataset_forget[dataset_forget_field].shuffle(seed=self.seed).select(range(self.max_train_samples))
            train_dataset_forget = dataset_forget[dataset_forget_field].with_transform(lambda examples: preprocess_train(examples, self._tokenizer, self.caption_column, self.image_column, train_transforms, self.overwritting_concept))
            train_dataset_retain = dataset_retain[dataset_retain_field].with_transform(lambda examples: preprocess_train(examples, self._tokenizer, self.caption_column, self.image_column, train_transforms))
        '''  # end of clean version

        # DataLoaders creation
        # ''' Ugly version (but works)
        def collate_micro_conditioning(batch, examples):  # type: ignore
            '''
            Carry the per-example micro-conditioning through to the batch, plus the second
            tokenization when a Stable Diffusion XL checkpoint is loaded. The sizes stay Python
            lists of pairs: they are turned into a tensor in _encode_conditioning, on the device
            and in the dtype the denoiser wants.
            '''
            batch["original_sizes"] = [example["original_sizes"] for example in examples]
            batch["crop_top_lefts"] = [example["crop_top_lefts"] for example in examples]
            for key in ("input_ids_2", "forget_ids_2"):
                if key in examples[0]:
                    batch[key] = torch.stack([example[key] for example in examples])
            return batch

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return collate_micro_conditioning({"pixel_values": pixel_values, "input_ids": input_ids}, examples)

        def collate_forget(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            forget_ids = torch.stack([example["forget_ids"] for example in examples])
            return collate_micro_conditioning({"pixel_values": pixel_values, "input_ids": input_ids, "forget_ids": forget_ids}, examples)

        train_forget_dataloader = torch.utils.data.DataLoader(
            train_dataset_forget,
            shuffle=True,
            collate_fn=collate_forget,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.dataloader_num_workers,
        )

        train_retain_dataloader = torch.utils.data.DataLoader(
            train_dataset_retain,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.dataloader_num_workers,
        )
        # ''' End of ugly version

        ''' Clean version (but does not work)
        from vision_unlearning.utils.training import collate_fn

        train_forget_dataloader = torch.utils.data.DataLoader(
            train_dataset_forget,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.dataloader_num_workers,
        )

        train_retain_dataloader = torch.utils.data.DataLoader(
            train_dataset_retain,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.dataloader_num_workers,
        )
        '''  # End of clean version

        logger.info(f"Number of training examples = {len(train_dataset_forget)} + {len(train_dataset_retain)}")
        return train_forget_dataloader, train_retain_dataloader

    def _train_one_batch(self, batch_forget, batch_retain):
        assert self._unet is not None
        assert isinstance(self.gradient_weighting_method, GradientWeightingMethodSimple)
        assert self._accelerator is not None
        assert self._vae is not None

        # Convert images to latent space
        latents_forget = self._vae.encode(batch_forget["pixel_values"].to(dtype=self._vae_dtype())).latent_dist.sample()
        latents_forget = (latents_forget * self._vae.config.scaling_factor).to(dtype=self._weight_dtype)

        latents_retain = self._vae.encode(batch_retain["pixel_values"].to(dtype=self._vae_dtype())).latent_dist.sample()
        latents_retain = (latents_retain * self._vae.config.scaling_factor).to(dtype=self._weight_dtype)

        # Sample noise that we'll add to the latents
        noise_forget = torch.randn_like(latents_forget)
        noise_retain = torch.randn_like(latents_retain)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_forget += self.noise_offset * torch.randn(
                (latents_forget.shape[0], latents_forget.shape[1], 1, 1), device=latents_forget.device
            )
            noise_retain += self.noise_offset * torch.randn(
                (latents_retain.shape[0], latents_retain.shape[1], 1, 1), device=latents_retain.device
            )

        bsz = latents_forget.shape[0]

        # Sample a random timestep for each image
        timesteps_forget = torch.randint(0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_forget.device)
        timesteps_forget = timesteps_forget.long()
        timesteps_retain = torch.randint(0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_retain.device)
        timesteps_retain = timesteps_retain.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents_forget = self._noise_scheduler.add_noise(latents_forget, noise_forget, timesteps_forget)
        noisy_latents_retain = self._noise_scheduler.add_noise(latents_retain, noise_retain, timesteps_retain)

        # Get the text embedding for conditioning
        encoder_hidden_states_forget, added_cond_kwargs_forget = self._encode_conditioning(batch_forget, "forget")
        encoder_hidden_states_retain, added_cond_kwargs_retain = self._encode_conditioning(batch_retain, "retain")
        encoder_hidden_states_overwt, added_cond_kwargs_overwt = self._encode_conditioning(batch_forget, "override")

        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self._noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        # TODO: in self distillation, I think the target isn't needed at all
        if self._noise_scheduler.config.prediction_type == "epsilon":
            target_forget = noise_forget
            target_retain = noise_retain
        elif self._noise_scheduler.config.prediction_type == "v_prediction":
            target_forget = self._noise_scheduler.get_velocity(latents_forget, noise_forget, timesteps_forget)
            target_retain = self._noise_scheduler.get_velocity(latents_retain, noise_retain, timesteps_retain)
        else:
            raise ValueError(f"Unknown prediction type {self._noise_scheduler.config.prediction_type}")

        # Predict the noise residual, and backpropagate each half before the other half is computed.
        #
        # The two halves used to be forwarded together and backpropagated from their weighted sum,
        # which holds both activation graphs in video memory at once. Splitting them is exact rather
        # than an approximation: gradients accumulate additively into .grad, and the gradient of a
        # weighted sum is the weighted sum of the gradients, so the parameter update is the same
        # while only one graph is alive at a time. It is what makes a Stable Diffusion XL step fit
        # on a 12 GB card.
        #
        # Everything the halves draw from the random number generator -- the latents, the noise and
        # the timesteps -- is still drawn above, in the order it always was, so the split changes
        # when memory is released and nothing about what is computed.
        loss_weighting = self.gradient_weighting_method  # This weights the losses, not the gradients... but in this case this works

        # Forget half
        model_pred_forget_new = self._unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states_forget, added_cond_kwargs=added_cond_kwargs_forget, return_dict=False)[0]
        self._unet.disable_adapters()
        # The adapters are disabled here, so this call has no path to any trainable parameter --
        # `torch.no_grad()` drops the activation graph it would otherwise build for a backward that
        # never uses it. Exact, not an approximation: the value of model_pred_forget_old is unaffected
        # by whether it carries a graph.
        with torch.no_grad():
            model_pred_forget_old = self._unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states_overwt, added_cond_kwargs=added_cond_kwargs_overwt, return_dict=False)[0]
        self._unet.enable_adapters()
        loss_forget = F.mse_loss(model_pred_forget_new.float(), model_pred_forget_old.float(), reduction="mean")  # This is a Tensor of shape [], aka is a float
        self._accelerator.backward(loss_weighting.forget_weight * loss_forget)
        del model_pred_forget_new, model_pred_forget_old

        # Retain half
        model_pred_retain_new = self._unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states_retain, added_cond_kwargs=added_cond_kwargs_retain, return_dict=False)[0]
        self._unet.disable_adapters()
        # Same reasoning as the forget half above: adapters disabled, no trainable parameter reachable.
        with torch.no_grad():
            model_pred_retain_old = self._unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states_retain, added_cond_kwargs=added_cond_kwargs_retain, return_dict=False)[0]
        self._unet.enable_adapters()
        loss_retain = F.mse_loss(model_pred_retain_new.float(), model_pred_retain_old.float(), reduction="mean")
        self._accelerator.backward(loss_weighting.retain_weight * loss_retain)
        del model_pred_retain_new, model_pred_retain_old

        # Peak memory measurement
        torch.cuda.synchronize()
        batch_peak = torch.cuda.max_memory_allocated()
        self._peak_mem = max(self._peak_mem, batch_peak)

        if self._accelerator.sync_gradients:
            params_to_clip = self._lora_layers
            self._accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad()

        return loss_forget, loss_retain


########################################
# Sparse-per-module version
########################################
class UnlearnerLoraDistillationSparsePerModule(UnlearnerLoraDistillation):
    sparsity_inclusiveness: float = Field(default=0.5, description="Percentage of top modules that will be finetuned; Increasing it selects more parameters; Between 0 and 1.")
    parameter_attribution_method: ParameterAttributionMethod
    attribution_overwrite_if_exists: bool = False
    _attribution_path: Optional[str] = None

    def _get_lora_config(self) -> LoraConfig:
        assert self._noise_scheduler is not None
        assert self._text_encoder is not None
        assert self._vae is not None
        assert self._unet is not None
        assert self._accelerator is not None
        assert self._train_forget_dataloader is not None

        # Get saliency
        self._attribution_path = os.path.join(self.output_dir, 'attribution.pt')
        if not self.attribution_overwrite_if_exists and os.path.exists(self._attribution_path):
            saliency = torch.load(self._attribution_path)
            logger.info(f"Loaded existing attribution from {self._attribution_path}")
        else:
            self._unet.requires_grad_(True)
            saliency = self.parameter_attribution_method.attribute(
                noise_scheduler=self._noise_scheduler,
                text_encoder=self._text_encoder,
                vae=self._vae,
                unet=self._unet,
                dataloader=self._train_forget_dataloader,
                device=self._accelerator.device,
                weight_dtype=self._weight_dtype,
            )
            self._unet.requires_grad_(False)
            assert len(saliency) > 0
            torch.save(saliency, self._attribution_path)
            logger.info(f"Saved new attribution to {self._attribution_path}")
        # for i, k in enumerate(saliency.keys()):
        #     m = saliency[k]
        #     print(k, '  |  ' ,type(m), '  |  ', m.shape, '  |  ', m.dtype, '  |  ', m.float().mean().item())

        # Threshold saliency per module
        # Get top p% layers
        # Already select only trainable parameters
        layer_means: Dict[str, float] = {}
        for k, m in saliency.items():
            if '.bias' in k:
                continue
            k = k.replace('.weight', '')
            t = type(self._unet.get_submodule(k))
            if (t == torch.nn.Linear) or (t == torch.nn.Conv2d) or (t == torch.nn.Conv1d) or (t == torch.nn.Conv3d) or (t == transformers.pytorch_utils.Conv1D) or (t == torch.nn.Embedding) or (t == torch.nn.MultiheadAttention):
                layer_means[k] = float(m.mean().item())
        assert len(layer_means) > 10
        sorted_layers = sorted(layer_means.items(), key=lambda x: x[1], reverse=True)
        sparsity_inclusiveness_count: int = max(1, int(len(sorted_layers) * self.sparsity_inclusiveness))  # At least one
        top_layers: List[str] = [name for name, _ in sorted_layers[:sparsity_inclusiveness_count]]
        assert np.abs(len(top_layers) / len(layer_means) - self.sparsity_inclusiveness) < 0.01, f"Expected a sparsity around {self.sparsity_inclusiveness}, but got {len(top_layers)/len(layer_means)} ({len(top_layers)} layers out of {len(saliency)} total layers, {len(top_layers)} of which are trainable)"  # noqa

        # TODO assert target_modules was not set by the user
        self.target_modules = top_layers
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,  # type: ignore  # TODO: should this be int or float?
            init_lora_weights="gaussian",
            target_modules=self.target_modules,
        )


########################################
# Sparse-per-weight version
########################################
def check_lora_sparsity(adapter_path: str, config_file_name: str = "adapter_config.json", model_file_name: str = "adapter_model.bin") -> float:
    # Load config
    config_path = os.path.join(adapter_path, config_file_name)
    model_path = os.path.join(adapter_path, model_file_name)
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load LoRA weights
    if model_path.endswith(".safetensors"):
        weights = load_file(model_path)
    else:
        weights = torch.load(model_path, map_location="cpu")

    # Expect keys like: 'unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default'
    lora_A_keys = [k for k in weights if ".lora_A." in k]
    total: int = 0
    zero: int = 0

    for A_key in lora_A_keys:
        B_key = A_key.replace("lora_A", "lora_B")
        if B_key not in weights:
            print(f"Missing B for A: {A_key}")
            continue

        A = weights[A_key]  # [r, in]
        B = weights[B_key]  # [out, r]
        AB = B @ A  # [out, in]

        total += int(AB.numel())
        zero += int((AB == 0).sum().item())

    sparsity = zero / total if total > 0 else 0.0
    return float(sparsity)


def is_valid_elastic_adapter_config(config):
    if not isinstance(config, list):
        return False
    for item in config:
        if not isinstance(item, dict):
            return False
        if "target" not in item or "search_space" not in item:
            return False
        if not isinstance(item["target"], list) or not isinstance(item["search_space"], list):
            return False
    return True


def is_valid_custom_adapter_config(config):
    if not isinstance(config, list):
        return False
    for item in config:
        if not isinstance(item, dict):
            return False
        if "target" not in item or "active_rank_value" not in item:
            return False
        if not isinstance(item["target"], list) or not isinstance(item["active_rank_value"], int):
            return False
    return True


def load_custom_adapter_config(custom_adapter_config_file):
    if custom_adapter_config_file is None:
        raise ValueError("Missing custom adapter config file.")

    with open(custom_adapter_config_file, "r") as f:
        custom_adapter_config = json.load(f)

    if not is_valid_custom_adapter_config(custom_adapter_config):
        raise ValueError("Invalid configuration: `custom_adapter_config`.")

    return custom_adapter_config


def get_rank_value(space, strategy):
    if strategy == "maximal":
        return space[0]
    elif strategy == "heuristic":
        return space[(len(space) - 1) // 2]
    elif strategy == "minimal":
        return space[-1]
    else:
        raise ValueError("Invalid strategy")


def extract_sub_adapter(
    adapter_model: str,  # Path to trained adapter
    elastic_adapter_config_file: str,  # The JSON file that stores the search space of the elastic adapter
    output_dir: str,  # The output directory where the sub-adapter will be saved
    adapter_version: Literal["maximal", "heuristic", "minimal"] = "heuristic",  # Version of the subnetwork to extract
):
    os.makedirs(output_dir, exist_ok=True)

    with open(elastic_adapter_config_file, "r") as f:
        elastic_adapter_config = json.load(f)
        if not is_valid_elastic_adapter_config(elastic_adapter_config):
            raise ValueError("Invalid configuration: `elastic_adapter_config`")

    sub_adapter_config = {}
    for group in elastic_adapter_config:
        modules = group["target"]
        for module in modules:
            sub_adapter_config[module] = get_rank_value(group["search_space"], adapter_version)

    # Load adapter weights
    adapter_weights_path = os.path.join(adapter_model, "pytorch_lora_weights.safetensors")
    super_adapter_weights = load_file(adapter_weights_path)

    sub_adapter_weights = {}
    for weight_key, weight_tensor in super_adapter_weights.items():
        target_key = weight_key.rsplit('.', 1)[0]
        if "lora_A" in weight_key or "lora_B" in weight_key:
            if "lora_B" in weight_key:
                target_key = target_key.replace("lora_B", "lora_A")
            if target_key in sub_adapter_config:
                active_r = sub_adapter_config[target_key]
                if "lora_A" in weight_key:
                    new_weight_tensor = weight_tensor[:active_r].clone()
                else:  # "lora_B" in weight_key
                    new_weight_tensor = weight_tensor[:, :active_r].clone()
            else:
                new_weight_tensor = weight_tensor.clone()
        else:
            new_weight_tensor = weight_tensor.clone()
        sub_adapter_weights[weight_key] = new_weight_tensor

    torch.save(sub_adapter_weights, os.path.join(output_dir, "pytorch_lora_weights.bin"))
    shutil.copy(elastic_adapter_config_file, output_dir)


class SharedR:
    def __init__(self, r_values):
        self.r_values = r_values
        self.current_r = random.choice(r_values)

    def __call__(self):
        return self.current_r

    def update_r(self):
        self.current_r = random.choice(self.r_values)


class ElasticLoRALinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, shared_r, is_lora_A, bias=False, is_group_head=False):
        super(ElasticLoRALinear, self).__init__(in_features, out_features, bias)
        self.shared_r = shared_r
        self.is_lora_A = is_lora_A
        self.is_group_head = is_group_head
        self.eval()
        self._weight = self.weight
        self._bias = self.bias

    def init_weights(self, orig: torch.nn.Linear):
        self._weight.data.copy_(orig.weight.data)
        if orig.bias is not None:
            self._bias.data.copy_(orig.bias.data)

    @torch.no_grad()
    def active_sub_adapter(self):
        if self.is_group_head:
            self.shared_r.update_r()
        return self.shared_r()

    @property
    def masked_weight(self):
        active_r: int = self.active_sub_adapter()
        if self.is_lora_A:
            return self._weight[:active_r, :]
        return self._weight[:, :active_r]

    @property
    def weight(self):
        return self.masked_weight if self.training else self._weight

    @property
    def bias(self):
        if self._bias is not None and self.is_lora_A and self.training:
            return self._bias[: self.shared_r()]
        return self._bias

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        sd = super(ElasticLoRALinear, self).state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd.pop(prefix + '_weight', None)
        sd.pop(prefix + '_bias', None)
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False):
        super(ElasticLoRALinear, self).load_state_dict(state_dict, strict=strict, assign=assign)
        self._weight = self.weight
        self._bias = self.bias


def make_lora_elastic(model, r_values, target_modules, share_rank_within_layer=True, config_save_dir=None):
    ADAPTER_NAME = "default"

    def replace_module(layer, named_modules, module_name, shared_r, is_lora_A, is_group_head=False):
        orig = named_modules[module_name]
        assert isinstance(orig, torch.nn.Linear)
        new = ElasticLoRALinear(
            orig.in_features,
            orig.out_features,
            shared_r,
            is_lora_A,
            bias=orig.bias is not None,
            is_group_head=is_group_head,
        )
        new.init_weights(orig)
        new.train()
        parent = layer
        *path, leaf = module_name.split('.')
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, new)

    elastic_config = []
    for idx, layer in enumerate(model.modules()):
        named = dict(layer.named_modules())
        shared_r = SharedR(r_values)
        is_head = True
        config_item = {"target": [], "search_space": shared_r.r_values}

        for name, module in named.items():
            if isinstance(module, torch.nn.Linear) and "lora_A" in name and any(tm in name for tm in target_modules):
                replace_module(layer, named, name, shared_r, is_lora_A=True, is_group_head=is_head)
                sd_name = f"unet.{name}".replace(f".{ADAPTER_NAME}", "")
                if share_rank_within_layer:
                    is_head = False
                    config_item["target"].append(sd_name)
                else:
                    elastic_config.append({"target": [sd_name], "search_space": shared_r.r_values})
                b_name = name.replace('lora_A', 'lora_B')
                replace_module(layer, named, b_name, shared_r, is_lora_A=False)

        if share_rank_within_layer and config_item["target"]:
            elastic_config.append(config_item)

    if config_save_dir:
        os.makedirs(config_save_dir, exist_ok=True)
        with open(os.path.join(config_save_dir, "elastic_adapter_config.json"), "w") as f:
            json.dump(elastic_config, f, indent=4)

    return model


def find_module_name_by_weights(model: nn.Module, target_module: nn.Module) -> str:
    """
    Find the module name in `model` that has the exact same parameters as `target_module`
    Same = match by memory address
    """
    target_params = list(target_module.parameters())
    if not target_params:
        raise ValueError("Target doesn't have parameters")
    target_param_ids = set(id(p) for p in target_params)
    for name, module in model.named_modules():
        module_param_ids = set(id(p) for p in module.parameters())
        if target_param_ids == module_param_ids:
            return name
    raise ValueError('Target not found in model')


def calculate_sparsity_lora(lora_path: str, filename: str, device: str = 'cuda', threshold: float = 1e-6):
    '''
    Calculate what is the percentage of zero values in the product A*B, per target module.
    Load from HuggingFace (TODO: implemenet for local files).

    @param threshold: Tolerance for "effectively zero"
    '''
    state_dict = torch.hub.load_state_dict_from_url(
        f'https://huggingface.co/{lora_path}/resolve/main/{filename}',
        map_location=device
    )

    # Collect all A (down) and B (up) matrices
    lora_down = {k.replace(".lora.down.weight", ""): v for k, v in state_dict.items() if "lora.down.weight" in k}
    lora_up = {k.replace(".lora.up.weight", ""): v for k, v in state_dict.items() if "lora.up.weight" in k}

    # Match and compute stats
    percentages = []
    for key in sorted(lora_down.keys()):
        if key in lora_up:
            A = lora_down[key]   # shape: [r, d]
            B = lora_up[key]     # shape: [d, r]
            W_delta = B @ A      # shape: [d, d]

            total_elements = W_delta.numel()
            zero_like = torch.isclose(W_delta, torch.tensor(0.0), atol=threshold)
            zero_count = zero_like.sum().item()
            zero_percent = 100 * zero_count / total_elements
            percentages.append({'module': key, 'zero_percent': zero_percent})
        else:
            logger.warning(f"No matching B matrix for {key}")

    # Overall
    # TODO: this is not weighted by number of parameters
    percentages.append({'module': 'overall', 'zero_percent': np.mean([p['zero_percent'] for p in percentages])})
    return percentages


class UnlearnerLoraDistillationSparsePerWeight(UnlearnerLoraDistillation):
    # Specific to this unlearner, general arguments
    nls: bool = Field(default=False, description="Whether to apply Neural LoRA Search (NLS) or not.")
    nls_target_modules: List[str] = Field(default=[], description="Which module will be added the elastic lora adapter.")
    search_space: List[int] = Field(default=[], description="Low-rank search space of NLS training.")
    share_rank_within_layer: bool = Field(default=True, description="Whether the adapter shares rank values within the layer.")
    quantization_aware: bool = Field(default=False, description="Enable quantization-aware SparsePEFT.")

    parameter_attribution_method: ParameterAttributionMethod
    attribution_overwrite_if_exists: bool = False
    sparsity_inclusiveness: float = Field(default=0.5, description="Percentage of top modules that will be finetuned; Increasing it selects more parameters; Between 0 and 1.")

    _output_dir_super: Optional[str] = None
    _output_dir_sub: Optional[str] = None
    _attribution_path: Optional[str] = None
    _mask_dict: Dict[str, torch.Tensor] = {}  # where the tensors are uint8 binary masks (0 or 1)

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        self._output_dir_super = os.path.join(self.output_dir, 'super')
        self._output_dir_sub = os.path.join(self.output_dir, 'sub')
        self._attribution_path = os.path.join(self.output_dir, 'attribution.pt')
        self._output_dir_checkpoints = self._output_dir_super  # TODO rename variables?
        self._output_dir_lora = self._output_dir_sub  # This is the only thing that  will be uploaded to huggingface. If we want to upload the super or the mask, we have to change the parent class
        self._lora_weight_name = 'pytorch_lora_weights.bin'  # TODO: change to save safetensors

    def _get_lora_config(self) -> LoraConfig:
        assert self._attribution_path is not None
        assert self._noise_scheduler is not None
        assert self._text_encoder is not None
        assert self._vae is not None
        assert self._unet is not None
        assert self._accelerator is not None
        assert self._train_forget_dataloader is not None

        # Get saliency
        if not self.attribution_overwrite_if_exists and os.path.exists(self._attribution_path):
            saliency = torch.load(self._attribution_path)
            logger.info(f"Loaded existing attribution from {self._attribution_path}")
        else:
            self._unet.requires_grad_(True)
            saliency = self.parameter_attribution_method.attribute(
                noise_scheduler=self._noise_scheduler,
                text_encoder=self._text_encoder,
                vae=self._vae,
                unet=self._unet,
                dataloader=self._train_forget_dataloader,
                device=self._accelerator.device,
                weight_dtype=self._weight_dtype,
            )
            self._unet.requires_grad_(False)
            assert len(saliency) > 0
            assert all([m.dtype == torch.float32 for m in self._mask_dict.values()])
            torch.save(saliency, self._attribution_path)
            logger.info(f"Saved new attribution to {self._attribution_path}")

        # Threshold saliency per parameter
        # find the value that’s the cutoff for the top-q
        # since we want the largest values, we take the (N-k+1)-th smallest
        logger.debug("Threshold saliency to get _mask_dict")
        all_vals = torch.cat([v.flatten() for v in saliency.values()])

        k = int(len(all_vals) * self.sparsity_inclusiveness)
        cutoff = torch.kthvalue(all_vals, len(all_vals) - k + 1).values.item()
        logger.debug(f"Threshold for top {self.sparsity_inclusiveness*100:.0f}% saliency is {cutoff:.6f}")

        assert len(self._mask_dict) == 0, 'Mask is not empty'
        for name, tensor in saliency.items():
            self._mask_dict[name] = (tensor >= cutoff).to(torch.uint8)

        assert len(self._mask_dict) > 0, 'Mask is empty'
        assert all([m.dtype == torch.uint8 for m in self._mask_dict.values()])

        # Define the sparsity calculation function that will be passed to the SparsePEFT library.
        def sparsity_calculation_function(
            weight: torch.nn.parameter.Parameter,
            base_layer: nn.Module,
            weight_name: str
        ) -> torch.nn.parameter.Parameter:
            """
            Returns the pre-computed binary mask for the given weight tensor.
            Tries exact lookup on mask_dict["<module_name>.<weight_name>"],
            but if not found, searches for any key containing both the module name
            and ending with the weight_name (to catch e.g. extra ".0" in between).

            Watch out, this expects the global following variables to be declared in this scope:
            * self._unet
            * self._mask_dict

            # TODO: make this a static function
            """
            assert isinstance(self._unet, UNet2DConditionModel), "Expecting `self._unet` to be already declared"
            assert isinstance(self._mask_dict, dict), f"Expecting `_mask_dict` to be a dictionary, got {type(self._mask_dict)}"
            module_name = find_module_name_by_weights(self._unet, base_layer)
            # logger.info(f"Module name: {module_name}")

            # first try exact match
            key = f"{module_name}.{weight_name}"
            if key in self._mask_dict:
                return self._mask_dict[key].to(weight.device)

            # fallback: find any self._mask_dict key that contains module_name and ends with weight_name
            candidates = [k for k in self._mask_dict if k.startswith(f"{module_name}.") and k.endswith(f".{weight_name}")]
            if not candidates:
                raise KeyError(
                    f"Mask for '{module_name}.{weight_name}' not found, "
                    f"and no close match among keys: {list(self._mask_dict.keys())[:5]}..."
                )
            if len(candidates) > 1:
                logger.warning(f"Multiple matches for module '{module_name}' and weight '{weight_name}': {candidates}. Using '{candidates[0]}'")
            chosen_key = candidates[0]
            # logger.debug(f"Using fallback mask key '{chosen_key}'")
            return self._mask_dict[chosen_key].to(weight.device)

        # Debug:
        # means = []
        # for name, module in self._unet.named_modules():
        #    for weight_name, weight in module.named_parameters(recurse=False):
        #        mask = sparsity_calculation_function(weight, module, weight_name)
        #        m = (mask == 1).cpu().numpy().mean()
        #        means.append(m)
        #        print(f'mask shape: {mask.shape}. Mean {m}')
        # p rint(sum(means)/len(means))

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.target_modules,
            sparsity_calculation_function=sparsity_calculation_function,  # type: ignore  # This parameter only exist in the SparsePEFT version of the code... this class currently doesn't check if the user is using the right lib...
        )

    def _hook_after_lora_init(self):
        '''
        Side-effect: modifies self._unet in-place
        '''
        if self.nls:
            # TODO: do I need to pass the sparsity_calculation_function here as well?
            # I believe I should
            # So since we currently don't, then I believe there is a bug in here; probably the current NLS is hardcoded for sparsity (instead of arbitrary mask)
            logger.info('Performing NLS')
            self._unet = make_lora_elastic(
                self._unet,
                self.search_space,
                target_modules=self.nls_target_modules,
                share_rank_within_layer=self.share_rank_within_layer,
                config_save_dir=self._output_dir_super,
            )

    def _get_accelerator(self):
        assert self._output_dir_super is not None
        return Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_config=accelerate.utils.ProjectConfiguration(
                project_dir=self._output_dir_super,
                logging_dir=Path(self._output_dir_super, self.logging_dir),
            ),
        )

    def _save_lora_layers(self):
        '''
        Side-effects: modifies self._unet in-place (casts to float32), saves two directories self._output_dir_super and self._output_dir_sub
        '''
        assert self._unet is not None
        assert self._accelerator is not None
        assert self._output_dir_super is not None
        assert self._output_dir_sub is not None

        self._unet = self._unet.to(torch.float32)
        unwrapped_unet = unwrap_model(self._unet, self._accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=self._output_dir_super,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        extract_sub_adapter(
            adapter_model=self._output_dir_super,
            elastic_adapter_config_file=os.path.join(self._output_dir_super, "elastic_adapter_config.json"),
            output_dir=self._output_dir_sub,
        )

        assert os.path.exists(self._output_dir_sub)

        sparsity = check_lora_sparsity(self._output_dir_sub, config_file_name="elastic_adapter_config.json", model_file_name="pytorch_lora_weights.bin")
        logger.info(f"Sparsity of LoRA adapter: {sparsity:.6f}")  # higher=more sparse
        # assert sparsity > 0.1
        # TODO: I think this is checking the sparsity of the whole model (which will be zero), instead of just the lora (which should be something between 0 and 1)
        # TODO: copy the content from {self._output_dir_sub}/* to the root of output_dir?


class UnlearnerLoraDistillationSparse(UnlearnerLoraDistillationSparsePerWeight):
    # TODO: this is here just for naming compatibility with the existing notebooks, and should be REMOVED in future
    pass
