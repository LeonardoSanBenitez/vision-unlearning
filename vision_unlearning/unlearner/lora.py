import math
import os
import shutil
import time
from pathlib import Path
from packaging import version
from typing import List, Optional, Tuple, Dict
from pydantic import Field
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import datasets
from datasets import load_dataset
from huggingface_hub.repocard_data import EvalResult
from huggingface_hub import create_repo, upload_folder
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm

from libs.metrics import MetricImageTextSimilarity
from libs.evaluator import EvaluatorTextToImage, plot_gradient_conflict_hist, log_validation
from libs.utils.logger import get_logger
from libs.utils.model_management import save_model_card
from libs.utils.training import unwrap_model, preprocess_train, collate_fn
from libs.utils.gradient_weighting import GradientWeightingMethod
from libs.unlearner.base import Unlearner


logger = get_logger('trainer')

DATASET_NAME_MAPPING = {  # TODO: is this necessary?
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "Hamdy20002/COCO_Person": ("image", "text"),
}

def unlearn_lora(model_original_id: str, model_lora_id: str, device: str) -> Tuple[StableDiffusionPipeline, StableDiffusionPipeline, StableDiffusionPipeline]:
    '''
    id can be both a local dir or a huggingface model id
    return pipeline_original, pipeline_learned, pipeline_unlearned

    Inspired by @inproceedings{zhang2023composing,
        title={Composing Parameter-Efficient Modules with Arithmetic Operations}, 
        author={Zhang, Jinghan and Chen, Shiqi and Liu, Junteng and He, Junxian},
        booktitle={Advances in Neural Information Processing Systems},
        year={2023}
    }
    Source: https://github.com/hkust-nlp/PEM_composition/tree/main/exps/composition_for_unlearning
    '''
    pipeline_original = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)

    pipeline_learned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline_learned.load_lora_weights(model_lora_id, weight_name="pytorch_lora_weights.safetensors")

    pipeline_unlearned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline_unlearned.load_lora_weights(model_lora_id, weight_name="pytorch_lora_weights.safetensors")
    total: int = 0
    sum_before_invert: float = sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
    for name, param in pipeline_unlearned.unet.named_parameters():
        if "lora_A" in name:
            logger.debug(f"Inverting param {name}")
            param.data = -1 * param.data
            total += 1
    assert sum_before_invert == -sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
    assert total > 0
    logger.debug(f"Inverted {total} params")

    return pipeline_original, pipeline_learned, pipeline_unlearned


class UnlearnerLora(Unlearner):
    """
    Fine-tuning script for Stable Diffusion for text2image with support for LoRA.
    Strongly based on the huggingface example (see credits in the end)
    
    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    """
    pretrained_model_name_or_path: str = Field(..., description="Path to pretrained model or model identifier from huggingface.co/models.")
    revision: Optional[str] = Field(None, description="Revision of pretrained model identifier from huggingface.co/models.")
    variant: Optional[str] = Field(None, description="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g., fp16.")
    
    gradient_weighting_method: GradientWeightingMethod = Field(..., description="The method to use for weighting the gradients.")
    compute_gradient_conflict: bool = Field(True, description="Whether to compute the gradient conflict, for evaluation purposes.")
    compute_runtimes: bool = Field(True, description="Whether to compute the runtimes of the training, for evaluation purposes.")

    dataset_forget_name: str = Field(..., description="The name or path of the dataset to be forgotten.")
    dataset_retain_name: str = Field(..., description="The name or path of the dataset to be retained.")
    dataset_forget_config_name: Optional[str] = Field(None, description="The config of the dataset for forgetting, leave as None if there's only one config.")
    dataset_retain_config_name: Optional[str] = Field(None, description="The config of the dataset for retaining, leave as None if there's only one config.")

    image_column: str = Field("image", description="The column of the dataset containing an image.")
    caption_column: str = Field("text", description="The column of the dataset containing a caption or a list of captions.")

    validation_prompt: Optional[str] = Field(None, description="A prompt that is sampled during training for inference.")
    num_validation_images: int = Field(4, description="Number of images to generate during validation with `validation_prompt`.")
    validation_epochs: int = Field(1, description="Run fine-tuning validation every X epochs.")

    max_train_samples: Optional[int] = Field(None, description="Limit the number of training examples for debugging or quicker training.")
    output_dir: str = Field("sd-model-finetuned-lora", description="Output directory for model predictions and checkpoints.")
    cache_dir: Optional[str] = Field(None, description="Directory where downloaded models and datasets will be stored.")
    
    seed: Optional[int] = Field(None, description="A seed for reproducible training.")
    resolution: int = Field(512, description="Resolution for input images.")
    center_crop: bool = Field(False, description="Whether to center crop the input images.")
    random_flip: bool = Field(False, description="Whether to randomly flip images horizontally.")

    train_batch_size: int = Field(16, description="Batch size per device for training.")
    num_train_epochs: int = Field(100, description="Number of training epochs.")
    max_train_steps: Optional[int] = Field(None, description="Total number of training steps, overrides num_train_epochs if provided.")

    gradient_accumulation_steps: int = Field(1, description="Number of steps to accumulate before performing backward/update pass.")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory at the expense of slower backward pass.")

    learning_rate: float = Field(1e-4, description="Initial learning rate after warmup period.")
    lr_scheduler: str = Field("constant", description="Scheduler type for learning rate.")
    lr_warmup_steps: int = Field(500, description="Number of warmup steps in the learning rate scheduler.")

    use_8bit_adam: bool = Field(False, description="Use 8-bit Adam optimizer from bitsandbytes.")
    allow_tf32: bool = Field(False, description="Allow TF32 on Ampere GPUs for potential training speed-up.")

    dataloader_num_workers: int = Field(0, description="Number of subprocesses for data loading.")
    adam_beta1: float = Field(0.9, description="Beta1 parameter for Adam optimizer.")
    adam_beta2: float = Field(0.999, description="Beta2 parameter for Adam optimizer.")
    adam_weight_decay: float = Field(1e-2, description="Weight decay for Adam optimizer.")
    adam_epsilon: float = Field(1e-8, description="Epsilon value for Adam optimizer.")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm.")

    push_to_hub: bool = Field(False, description="Push the model to Hugging Face Hub.")
    hub_token: Optional[str] = Field(None, description="Token for authentication to push to Model Hub.")
    hub_model_id: Optional[str] = Field(None, description="Repository name to sync with `output_dir`.")

    logging_dir: str = Field("logs", description="Directory for TensorBoard logs.")
    mixed_precision: Optional[str] = Field(None, description="Use mixed precision training: 'fp16' or 'bf16'.")
    report_to: str = Field("tensorboard", description="Logging integration for reporting results (e.g., tensorboard, wandb).")

    local_rank: int = Field(-1, description="Local rank for distributed training.")
    checkpointing_steps: int = Field(500, description="Save training state checkpoint every X updates.")
    checkpoints_total_limit: Optional[int] = Field(None, description="Maximum number of checkpoints to store.")
    resume_from_checkpoint: Optional[str] = Field(None, description="Resume training from a previous checkpoint.")

    enable_xformers_memory_efficient_attention: bool = Field(False, description="Use xformers for memory-efficient attention.")
    noise_offset: float = Field(0.0, description="Scale of noise offset.")
    rank: int = Field(4, description="Dimension of the LoRA update matrices.")

    final_eval_prompts_forget: str|List[str] = Field([], description="Prompts for final evaluation on the forget dataset (ModelHub identifier or directly the prompts).")
    final_eval_prompts_retain: str|List[str] = Field([], description="Prompts for final evaluation on the retain dataset (ModelHub identifier or directly the prompts).")

    prediction_type: Optional[str] = Field(None, description="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.")

    def train(self):
        t0 = time.time()
        if self.report_to == "wandb" and self.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        # Acelerator config
        accelerator_project_config = accelerate.utils.ProjectConfiguration(project_dir=self.output_dir, logging_dir=Path(self.output_dir, self.logging_dir))

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_config=accelerator_project_config,
        )

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.seed is not None:
            accelerate.utils.set_seed(self.seed)

        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        
        logger.info(accelerator.state)

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

            if self.push_to_hub:
                repo_id = create_repo(repo_id=self.hub_model_id or Path(self.output_dir).name, exist_ok=True, token=self.hub_token).repo_id

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.revision)
        text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.revision)
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision, variant=self.variant)
        unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet", revision=self.revision, variant=self.variant)

        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        unet_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        if self.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

        if self.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-s
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


        # Initialize the optimizer
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            lora_layers,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        t1 = time.time()

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        # Downloading and loading a dataset from the hub.
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

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset_forget["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(self.dataset_forget_name, None)
        if self.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = self.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{self.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = self.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{self.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Set the training transforms
        with accelerator.main_process_first():
            if self.max_train_samples is not None:
                dataset_forget["train"] = dataset_forget["train"].shuffle(seed=self.seed).select(range(self.max_train_samples))
            train_dataset_forget = dataset_forget["train"].with_transform(lambda examples: preprocess_train(examples, tokenizer, caption_column, image_column, train_transforms))
            train_dataset_retain = dataset_retain["train"].with_transform(lambda examples: preprocess_train(examples, tokenizer, caption_column, image_column, train_transforms))

        # DataLoaders creation:
        train_forget_dataloader = torch.utils.data.DataLoader(
            train_dataset_forget,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
        )
        train_retain_dataloader = torch.utils.data.DataLoader(
            train_dataset_retain,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        num_warmup_steps_for_scheduler = self.lr_warmup_steps * accelerator.num_processes
        if self.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(train_forget_dataloader) / accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                self.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.max_train_steps * accelerator.num_processes

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_forget_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_forget_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_forget_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != self.max_train_steps * accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_forget_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-fine-tune", config={k: v for k, v in self.model_dump().items() if isinstance(v, (str, float, int, type(None)))})

        # Train!
        t2 = time.time()
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset_forget)} + {len(train_dataset_retain)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint != "latest":
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        similarities_gr: List[float] = []  # Cosine similarlities between \tilde g and g_r, one element per step update
        similarities_gf: List[float] = []  # Cosine similarlities between \tilde g and g_f, one element per step update

        for epoch in range(first_epoch, self.num_train_epochs):
            unet.train()
            train_loss_forget = 0.0  # TODO: plot graph of losses after training
            train_loss_retain = 0.0
            for step, batch_forget in enumerate(train_forget_dataloader):
                batch_retain = next(iter(train_retain_dataloader))
                min_length = min(len(batch_forget["pixel_values"]), len(batch_retain["pixel_values"]))
                batch_forget["pixel_values"] = batch_forget["pixel_values"][:min_length]
                batch_retain["pixel_values"] = batch_retain["pixel_values"][:min_length]
                batch_forget["input_ids"] = batch_forget["input_ids"][:min_length]
                batch_retain["input_ids"] = batch_retain["input_ids"][:min_length]
                assert batch_forget["pixel_values"].shape == batch_retain["pixel_values"].shape
                
                batch_forget["pixel_values"] = batch_forget["pixel_values"].to(accelerator.device)
                batch_retain["pixel_values"] = batch_retain["pixel_values"].to(accelerator.device)
                
                batch_forget["input_ids"] = batch_forget["input_ids"].to(accelerator.device)
                batch_retain["input_ids"] = batch_retain["input_ids"].to(accelerator.device)

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents_forget = vae.encode(batch_forget["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents_forget = latents_forget * vae.config.scaling_factor

                    latents_retain = vae.encode(batch_retain["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents_retain = latents_retain * vae.config.scaling_factor

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
                    timesteps_forget = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_forget.device)
                    timesteps_forget = timesteps_forget.long()
                    timesteps_retain = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_retain.device)
                    timesteps_retain = timesteps_retain.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents_forget = noise_scheduler.add_noise(latents_forget, noise_forget, timesteps_forget)
                    noisy_latents_retain = noise_scheduler.add_noise(latents_retain, noise_forget, timesteps_forget)

                    # Get the text embedding for conditioning
                    encoder_hidden_states_forget = text_encoder(batch_forget["input_ids"], return_dict=False)[0]
                    encoder_hidden_states_retain = text_encoder(batch_retain["input_ids"], return_dict=False)[0]

                    # Get the target for loss depending on the prediction type
                    if self.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=self.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target_forget = noise_forget
                        target_retain = noise_retain
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target_forget = noise_scheduler.get_velocity(latents_forget, noise_forget, timesteps_forget)
                        target_retain = noise_scheduler.get_velocity(latents_retain, noise_retain, timesteps_retain)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred_forget = unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states_forget, return_dict=False)[0]
                    model_pred_retain = unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states_retain, return_dict=False)[0]

                    loss_forget = F.mse_loss(model_pred_forget.float(), target_forget.float(), reduction="mean")  # This is a Tensor of shape [], aka is a float
                    loss_retain = F.mse_loss(model_pred_retain.float(), target_retain.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    train_loss_forget += accelerator.gather(loss_forget.repeat(self.train_batch_size)).mean().item() / self.gradient_accumulation_steps
                    train_loss_retain += accelerator.gather(loss_retain.repeat(self.train_batch_size)).mean().item() / self.gradient_accumulation_steps

                    #########################################
                    # Backpropagate
                    #########################################
                    
                    # This is how it was before the munba trick:
                    #accelerator.backward(loss)
                    #if accelerator.sync_gradients:
                    #    params_to_clip = lora_layers
                    #    accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    #optimizer.step()
                    #lr_scheduler.step()
                    #optimizer.zero_grad()
                    
                    # This is with the munba trick:
                    
                    # Compute gradients
                    optimizer.zero_grad()
                    accelerator.backward(loss_forget)
                    grads_forget = [p.grad.clone() for p in unet.parameters() if p.requires_grad]  # This list has 256 elements; each element is a torch.Tensor of shapes like [4, 320], then [320, 4], then [4, 640], then [640, 4], etc
                    
                    optimizer.zero_grad()
                    accelerator.backward(loss_retain)
                    grads_retain = [p.grad.clone() for p in unet.parameters() if p.requires_grad]
                    
                    #for e in grads_forget:
                    #    print(e.shape)
                    scaled_grad = self.gradient_weighting_method.weight_grads(grads_forget, grads_retain, accelerator)

                    if self.compute_gradient_conflict:
                        similarities_gr.append(F.cosine_similarity(scaled_grad[:, 0], torch.cat([g.view(-1) for g in grads_retain]), dim=0).item())
                        similarities_gf.append(F.cosine_similarity(scaled_grad[:, 0], torch.cat([g.view(-1) for g in grads_forget]), dim=0).item())

                    # Overwrite gradients for the optimizer
                    for param, update in zip((p for p in unet.parameters() if p.requires_grad), 
                                            torch.split(scaled_grad, [p.numel() for p in unet.parameters() if p.requires_grad])):
                        param.grad = update.view(param.shape)

                    # Gradient clipping
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers
                        accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)

                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
        
                    #########################################
                    # End of Backpropagate
                    #########################################

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss_forget": train_loss_forget}, step=global_step)
                    accelerator.log({"train_loss_retain": train_loss_retain}, step=global_step)
                    train_loss_forget = 0.0
                    train_loss_retain = 0.0

                    if global_step % self.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            unwrapped_unet = unwrap_model(unet, accelerator)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss_forget.detach().item(), "step_loss_forget": loss_forget.detach().item(), "step_loss_retain": loss_retain.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.max_train_steps:
                    break

            if accelerator.is_main_process:
                if self.validation_prompt is not None and epoch % self.validation_epochs == 0:
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.pretrained_model_name_or_path,
                        unet=unwrap_model(unet, accelerator),
                        revision=self.revision,
                        variant=self.variant,
                        torch_dtype=weight_dtype,
                    )
                    _ = log_validation(pipeline, accelerator, epoch, self.num_validation_images, self.validation_prompt, self.seed)

                    del pipeline
                    torch.cuda.empty_cache()

        images: Dict[str, Image] = {}
        if self.compute_gradient_conflict:
            similarities_gr = list(filter(lambda e: not np.isnan(e), similarities_gr))  # TODO: why are there nan values?
            similarities_gf = list(filter(lambda e: not np.isnan(e), similarities_gf))
            images['histogram_conflict_gr'] = plot_gradient_conflict_hist(similarities_gr, r"Cosine Similarity between $\tilde{g}$ and $g_r$", "#1f77b4")  # Another nice color: #f4b400
            images['histogram_conflict_gf'] = plot_gradient_conflict_hist(similarities_gf, r"Cosine Similarity between $\tilde{g}$ and $g_f$", "#1f77b4")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save the lora layers
            unet = unet.to(torch.float32)
            unwrapped_unet = unwrap_model(unet, accelerator)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=self.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

            t3 = time.time()

            # Final inference
            # Load previous pipeline
            if self.validation_prompt is not None:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    revision=self.revision,
                    variant=self.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline.load_lora_weights(self.output_dir)  # load attention processors
                images.update(log_validation(pipeline, accelerator, epoch, self.num_validation_images, self.validation_prompt, self.seed, is_final_validation=True))  # run inference

            pipeline_original, pipeline_learned, pipeline_unlearned = unlearn_lora(self.pretrained_model_name_or_path, self.output_dir, device=accelerator.device)

            evaluator = EvaluatorTextToImage(
                pipeline_original=pipeline_original,
                pipeline_learned=pipeline_learned,
                pipeline_unlearned=pipeline_unlearned,
                prompts_forget=self.final_eval_prompts_forget,
                prompts_retain=self.final_eval_prompts_retain,
                metric_clip=MetricImageTextSimilarity(metrics=['clip']),
                compute_runtimes=self.compute_runtimes,
            )

            eval_results, images2 = evaluator.evaluate()
            images.update(images2)

            t4 = time.time()

            metric_common_attributes = {
                "task_type": "text-to-image",
                "dataset_type": f"forget-and-retain-together",
                "dataset_name": f"{self.dataset_forget_name} (forget) and {self.dataset_retain_name} (retain) sets",
            }


            if self.compute_runtimes:
                eval_results.append(EvalResult(
                    metric_type = 'runtime',
                    metric_name = f'Runtime init seconds (~↓)',
                    metric_value = t1-t0,
                    **metric_common_attributes,
                ))
                eval_results.append(EvalResult(
                    metric_type = 'runtime',
                    metric_name = f'Runtime data loading seconds (~↓)',
                    metric_value = t2-t1,
                    **metric_common_attributes,
                ))
                eval_results.append(EvalResult(
                    metric_type = 'runtime',
                    metric_name = f'Runtime training seconds (↓)',
                    metric_value = t3-t2,
                    **metric_common_attributes,
                ))
                eval_results.append(EvalResult(
                    metric_type = 'runtime',
                    metric_name = f'Runtime eval seconds (~↓)',
                    metric_value = t4-t3,
                    **metric_common_attributes,
                ))

            ################################
            if self.push_to_hub:
                save_model_card(
                    repo_id,
                    images=images,
                    base_model=self.pretrained_model_name_or_path,
                    dataset_forget_name=self.dataset_forget_name,
                    dataset_retain_name=self.dataset_retain_name,
                    repo_folder=self.output_dir,
                    eval_results=eval_results,
                    tags = [
                        "stable-diffusion",
                        "stable-diffusion-diffusers",
                        "text-to-image",
                        "diffusers",
                        "diffusers-training",
                        "lora",
                    ],
                    hyperparameters={k: v for k, v in self.model_dump().items() if isinstance(v, (str, float, int, type(None)))},
                    similarities_gr=similarities_gr,
                    similarities_gf=similarities_gf,
                )

                upload_folder(
                    repo_id=repo_id,
                    folder_path=self.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()

        logger.info('Training completed successfully =D')
