
import os
import math
import shutil
import time
from pathlib import Path
import random
from typing import List, Optional, Tuple, Dict, Any
from pydantic import Field
from abc import abstractmethod
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import set_seed

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
import peft
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm

from vision_unlearning.unlearner.base import Unlearner, logger
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.evaluator import EvaluatorTextToImage, plot_gradient_conflict_hist, log_validation
from vision_unlearning.utils.model_management import save_model_card
from vision_unlearning.utils.training import unwrap_model, preprocess_train, collate_fn
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethod, GradientWeightingMethodSimple


def unlearn_lora(
    model_original_id: str,
    model_lora_id: str,
    device: str,
    weight_name: str = "pytorch_lora_weights.safetensors",
    requires_inversion: bool = True,
    return_original: bool = True,
    return_learned: bool = True,
) -> Tuple[Optional[StableDiffusionPipeline], Optional[StableDiffusionPipeline], StableDiffusionPipeline]:
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
    pipeline_original: Optional[StableDiffusionPipeline] = None
    if return_original:
        pipeline_original = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)

    pipeline_learned: Optional[StableDiffusionPipeline] = None
    if return_learned:
        pipeline_learned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
        pipeline_learned.load_lora_weights(model_lora_id, weight_name=weight_name)  # type: ignore

    pipeline_unlearned = AutoPipelineForText2Image.from_pretrained(model_original_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline_unlearned.load_lora_weights(model_lora_id, weight_name=weight_name)

    # TODO: put inversion in function
    if requires_inversion:
        # pipeline_unlearned is inverted, pipeline_learned remains as it was trained
        # Munba, for example, falls in this case
        total: int = 0
        sum_before_invert: float = sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
        for name, param in pipeline_unlearned.unet.named_parameters():
            if "lora_A" in name:
                logger.debug(f"Inverting param {name}")
                param.data = -1 * param.data
                total += 1
        assert sum_before_invert == -sum([float(param.sum()) for name, param in pipeline_unlearned.unet.named_parameters() if "lora_A" in name])
        assert total > 0
        logger.info(f"Inverted {total} params for pipeline_unlearned")
    else:
        # pipeline_unlearned remains as it was trained, pipeline_learned is inverted
        # FADE, for example, falls in this case
        if return_learned:
            total: int = 0  # type: ignore
            sum_before_invert: float = sum([float(param.sum()) for name, param in pipeline_learned.unet.named_parameters() if "lora_A" in name])  # type: ignore
            for name, param in pipeline_learned.unet.named_parameters():  # type: ignore
                if "lora_A" in name:
                    logger.debug(f"Inverting param {name}")
                    param.data = -1 * param.data
                    total += 1
            assert sum_before_invert == -sum([float(param.sum()) for name, param in pipeline_learned.unet.named_parameters() if "lora_A" in name])  # type: ignore
            assert total > 0
            logger.info(f"Inverted {total} params for pipeline_learned")
    return pipeline_original, pipeline_learned, pipeline_unlearned


class UnlearnerLora(Unlearner):
    '''
    Fine-tuning script for Stable Diffusion for text2image with support for LoRA.
    Strongly based on the huggingface example (see credits in the end)

    Adapted from The HuggingFace Inc. team. All rights reserved.
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    '''
    # General arguments
    lora_r: int = Field(default=32, description="Dimensionality of the LoRA rank (R).")
    lora_alpha: int = Field(default=64, description="Lora alpha.")
    # lora_dropout: float = Field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: List[str] = Field(default=["to_k", "to_q", "to_v", "to_out.0"], description="Which module will be added the lora adapter.")  # See docs: https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig
    is_lora_negated: bool = Field(default=True, description="If Lora is trained to be good at the task (as suggestion by Zhang2023). If true, the trained model should be inverted using `unlearn_lora` before usage")  # noqa
    seed: int = Field(default=42, description="Random seed for initialization.")

    # Training arguments
    model_name_or_path: str = Field(description="Path to the pre-trained model or model identifier from huggingface.co/models")
    revision: Optional[str] = Field(None, description="Revision of pretrained model identifier from huggingface.co/models.")
    variant: Optional[str] = Field(None, description="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g., fp16.")
    # tokenizer_name: Optional[str] = Field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})

    # Dataset related
    dataset_forget_name: str = Field(..., description="The name or path of the dataset to be forgotten.")
    dataset_retain_name: str = Field(..., description="The name or path of the dataset to be retained.")
    dataset_forget_config_name: Optional[str] = Field(None, description="The config of the dataset for forgetting, leave as None if there's only one config.")
    dataset_retain_config_name: Optional[str] = Field(None, description="The config of the dataset for retaining, leave as None if there's only one config.")

    image_column: str = Field("image", description="The column of the dataset containing an image.")
    caption_column: str = Field("text", description="The column of the dataset containing a caption or a list of captions.")

    validation_prompt: Optional[str] = Field(None, description="A prompt that is sampled during training for inference.")
    num_validation_images: int = Field(4, description="Number of images to generate during validation with `validation_prompt`.")
    validation_epochs: int = Field(1, description="Run fine-tuning validation every X epochs.")

    resolution: int = Field(512, description="Resolution for input images.")
    center_crop: bool = Field(False, description="Whether to center crop the input images.")
    random_flip: bool = Field(False, description="Whether to randomly flip images horizontally.")

    max_train_samples: Optional[int] = Field(None, description="Limit the number of training examples for debugging or quicker training.")
    dataloader_num_workers: int = Field(0, description="Number of subprocesses for data loading.")

    final_eval_prompts_forget: str | List[str] = Field([], description="Prompts for final evaluation on the forget dataset (ModelHub identifier or directly the prompts).")
    final_eval_prompts_retain: str | List[str] = Field([], description="Prompts for final evaluation on the retain dataset (ModelHub identifier or directly the prompts).")
    prediction_type: Optional[str] = Field(None, description="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. "
                                                             "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.")

    # training_args (from huggingface)
    # do_train: bool = Field(default=True, metadata={"help": "Whether to run training."})
    # do_eval: bool = Field(default=False, metadata={"help": "Whether to run evaluation on the validation set."})
    per_device_train_batch_size: int = Field(default=1, description="Batch size per device during training.")
    gradient_accumulation_steps: int = Field(default=128, description="Number of updates steps to accumulate before performing a backward/update pass.")
    num_train_epochs: int = Field(default=1, description="Total number of training epochs to perform.")
    learning_rate: float = Field(default=3e-4, description="The initial learning rate for AdamW.")
    lr_scheduler_type: str = Field(default="cosine", description="The scheduler type to use.")

    should_log: bool = Field(default=True, description="Whether to log the training process.")  # TODO not used?
    local_rank: int = Field(default=-1, description="Local rank for distributed training.")  # TODO not used?
    device: str = Field(default="cuda", description="Device to use for training.")
    n_gpu: int = Field(default=1, description="Number of GPUs to use.")  # TODO not used?

    # Other stuff (some for compatibility with UnlearnerLora)
    output_dir: str = Field(default="assets", description="The output directory where the model predictions and checkpoints will be written.")
    cache_dir: Optional[str] = Field(None, description="Directory where downloaded models and datasets will be stored.")
    hub_token: Optional[str] = Field(None, description="Token for authentication to push to Model Hub.")
    hub_model_id: Optional[str] = Field(None, description="Repository name to sync with `output_dir`. None for not push")
    logging_dir: str = Field(default="logs", description="Directory for storing logs.")
    logging_steps: int = Field(default=20, description="Log every X updates steps.")  # TODO not used?
    save_strategy: str = Field(default="epoch", description="The checkpoint save strategy to adopt during training.")  # TODO not used?
    save_total_limit: int = Field(default=2, description="Limit the total amount of checkpoints.")  # TODO not used?

    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory at the expense of slower backward pass.")
    enable_xformers_memory_efficient_attention: bool = Field(False, description="Use xformers for memory-efficient attention.")
    mixed_precision: Optional[str] = Field(None, description="Use mixed precision training: 'fp16' or 'bf16'.")
    allow_tf32: bool = Field(False, description="Allow TF32 on Ampere GPUs for potential training speed-up.")
    use_8bit_adam: bool = Field(False, description="Use 8-bit Adam optimizer from bitsandbytes.")
    report_to: str = Field("tensorboard", description="Logging integration for reporting results (e.g., tensorboard, wandb).")

    gradient_weighting_method: GradientWeightingMethod = Field(..., description="The method to use for weighting the gradients.")
    compute_gradient_conflict: bool = Field(False, description="Whether to compute the gradient conflict, for evaluation purposes.")
    compute_runtimes: bool = Field(True, description="Whether to compute the runtimes of the training, for evaluation purposes.")
    max_train_steps: Optional[int] = Field(None, description="Total number of training steps, overrides num_train_epochs if provided.")
    lr_warmup_steps: int = Field(500, description="Number of warmup steps in the learning rate scheduler.")
    adam_beta1: float = Field(0.9, description="Beta1 parameter for Adam optimizer.")
    adam_beta2: float = Field(0.999, description="Beta2 parameter for Adam optimizer.")
    adam_weight_decay: float = Field(1e-2, description="Weight decay for Adam optimizer.")
    adam_epsilon: float = Field(1e-8, description="Epsilon value for Adam optimizer.")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm.")
    checkpointing_steps: int = Field(500, description="Save training state checkpoint every X updates.")
    checkpoints_total_limit: Optional[int] = Field(None, description="Maximum number of checkpoints to store.")
    resume_from_checkpoint: Optional[str] = Field(None, description="Resume training from a previous checkpoint.")
    noise_offset: float = Field(0.0, description="Scale of noise offset.")

    # Internal attributes
    # TODO: proper types
    _accelerator: Optional[Accelerator] = None
    _output_dir_checkpoints: Optional[str] = None
    _output_dir_lora: Optional[str] = None
    _lora_weight_name: str = 'pytorch_lora_weights.safetensors'
    _images: Dict[str, Image.Image] = {}
    _weight_dtype: Any = torch.float32
    _similarities_gr: List[float] = []  # Cosine similarlities between \tilde g and g_r, one element per step update
    _similarities_gf: List[float] = []  # Cosine similarlities between \tilde g and g_f, one element per step update

    _noise_scheduler: Any = None
    _tokenizer: Any = None
    _text_encoder: Any = None
    _vae: Any = None
    _unet: Optional[diffusers.models.unets.unet_2d_condition.UNet2DConditionModel] = None

    _optimizer: Any = None
    _lr_scheduler: Any = None
    _lora_layers: Any = None

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        self._output_dir_checkpoints = self.output_dir
        self._output_dir_lora = self.output_dir

    def _pre_checks(self) -> None:
        # TODO maybe this should use some native pydantic validation?
        if isinstance(self.final_eval_prompts_retain, str):
            raise NotImplementedError("final_eval_prompts_retain should be a list of prompts, not a string.")
        if isinstance(self.final_eval_prompts_forget, str):
            raise NotImplementedError("final_eval_prompts_forget should be a list of prompts, not a string.")

        if self.report_to == "wandb" and self.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        if not self.is_lora_negated:
            # TODO: this shiould be a simple matter of following the gradinet or its negation
            raise NotImplementedError()

    def _get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,  # type: ignore  # TODO: should this be int or float?
            init_lora_weights="gaussian",
            target_modules=self.target_modules,
        )

    def _get_accelerator(self):
        return Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_config=accelerate.utils.ProjectConfiguration(
                project_dir=self.output_dir,
                logging_dir=Path(self.output_dir, self.logging_dir),
            ),
        )

    def _hook_after_lora_init(self):
        return None

    def _hook_before_load_model(self):
        return None

    def _save_lora_layers(self):
        '''
        Side-effects: modifies self._unet in-place (casts to float32), saves two directories self._output_dir_super and self._output_dir_sub
        '''
        assert self._unet is not None
        self._unet = self._unet.to(torch.float32)
        unwrapped_unet = unwrap_model(self._unet, self._accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=self.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    def train(self):
        self._pre_checks()
        t0 = time.time()

        os.makedirs(self.output_dir, exist_ok=True)

        set_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Acelerator config
        self._accelerator = self._get_accelerator()

        if self._accelerator.mixed_precision == "fp16":
            logger.info('Using weight_dtype=float16')
            self._weight_dtype = torch.float16
        elif self._accelerator.mixed_precision == "bf16":
            logger.info('Using weight_dtype=bfloat16')
            self._weight_dtype = torch.bfloat16

        # TODO
        # Handle the repository creation
        # if accelerator.is_main_process:
        #    if self.output_dir is not None:
        #        os.makedirs(self.output_dir, exist_ok=True)
        #    if self.push_to_hub:
        #        repo_id = create_repo(repo_id=self.hub_model_id or Path(self.output_dir).name, exist_ok=True, token=self.hub_token).repo_id

        # Load scheduler, tokenizer and models
        self._hook_before_load_model()
        self._noise_scheduler = DDPMScheduler.from_pretrained(self.model_name_or_path, subfolder="scheduler")
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name_or_path, subfolder="tokenizer")
        self._text_encoder = CLIPTextModel.from_pretrained(self.model_name_or_path, subfolder="text_encoder")
        self._vae = AutoencoderKL.from_pretrained(self.model_name_or_path, subfolder="vae")
        self._unet = UNet2DConditionModel.from_pretrained(self.model_name_or_path, subfolder="unet")
        assert self._noise_scheduler is not None
        assert self._tokenizer is not None
        assert self._text_encoder is not None
        assert self._vae is not None
        assert self._unet is not None

        # freeze parameters of models to save more memory
        self._unet.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._text_encoder.requires_grad_(False)

        self._unet.to(self._accelerator.device, dtype=self._weight_dtype)
        self._vae.to(self._accelerator.device, dtype=self._weight_dtype)
        self._text_encoder.to(self._accelerator.device, dtype=self._weight_dtype)

        # Add adapter and make sure the trainable params are in float32.
        self._unet.add_adapter(self._get_lora_config())

        if self.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(self._unet, dtype=torch.float32)

        if self.enable_xformers_memory_efficient_attention:
            raise NotImplementedError()
        # TODO
        # from diffusers.utils.import_utils import is_xformers_available
        # if self.enable_xformers_memory_efficient_attention:
        #    if is_xformers_available():
        #        import xformers
        #        xformers_version = version.parse(xformers.__version__)
        #        if xformers_version == version.parse("0.0.16"):
        #            logger.warning(
        #                "xFormers 0.0.16 cannot be used for training in some GPUs. "
        #                "If you observe problems during training, please update xFormers to at least 0.0.17. "
        #                "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
        #            )
        #        self._unet.enable_xformers_memory_efficient_attention()
        #    else:
        #        raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self._accelerator.mixed_precision == "fp16":
            logger.info('Using weight_dtype=float16')
            self._weight_dtype = torch.float16
        elif self._accelerator.mixed_precision == "bf16":
            logger.info('Using weight_dtype=bfloat16')
            self._weight_dtype = torch.bfloat16

        self._hook_after_lora_init()

        self._lora_layers = filter(lambda p: p.requires_grad, self._unet.parameters())
        logger.info(f"Number of lora layers: {len(list(filter(lambda p: p.requires_grad, self._unet.parameters())))}")  # I think this _has_ to be recalculated, even if it looks ugly, not sure
        # [x for x in self._lora_layers]

        if self.gradient_checkpointing:
            self._unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-s
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.use_8bit_adam:
            # try:
            #     import bitsandbytes as bnb
            # except ImportError:
            #     raise ImportError(
            #         "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            #     )
            # optimizer_cls = bnb.optim.AdamW8bit
            raise NotImplementedError()
        else:
            optimizer_cls = torch.optim.AdamW

        self._optimizer = optimizer_cls(
            self._lora_layers,
            lr=self.learning_rate,  # TODO: should this use _lr_scheduler?
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        # for idx, layer in enumerate(self._unet.modules()):
        #     named_modules = dict(layer.named_modules())
        #     print(named_modules.keys())
        #     break

        t1 = time.time()

        train_forget_dataloader, train_retain_dataloader = self._prepare_dataloaders()

        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        num_warmup_steps_for_scheduler = self.lr_warmup_steps * self._accelerator.num_processes
        if self.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(train_forget_dataloader) / self._accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                self.num_train_epochs * num_update_steps_per_epoch * self._accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.max_train_steps * self._accelerator.num_processes

        self._lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self._optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

        # Prepare everything with our `self._accelerator`.
        self._unet, self._optimizer, train_forget_dataloader, self._lr_scheduler = self._accelerator.prepare(
            # self._unet, self._optimizer, train_forget_dataloader, train_retain_dataloader, self._lr_scheduler   # TODO: what has to be changed so this works? I guess BOHT datloaders hsould pass though accelerate; it works without but probably there is some perforamcne deradation  # noqa
            self._unet, self._optimizer, train_forget_dataloader, self._lr_scheduler
        )

        # Recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_forget_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != self.max_train_steps * self._accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'self._accelerator.prepare' ({len(train_forget_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)  # Recalculate our number of training epochs

        # Initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self._accelerator.is_main_process:
            self._accelerator.init_trackers("text2image-fine-tune", config={k: v for k, v in self.model_dump().items() if isinstance(v, (str, float, int, type(None)))})

        # Train!
        t2 = time.time()
        total_batch_size = self.per_device_train_batch_size * self._accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            path: Optional[str]
            if self.resume_from_checkpoint != "latest":
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self._output_dir_checkpoints)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self._accelerator.print(
                    f"Checkpoint '{self.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                assert self._output_dir_checkpoints is not None
                self._accelerator.print(f"Resuming from checkpoint {path}")
                self._accelerator.load_state(os.path.join(self._output_dir_checkpoints, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self._accelerator.is_local_main_process,  # Only show the progress bar once on each machine.
        )

        for epoch in range(first_epoch, self.num_train_epochs):
            assert self._unet is not None
            self._unet.train()
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

                batch_forget["pixel_values"] = batch_forget["pixel_values"].to(self._accelerator.device)
                batch_retain["pixel_values"] = batch_retain["pixel_values"].to(self._accelerator.device)

                batch_forget["input_ids"] = batch_forget["input_ids"].to(self._accelerator.device)
                batch_retain["input_ids"] = batch_retain["input_ids"].to(self._accelerator.device)

                with self._accelerator.accumulate(self._unet):
                    loss_forget, loss_retain = self._train_one_batch(batch_forget, batch_retain)
                    # Gather the losses across all processes for logging (if we use distributed training).
                    train_loss_forget += self._accelerator.gather(loss_forget.repeat(self.per_device_train_batch_size)).mean().item() / self.gradient_accumulation_steps
                    train_loss_retain += self._accelerator.gather(loss_retain.repeat(self.per_device_train_batch_size)).mean().item() / self.gradient_accumulation_steps

                # Checks if the self._accelerator has performed an optimization step behind the scenes
                if self._accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self._accelerator.log({"train_loss_forget": train_loss_forget}, step=global_step)
                    self._accelerator.log({"train_loss_retain": train_loss_retain}, step=global_step)
                    train_loss_forget = 0.0
                    train_loss_retain = 0.0

                    # TODO: if global_step >= 2 and (global_step & (global_step - 1) == 0):
                    # ...

                    # Checkpointing
                    if global_step % self.checkpointing_steps == 0:
                        if self._accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self._output_dir_checkpoints)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.checkpoints_total_limit:
                                    assert self.checkpoints_total_limit > 0
                                    assert self._output_dir_checkpoints is not None
                                    num_to_remove = len(checkpoints) - self.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self._output_dir_checkpoints, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self._output_dir_checkpoints, f"checkpoint-{global_step}")  # type: ignore
                            self._accelerator.save_state(save_path)

                            unwrapped_unet = unwrap_model(self._unet, self._accelerator)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                            logger.info(f"Saved state to {save_path}")

                progress_bar.set_postfix(**{  # type: ignore
                    "step_loss": loss_forget.detach().item(), "step_loss_forget": loss_forget.detach().item(), "step_loss_retain": loss_retain.detach().item(), "lr": self._lr_scheduler.get_last_lr()[0]
                })

                if global_step >= self.max_train_steps:
                    break

            if self._accelerator.is_main_process:
                if self.validation_prompt is not None and epoch % self.validation_epochs == 0:
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.model_name_or_path,
                        unet=unwrap_model(self._unet, self._accelerator),
                        revision=self.revision,
                        variant=self.variant,
                        torch_dtype=self._weight_dtype,
                        safety_checker=None,
                    )
                    self._images.update(log_validation(pipeline, self._accelerator, epoch, self.num_validation_images, self.validation_prompt, self.seed))

                    del pipeline
                    torch.cuda.empty_cache()

        self._accelerator.wait_for_everyone()
        if self._accelerator.is_main_process:
            self._save_lora_layers()

        del self._unet
        del self._noise_scheduler
        del self._tokenizer
        del self._text_encoder
        del self._vae
        torch.cuda.empty_cache()

        ###############################
        # Post training evaluation
        # TODO: most of this should be done by the EvaluatorTextToImage
        if self.compute_gradient_conflict:
            self._similarities_gr = list(filter(lambda e: not np.isnan(e), self._similarities_gr))  # TODO: why are there nan values?
            self._similarities_gf = list(filter(lambda e: not np.isnan(e), self._similarities_gf))
            self._images['histogram_conflict_gr'] = plot_gradient_conflict_hist(self._similarities_gr, r"Cosine Similarity between $\tilde{g}$ and $g_r$", "#1f77b4")  # Another nice color: #f4b400
            self._images['histogram_conflict_gf'] = plot_gradient_conflict_hist(self._similarities_gf, r"Cosine Similarity between $\tilde{g}$ and $g_f$", "#1f77b4")

        # Final inference
        if self._accelerator.is_main_process:
            t3 = time.time()
            if self.validation_prompt is not None:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_name_or_path,
                    revision=self.revision,
                    variant=self.variant,
                    torch_dtype=self._weight_dtype,
                )
                pipeline.load_lora_weights(self._output_dir_lora)  # load attention processors
                self._images.update(log_validation(pipeline, self._accelerator, epoch, self.num_validation_images, self.validation_prompt, self.seed, is_final_validation=True))  # run inference
                del pipeline
                torch.cuda.empty_cache()

            assert self._output_dir_lora is not None
            pipeline_original, pipeline_learned, pipeline_unlearned = unlearn_lora(self.model_name_or_path, self._output_dir_lora, device=self._accelerator.device, weight_name=self._lora_weight_name, requires_inversion=self.is_lora_negated)

            assert type(self.final_eval_prompts_forget) == list  # noqa
            assert type(self.final_eval_prompts_retain) == list  # noqa

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
            self._images.update(images2)

            t4 = time.time()

            metric_common_attributes = {
                "task_type": "text-to-image",
                "dataset_type": f"forget-and-retain-together",
                "dataset_name": f"{self.dataset_forget_name} (forget) and {self.dataset_retain_name} (retain) sets",
            }

            if self.compute_runtimes:
                eval_results.append(EvalResult(
                    metric_type='runtime',
                    metric_name=f'Runtime init seconds (~↓)',
                    metric_value=t1 - t0,
                    **metric_common_attributes,  # type: ignore
                ))
                eval_results.append(EvalResult(
                    metric_type='runtime',
                    metric_name=f'Runtime data loading seconds (~↓)',
                    metric_value=t2 - t1,
                    **metric_common_attributes,  # type: ignore
                ))
                eval_results.append(EvalResult(
                    metric_type='runtime',
                    metric_name=f'Runtime training seconds (↓)',
                    metric_value=t3 - t2,
                    **metric_common_attributes,  # type: ignore
                ))
                eval_results.append(EvalResult(
                    metric_type='runtime',
                    metric_name=f'Runtime eval seconds (~↓)',
                    metric_value=t4 - t3,
                    **metric_common_attributes,  # type: ignore
                ))

            assert self._output_dir_lora is not None
            save_model_card(
                str(self.hub_model_id),
                images=self._images,
                base_model=self.model_name_or_path,
                dataset_forget_name=self.dataset_forget_name,
                dataset_retain_name=self.dataset_retain_name,
                repo_folder=self._output_dir_lora,
                eval_results=eval_results,
                tags=[
                    "stable-diffusion",
                    "stable-diffusion-diffusers",
                    "text-to-image",
                    "diffusers",
                    "diffusers-training",
                    "lora",
                ],
                hyperparameters={k: v for k, v in self.model_dump().items() if isinstance(v, (str, float, int, type(None)))},
                similarities_gr=self._similarities_gr,
                similarities_gf=self._similarities_gf,
            )

            if self.hub_model_id is not None:
                upload_folder(
                    repo_id=self.hub_model_id,
                    folder_path=self._output_dir_lora,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self._accelerator.end_training()

        logger.info('Training completed successfully =D')

        # TODO: merging the lora the with model isn't supported
        # Merge is tricky with diffusers, see https://github.com/huggingface/diffusers/issues/2900
        # def merge(self, model):

        return eval_results

    @abstractmethod
    def _prepare_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def _train_one_batch(self, batch_forget, batch_retain):
        pass


class UnlearnerLoraDirect(UnlearnerLora):
    '''
    Straight-forward finetuning
    '''

    def _prepare_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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

        logger.info(f'Retain dataset: {dataset_retain}\nForget dataset: {dataset_forget}')

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset_forget["train"].column_names
        if self.image_column not in column_names:
            raise ValueError(f"image_column value '{self.image_column}' needs to be one of: {', '.join(column_names)}")
        if self.caption_column not in column_names:
            raise ValueError(f"caption_column' value '{self.caption_column}' needs to be one of: {', '.join(column_names)}")
        logger.info(f"Dataset config - Image column: {self.image_column}, caption column: {self.caption_column}")

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
        assert self._accelerator is not None
        with self._accelerator.main_process_first():
            if self.max_train_samples is not None:
                dataset_forget["train"] = dataset_forget["train"].shuffle(seed=self.seed).select(range(self.max_train_samples))
            train_dataset_forget = dataset_forget["train"].with_transform(lambda examples: preprocess_train(examples, self._tokenizer, self.caption_column, self.image_column, train_transforms))
            train_dataset_retain = dataset_retain["train"].with_transform(lambda examples: preprocess_train(examples, self._tokenizer, self.caption_column, self.image_column, train_transforms))

        # DataLoaders creation:
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
        logger.info(f"Number of training examples = {len(train_dataset_forget)} + {len(train_dataset_retain)}")
        return train_forget_dataloader, train_retain_dataloader

    def _train_one_batch(self, batch_forget, batch_retain):
        # Convert images to latent space
        latents_forget = self._vae.encode(batch_forget["pixel_values"].to(dtype=self._weight_dtype)).latent_dist.sample()
        latents_forget = latents_forget * self._vae.config.scaling_factor

        latents_retain = self._vae.encode(batch_retain["pixel_values"].to(dtype=self._weight_dtype)).latent_dist.sample()
        latents_retain = latents_retain * self._vae.config.scaling_factor

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
        noisy_latents_retain = self._noise_scheduler.add_noise(latents_retain, noise_forget, timesteps_forget)

        # Get the text embedding for conditioning
        encoder_hidden_states_forget = self._text_encoder(batch_forget["input_ids"], return_dict=False)[0]
        encoder_hidden_states_retain = self._text_encoder(batch_retain["input_ids"], return_dict=False)[0]

        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self._noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        if self._noise_scheduler.config.prediction_type == "epsilon":
            target_forget = noise_forget
            target_retain = noise_retain
        elif self._noise_scheduler.config.prediction_type == "v_prediction":
            target_forget = self._noise_scheduler.get_velocity(latents_forget, noise_forget, timesteps_forget)
            target_retain = self._noise_scheduler.get_velocity(latents_retain, noise_retain, timesteps_retain)
        else:
            raise ValueError(f"Unknown prediction type {self._noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred_forget = self._unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states_forget, return_dict=False)[0]  # type: ignore
        model_pred_retain = self._unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states_retain, return_dict=False)[0]  # type: ignore

        loss_forget = F.mse_loss(model_pred_forget.float(), target_forget.float(), reduction="mean")  # This is a Tensor of shape [], aka is a float
        loss_retain = F.mse_loss(model_pred_retain.float(), target_retain.float(), reduction="mean")

        #########################################
        # Backpropagate
        #########################################

        # This is how it was before the munba trick:
        # accelerator.backward(loss)
        # if accelerator.sync_gradients:
        #     params_to_clip = lora_layers
        #     accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
        # optimizer.step()
        # lr_scheduler.step()
        # optimizer.zero_grad()

        # This is with the munba trick:

        # Compute gradients
        self._optimizer.zero_grad()
        self._accelerator.backward(loss_forget)  # type: ignore
        grads_forget = [p.grad.clone() for p in self._unet.parameters() if p.requires_grad]  # type: ignore
        # This list has 256 elements; each element is a torch.Tensor of shapes like [4, 320], then [320, 4], then [4, 640], then [640, 4], etc

        self._optimizer.zero_grad()
        self._accelerator.backward(loss_retain)  # type: ignore
        grads_retain = [p.grad.clone() for p in self._unet.parameters() if p.requires_grad]  # type: ignore

        # for e in grads_forget:
        #    print(e.shape)
        scaled_grad = self.gradient_weighting_method.weight_grads(grads_forget, grads_retain, self._accelerator)

        if self.compute_gradient_conflict:
            self._similarities_gr.append(F.cosine_similarity(scaled_grad[:, 0], torch.cat([g.view(-1) for g in grads_retain]), dim=0).item())
            self._similarities_gf.append(F.cosine_similarity(scaled_grad[:, 0], torch.cat([g.view(-1) for g in grads_forget]), dim=0).item())

        # Overwrite gradients for the optimizer
        for param, update in zip(
            (p for p in self._unet.parameters() if p.requires_grad),  # type: ignore
            torch.split(scaled_grad, [p.numel() for p in self._unet.parameters() if p.requires_grad]),  # type: ignore
        ):
            param.grad = update.view(param.shape)

        # Gradient clipping
        if self._accelerator.sync_gradients:  # type: ignore
            params_to_clip = self._lora_layers
            self._accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)  # type: ignore

        # Optimizer step
        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad()

        return loss_forget, loss_retain
