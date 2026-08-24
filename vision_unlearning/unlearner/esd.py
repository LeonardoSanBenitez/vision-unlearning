from __future__ import annotations
import os
import time
import random
import contextlib
from typing import Optional, List, Tuple, Dict, Any, Iterator, Protocol, cast
from PIL import Image

import torch
import torch.nn.functional as F
from pydantic import Field
from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoPipelineForText2Image
from huggingface_hub.repocard_data import EvalResult
from huggingface_hub import upload_folder

from vision_unlearning.unlearner.base import Unlearner, logger
from vision_unlearning.evaluator import EvaluatorTextToImage
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.utils.model_management import save_model_card


ESD_WEIGHTS_FILENAME = 'esd_sd_weights.safetensors'

#: Module classes whose parameters may be selected for training. Mirrors the reference
#: implementation's own set; the two ``LoRACompatible*`` names are diffusers' legacy aliases and are
#: included so the selection does not silently shrink on an older checkpoint.
TARGET_MODULE_TYPES = frozenset({
    'Linear',
    'Conv2d',
    'LoRACompatibleLinear',
    'LoRACompatibleConv',
})


class _DiffusionPipelineComponents(Protocol):
    """The parts of a Stable Diffusion pipeline this unlearner touches.

    Diffusers resolves a pipeline's components dynamically, so annotating with a concrete pipeline
    class makes the type checker reject every `pipeline.unet` -- the annotation buys no safety and
    costs an error per access. The sibling weight-editing unlearner sidesteps that by leaving the
    type inferred, i.e. unchecked; declaring the seven members actually used is the same amount of
    code and checks them, including the argument of `_fit`, which is documented as drivable with a
    small stand-in rather than a real checkpoint.
    """

    unet: Any
    vae: Any
    text_encoder: Any
    scheduler: Any
    vae_scale_factor: int

    def encode_prompt(self, *arguments: Any, **keywords: Any) -> Any:
        raise NotImplementedError

    def set_progress_bar_config(self, *arguments: Any, **keywords: Any) -> None:
        raise NotImplementedError


class ESD(Unlearner):
    '''
    Erased Stable Diffusion (ESD): erases a concept by fine-tuning a slice of the denoiser against a
    negative-guidance target, with no dataset, no anchor concept and no low-rank adapter.
    Adapted from:
        GitHub: https://github.com/rohitgandikota/erasing
        Arxiv: https://arxiv.org/abs/2303.07345
        Gandikota, R., Materzyńska, J., Fiotto-Kaufman, J., & Bau, D. (2023).
        Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International
        Conference on Computer Vision (pp. 2426-2436).

    This unlearner does not use LoRA. It updates the selected denoiser tensors in place and saves
    only those tensors, which is the same artifact shape `UCE` produces (a partial weight file plus a
    static loader) and a different one from the adapter directory `UnlearnerLora` produces.

    One optimizer step works as follows. A denoising-step index ``k`` is drawn uniformly from
    ``0 .. num_inference_steps - 1``; the *frozen* denoiser is run with classifier-free guidance for
    ``k`` steps to produce an intermediate latent; three predictions are taken at that latent under
    no gradient, conditioned on the erased concept, on the empty prompt, and on the concept the
    erasure is anchored to; and the target is

        target = pred(erase_from) - negative_guidance * (pred(erase) - pred(null))

    The trainable copy of the denoiser then predicts on the same latent and the mean squared error
    against that target is minimised. Pushing the model's prediction *away* from the concept, rather
    than towards a replacement, is what distinguishes this method from a distillation onto an
    overwriting concept.
    '''

    # Specific to this unlearner
    pretrained_model_name_or_path: str = Field(
        default='CompVis/stable-diffusion-v1-4',
        description='Path to pretrained model or model identifier from huggingface.co/models.'
    )
    erase_concept: str = Field(..., description='The concept to erase, as it would be written in a prompt.')
    erase_from: Optional[str] = Field(
        default=None,
        description='Anchor the erasure to this concept instead of to the erased one, so that only the '
                    'erased concept is removed from it. None means anchor to the erased concept itself.'
    )
    train_method: str = Field(
        default='esd-x',
        description="Which denoiser parameters to train. 'esd-x' is cross-attention only (the parameters "
                    "that read the text conditioning), 'esd-u' everything except cross-attention, "
                    "'esd-all' every eligible parameter, 'esd-x-strict' only the cross-attention key and "
                    "value projections, 'selfattn' self-attention only."
    )
    negative_guidance: float = Field(
        default=2.0,
        description='How hard the model is pushed away from the concept. This is the strength dial: '
                    'larger values erase harder. Equalization against other methods tunes this.'
    )
    guidance_scale: float = Field(
        default=3.0,
        description='Classifier-free guidance scale used when sampling the intermediate latent that each '
                    'step trains on. Not the erasure strength; see negative_guidance.'
    )
    num_inference_steps: int = Field(
        default=50,
        description='Length of the denoising schedule the intermediate latent is sampled from. Each step '
                    'samples a random prefix of it, so this scales the cost of a step as well as the '
                    'range of noise levels the erasure is fitted over.'
    )
    resolution: int = Field(default=512, description='Resolution of the latents sampled during training.')

    # Training loop
    num_train_epochs: int = Field(
        default=200,
        description='Number of optimizer steps. Named for consistency with the other unlearners in this '
                    'library: this method has no dataset, so one epoch is one pass over the single forget '
                    'concept, which is one optimizer step.'
    )
    max_train_steps: Optional[int] = Field(
        default=None,
        description='Total number of optimizer steps, overrides num_train_epochs if provided.'
    )
    learning_rate: float = Field(default=5e-5, description='The initial learning rate for Adam.')
    seed: int = Field(default=42, description='Seed for the sampled denoising depths and the sampling noise.')
    mixed_precision: Optional[str] = Field(
        default=None,
        description="Precision the models are loaded and trained in: 'fp16', 'bf16', or None for float32."
    )

    # Evaluation
    final_eval_prompts_forget: str | List[str] = Field([], description='Prompts for final evaluation on the forget dataset (ModelHub identifier or directly the prompts).')
    final_eval_prompts_retain: str | List[str] = Field([], description='Prompts for final evaluation on the retain dataset (ModelHub identifier or directly the prompts).')

    # Other stuff (some for compatibility with the other unlearners)
    output_dir: str = Field(
        default='../esd_models',
        description='Output directory for model predictions and checkpoints.'
    )
    device: str = 'cuda:0'
    compute_runtimes: bool = Field(True, description='Whether to compute the runtimes of the training, for evaluation purposes.')
    hub_model_id: Optional[str] = Field(None, description='Repository name to sync with `output_dir`. None for not push')

    def __init__(self, **data: Any):
        '''Custom initializer for ESD with informative logging.'''
        super().__init__(**data)

        logger.info('\n[INFO] Initializing Erased Stable Diffusion (ESD)...')
        logger.info(f' - Base model:         {self.pretrained_model_name_or_path}')
        logger.info(f' - Device:             {self.device}')
        logger.info(f' - Erase concept:      {self.erase_concept}')
        logger.info(f' - Erase from:         {self.erase_from if self.erase_from is not None else self.erase_concept}')
        logger.info(f' - Train method:       {self.train_method}')
        logger.info(f' - Negative guidance:  {self.negative_guidance}')
        logger.info(f' - Guidance scale:     {self.guidance_scale}')
        logger.info(f' - Inference steps:    {self.num_inference_steps}')
        logger.info(f' - Optimizer steps:    {self._total_steps()}')
        logger.info(f' - Learning rate:      {self.learning_rate}')
        logger.info(f' - Resolution:         {self.resolution}')
        logger.info(f' - Seed:               {self.seed}')
        logger.info(f' - Output directory:   {self.output_dir}')
        logger.info(f' - Hub Model Id:       {self.hub_model_id}\n')

    ##########################################
    # Small helpers
    ##########################################
    def _total_steps(self) -> int:
        '''Optimizer steps this run will take. `max_train_steps` overrides the epoch count.'''
        return self.max_train_steps if self.max_train_steps is not None else self.num_train_epochs

    def _weight_dtype(self) -> torch.dtype:
        if self.mixed_precision == 'fp16':
            return torch.float16
        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _empty_device_cache(device: str) -> None:
        '''Free cached device memory, if there is a device and its context exists.

        `torch.cuda.is_available()` is not sufficient on the ROCm build used here: it returns True
        while the context is still uninitialized, and some `torch.cuda` calls raise in that state.
        '''
        if str(device).startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def _select_parameter_names(self, unet: torch.nn.Module) -> List[str]:
        '''Names of the denoiser parameters this train method trains, in module order.

        Selection is by module, not by parameter: a module of an eligible class whose name matches
        contributes all of its own parameters (weights and, where present, biases).
        '''
        def matches(module_name: str) -> bool:
            if self.train_method == 'esd-x':
                return 'attn2' in module_name
            if self.train_method == 'esd-u':
                return 'attn2' not in module_name
            if self.train_method == 'esd-all':
                return True
            if self.train_method == 'esd-x-strict':
                return 'attn2.to_k' in module_name or 'attn2.to_v' in module_name
            if self.train_method == 'selfattn':
                return 'attn1' in module_name
            raise ValueError(f'Unsupported train_method: {self.train_method!r}')

        selected: List[str] = []
        seen = set()
        for module_name, module in unet.named_modules():
            if type(module).__name__ not in TARGET_MODULE_TYPES:
                continue
            if not matches(module_name):
                continue
            for parameter_name, _ in module.named_parameters(recurse=False):
                full_name = f'{module_name}.{parameter_name}' if module_name else parameter_name
                if full_name in seen:
                    continue
                seen.add(full_name)
                selected.append(full_name)

        if not selected:
            raise ValueError(
                f'No trainable parameters were selected for train_method={self.train_method!r}. '
                'The checkpoint is not shaped the way this method expects.'
            )
        return selected

    ##########################################
    # Training
    ##########################################
    def train(self) -> List[EvalResult]:
        '''Erase the concept, save the modified tensors, evaluate, and return the evaluation records.

        The orchestration is deliberately the same as every other unlearner in this library: load,
        fit, save, evaluate. The steps are separate methods so that a test can drive one of them.
        '''
        # ==== Sanity checks ====
        t0 = time.time()
        assert self.pretrained_model_name_or_path, 'Pretrained model path must not be empty.'
        assert self.erase_concept, 'erase_concept must not be empty.'
        assert self.negative_guidance > 0, 'negative_guidance must be positive.'
        assert self.num_inference_steps > 0, 'num_inference_steps must be positive.'
        assert self._total_steps() > 0, 'The number of optimizer steps must be positive.'
        if isinstance(self.final_eval_prompts_retain, str):
            raise NotImplementedError('final_eval_prompts_retain should be a list of prompts, not a string.')
        if isinstance(self.final_eval_prompts_forget, str):
            raise NotImplementedError('final_eval_prompts_forget should be a list of prompts, not a string.')

        os.makedirs(self.output_dir, exist_ok=True)

        pipeline = self._load_pipeline()
        t1 = time.time()

        trained_tensors = self._fit(pipeline)
        t2 = time.time()

        self._save_weights(trained_tensors)
        del pipeline
        self._empty_device_cache(self.device)

        eval_results, eval_images = self.evaluate()
        t3 = time.time()

        metric_common_attributes = {
            'task_type': 'text-to-image',
            'dataset_type': 'forget-and-retain-together',
            'dataset_name': f'{self.erase_concept} (forget)',
        }
        if self.compute_runtimes:
            for metric_name, seconds in [
                ('Runtime init seconds (~↓)', t1 - t0),
                ('Runtime training seconds (↓)', t2 - t1),
                ('Runtime eval seconds (~↓)', t3 - t2),
            ]:
                eval_results.append(EvalResult(
                    metric_type='runtime',
                    metric_name=metric_name,
                    metric_value=seconds,
                    **metric_common_attributes,  # type: ignore
                ))

        save_model_card(
            str(self.hub_model_id),
            images=eval_images,
            base_model=self.pretrained_model_name_or_path,
            dataset_forget_name=self.erase_concept,
            dataset_retain_name='',
            repo_folder=self.output_dir,
            eval_results=eval_results,
            tags=[
                'stable-diffusion',
                'stable-diffusion-diffusers',
                'text-to-image',
                'diffusers',
                'diffusers-training',
            ],
            hyperparameters={k: v for k, v in self.model_dump().items() if isinstance(v, (str, float, int, type(None)))},
        )  # type: ignore[arg-type]

        if self.hub_model_id is not None:
            upload_folder(
                repo_id=self.hub_model_id,
                folder_path=self.output_dir,
                commit_message='End of training',
                ignore_patterns=['step_*', 'epoch_*'],
            )

        return eval_results

    def _load_pipeline(self) -> _DiffusionPipelineComponents:
        '''Load the base pipeline and freeze everything that is never trained.'''
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=self._weight_dtype(),
            safety_checker=None,
        ).to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)
        return cast(_DiffusionPipelineComponents, pipeline)

    def _fit(self, pipeline: _DiffusionPipelineComponents) -> Dict[str, torch.Tensor]:
        '''Run the optimization and return the trained tensors, keyed by their denoiser parameter name.

        Separated from `train()` so that one step can be driven directly, with a small stand-in
        pipeline, without loading a checkpoint or evaluating anything.
        '''
        unet = pipeline.unet
        parameter_names = self._select_parameter_names(unet)
        logger.info(f'ESD trains {len(parameter_names)} denoiser tensors ({self.train_method}).')

        # Two copies of the selected tensors over one module: the frozen originals build the target,
        # the trainable ones are optimized. Swapping them in place keeps a single denoiser resident,
        # which is what makes this fit on one card.
        named_parameters = dict(unet.named_parameters())
        frozen: Dict[str, torch.Tensor] = {}
        trainable: Dict[str, torch.nn.Parameter] = {}
        for parameter_name in parameter_names:
            original = named_parameters[parameter_name]
            frozen[parameter_name] = original.detach().clone()
            trainable[parameter_name] = torch.nn.Parameter(original.detach().clone(), requires_grad=True)

        conditioning = self._encode_conditioning(pipeline)
        optimizer = torch.optim.Adam(list(trainable.values()), lr=self.learning_rate)

        # The reference draws both the denoising depth and the sampling noise from Python's global
        # `random`, which nothing seeds, so its runs are not reproducible. This is the one deliberate
        # deviation: the same two draws come from a generator owned by this run.
        depth_sampler = random.Random(self.seed)

        total_steps = self._total_steps()
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)

            depth = depth_sampler.randint(0, self.num_inference_steps - 1)
            noise_seed = depth_sampler.randint(0, 2 ** 15)

            with self._parameters_applied(unet, frozen):
                unet.eval()
                with torch.no_grad():
                    latents, timestep = self._sample_intermediate_latent(
                        pipeline, conditioning, depth=depth, noise_seed=noise_seed,
                    )
                    target = self._negative_guidance_target(unet, conditioning, latents, timestep)

            with self._parameters_applied(unet, trainable):
                unet.train()
                prediction = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=conditioning['student'],
                    return_dict=False,
                )[0]
                loss = F.mse_loss(prediction.float(), target.float())
                loss.backward()
                optimizer.step()

            logger.info(f'ESD step {step + 1}/{total_steps}: loss {loss.item():.6f}, depth {depth}')

        return {name: parameter.detach().cpu().contiguous() for name, parameter in trainable.items()}

    @staticmethod
    @contextlib.contextmanager
    def _parameters_applied(unet: torch.nn.Module, tensors: Dict[str, Any]) -> Iterator[None]:
        '''Temporarily install *tensors* into the denoiser, by parameter name.

        Both the frozen and the trainable copies live over the same module, so which of them the
        forward pass sees is a property of this context and never of the surrounding code.
        '''
        previous: Dict[str, Any] = {}
        for parameter_name, tensor in tensors.items():
            module_path, _, attribute = parameter_name.rpartition('.')
            module = unet.get_submodule(module_path) if module_path else unet
            previous[parameter_name] = getattr(module, attribute)
            value = tensor if isinstance(tensor, torch.nn.Parameter) else torch.nn.Parameter(tensor, requires_grad=False)
            setattr(module, attribute, value)
        try:
            yield
        finally:
            for parameter_name, original in previous.items():
                module_path, _, attribute = parameter_name.rpartition('.')
                module = unet.get_submodule(module_path) if module_path else unet
                setattr(module, attribute, original)

    def _encode_conditioning(self, pipeline: _DiffusionPipelineComponents) -> Dict[str, torch.Tensor]:
        '''Encode the three prompts this method conditions on, once, before the loop.'''
        with torch.no_grad():
            erase_embeds, null_embeds = pipeline.encode_prompt(
                prompt=self.erase_concept,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt='',
            )
            if self.erase_from is not None:
                erase_from_embeds, _ = pipeline.encode_prompt(
                    prompt=self.erase_from,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt='',
                )
            else:
                # Anchored to itself: the concept is pushed away from its own prediction, which is the
                # published default. The two entries are the same tensor by design, not by accident.
                erase_from_embeds = erase_embeds

        return {
            'erase': erase_embeds.to(self.device),
            'null': null_embeds.to(self.device),
            'erase_from': erase_from_embeds.to(self.device),
            'student': erase_from_embeds.to(self.device),
        }

    def _sample_intermediate_latent(
        self,
        pipeline: _DiffusionPipelineComponents,
        conditioning: Dict[str, torch.Tensor],
        depth: int,
        noise_seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Denoise from pure noise for *depth* steps and return the latent and the next timestep.

        `depth = 0` returns the initial noise itself, paired with the first timestep of the schedule.
        The returned timestep is always the one that has *not* been applied yet, which is the noise
        level the erasure is fitted at on this step.
        '''
        scheduler = pipeline.scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        latent_channels = pipeline.unet.config.in_channels
        latent_size = self.resolution // pipeline.vae_scale_factor
        generator = torch.Generator(device=self.device).manual_seed(noise_seed) \
            if str(self.device).startswith('cuda') else torch.Generator().manual_seed(noise_seed)
        latents = torch.randn(
            (1, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self.device,
            dtype=self._weight_dtype(),
        ) * scheduler.init_noise_sigma

        prompt_embeds = torch.cat([conditioning['null'], conditioning['erase_from']])
        for timestep in timesteps[:depth]:
            latent_model_input = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise_uncond, noise_cond = pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0].chunk(2)
            noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents, timesteps[depth]

    def _negative_guidance_target(
        self,
        unet: torch.nn.Module,
        conditioning: Dict[str, torch.Tensor],
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        '''The prediction the trainable denoiser is fitted to, built from the frozen one.

        Reads as: start from what the model predicts for the anchor concept, then move away from the
        erased concept by `negative_guidance` times however much the concept changes the prediction.
        '''
        prediction_erase = unet(latents, timestep, encoder_hidden_states=conditioning['erase'], return_dict=False)[0]
        prediction_null = unet(latents, timestep, encoder_hidden_states=conditioning['null'], return_dict=False)[0]
        if conditioning['erase_from'] is conditioning['erase']:
            prediction_erase_from = prediction_erase
        else:
            prediction_erase_from = unet(
                latents, timestep, encoder_hidden_states=conditioning['erase_from'], return_dict=False,
            )[0]

        return prediction_erase_from - self.negative_guidance * (prediction_erase - prediction_null)

    ##########################################
    # Saving and loading
    ##########################################
    def _save_weights(self, tensors: Dict[str, torch.Tensor]) -> None:
        '''Save only the modified denoiser tensors, keyed by their name in the denoiser.'''
        os.makedirs(self.output_dir, exist_ok=True)
        save_file(
            tensors,
            os.path.join(self.output_dir, ESD_WEIGHTS_FILENAME),
            metadata={
                'base_model_id': self.pretrained_model_name_or_path,
                'train_method': self.train_method,
                'erase_concept': self.erase_concept,
                'erase_from': self.erase_from or '',
                'negative_guidance': str(self.negative_guidance),
                'guidance_scale': str(self.guidance_scale),
                'num_inference_steps': str(self.num_inference_steps),
                'num_train_epochs': str(self._total_steps()),
                'seed': str(self.seed),
            },
        )
        logger.info(f'Saved {len(tensors)} ESD weight tensors to {self.output_dir}')

    @staticmethod
    def get_pipeline_from_modified_weights(pretrained_model_name_or_path: str, device: str | torch.device, output_dir: str) -> DiffusionPipeline:
        '''Rebuild the unlearned pipeline: the base model with the saved tensors applied.

        Same signature as `UCE.get_pipeline_from_modified_weights`, because both methods produce the
        same kind of artifact and the benchmark loads them through one code path.
        '''
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

        esd_state_dict = load_file(os.path.join(output_dir, ESD_WEIGHTS_FILENAME))
        logger.debug(f'Loaded {len(esd_state_dict)} ESD weight tensors')

        applied = 0
        with torch.no_grad():
            for name, param in pipe.unet.named_parameters():  # type: ignore[union-attr]
                if name in esd_state_dict:
                    param.copy_(esd_state_dict[name])
                    applied += 1

        if applied != len(esd_state_dict):
            raise ValueError(
                f'{ESD_WEIGHTS_FILENAME} holds {len(esd_state_dict)} tensors but only {applied} matched a '
                'parameter of the denoiser. The checkpoint and the saved weights do not correspond.'
            )

        return pipe

    ##########################################
    # Evaluation
    ##########################################
    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        '''Score the unlearned model against the original one on the final evaluation prompts.

        Public and called by `train()`, mirroring `UCE`. Two of the records it returns carry the
        metric names the I-CARE benchmark's equalization procedure tunes on.
        '''
        assert type(self.final_eval_prompts_forget) == list  # noqa
        assert type(self.final_eval_prompts_retain) == list  # noqa

        pipeline_original = AutoPipelineForText2Image.from_pretrained(
            self.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None,
        ).to(self.device)
        pipeline_unlearned = self.__class__.get_pipeline_from_modified_weights(
            self.pretrained_model_name_or_path,
            self.device,
            self.output_dir,
        )

        evaluator = EvaluatorTextToImage(
            # Both are Stable Diffusion pipelines; only the loader's declared return type is the
            # permissive base, for the reason given at _load_pipeline.
            pipeline_original=cast(StableDiffusionPipeline, pipeline_original),
            pipeline_unlearned=cast(StableDiffusionPipeline, pipeline_unlearned),
            pipeline_learned=None,
            prompts_forget=self.final_eval_prompts_forget,
            prompts_retain=self.final_eval_prompts_retain,
            metric_clip=MetricImageTextSimilarity(metrics=['clip']),
            compute_runtimes=self.compute_runtimes,
        )

        eval_result, eval_images = evaluator.evaluate()

        return eval_result, eval_images
