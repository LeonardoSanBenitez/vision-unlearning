from __future__ import annotations
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from PIL import Image

import torch
import torch.nn.functional as F
from pydantic import Field
from safetensors.torch import save_file, load_file
from datasets import load_dataset
from torchvision import transforms
from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoPipelineForText2Image, DDPMScheduler
from huggingface_hub.repocard_data import EvalResult
from huggingface_hub import upload_folder

from vision_unlearning.unlearner.base import Unlearner, logger
from vision_unlearning.evaluator import EvaluatorTextToImage
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.utils.model_management import save_model_card
from vision_unlearning.utils.training import preprocess_train, collate_fn
from vision_unlearning.utils import device as device_utils


SALUN_WEIGHTS_FILENAME = 'salun_sd_weights.safetensors'

#: The saliency mask, saved beside the weights rather than kept only in memory. It is a deterministic
#: function of the forget images and the seed, so a session can be re-run from it without repeating
#: the mask stage, and it is the artifact that makes the run auditable: the weights alone do not say
#: which parameters were allowed to move.
SALUN_MASK_FILENAME = 'salun_mask.safetensors'


class SalUn(Unlearner):
    '''
    Saliency Unlearning (SalUn): restricts an unlearning update to the weights the forget set is most
    salient for, by building a gradient-magnitude mask and applying it to every gradient before the
    optimizer steps.
    Adapted from:
        GitHub: https://github.com/OPTML-Group/Unlearn-Saliency
        Arxiv: https://arxiv.org/abs/2310.12508
        Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2024).
        SalUn: Empowering machine unlearning via gradient-based weight saliency in both image
        classification and generation. In International Conference on Learning Representations.

    This unlearner does not use LoRA. It updates the selected denoiser tensors in place and saves only
    those tensors, the same artifact shape `UCE` and `ESD` produce.

    The method has two stages.

    **The saliency mask.** One pass over the forget images against a negated classifier-free-guidance
    objective, ``loss = -MSE(noise, (1 + g) * F - g * N)``, where ``F`` is the denoiser conditioned on
    the image's caption and ``N`` the same denoiser on the empty caption at the same noised latent.
    The per-parameter gradients are summed over the pass and their absolute value taken, and the mask
    keeps the elements whose magnitude is in the top ``mask_threshold`` fraction of a **global** rank
    over every parameter of the denoiser. The rank is global, so restricting the computation to the
    trained subset would not give the same mask.

    **The masked fine-tune.** For each step, a forget batch and a retain batch. The forget term
    distils the model's prediction under the image's own caption onto its own prediction under a
    substitute caption -- the same noised latent, the same timestep, the substitute branch detached.
    The retain term is the ordinary diffusion loss on retain images with their own captions. The two
    are combined as ``forget_loss + alpha * retain_loss``, and every gradient is multiplied by the
    mask before the optimizer moves anything.

    Two deviations from the reference implementation, both deliberate and both measured:

    * The mask stage runs in ``mask_precision`` (bfloat16 by default) because full precision does not
      fit a 12 GB card at this model size. On the trained cross-attention parameters this changes
      about 4.5 % of the mask elements, which is equivalent to moving ``mask_threshold`` by roughly
      0.05 -- about half the distance between the two published implementations' own settings for it.
      Set ``mask_precision='fp32'`` where the memory is available.
    * The threshold is found by bisection on the float bit pattern rather than by sorting every
      parameter element, which needs about 17 GB for this model. The two constructions were verified
      to select identical masks, including where magnitudes tie at the boundary.
    '''

    # ==== Model ====
    pretrained_model_name_or_path: str = Field(
        default='CompVis/stable-diffusion-v1-4',
        description='Path to pretrained model or model identifier from huggingface.co/models.'
    )
    revision: Optional[str] = Field(None, description='Revision of the pretrained model identifier.')
    device: str = Field(default='cuda:0', description='Device to train on.')

    # ==== Data ====
    dataset_forget_name: str = Field(..., description='Name or path of the dataset to be forgotten.')
    dataset_retain_name: str = Field(..., description='Name or path of the dataset to be retained.')
    dataset_forget_config_name: Optional[str] = Field(None, description='Config of the forget dataset, None if it has only one.')
    dataset_retain_config_name: Optional[str] = Field(None, description='Config of the retain dataset, None if it has only one.')
    cache_dir: Optional[str] = Field(None, description='Where to cache downloaded datasets.')
    image_column: str = Field('image', description='The column of the dataset containing an image.')
    caption_column: str = Field('text', description='The column of the dataset containing a caption.')
    resolution: int = Field(512, description='Resolution the training images are resized and cropped to.')
    center_crop: bool = Field(False, description='Whether to center crop the input images instead of cropping at random.')
    random_flip: bool = Field(False, description='Whether to randomly flip images horizontally.')
    max_train_samples: Optional[int] = Field(None, description='Limit the number of forget examples, for debugging or a quicker run.')
    dataloader_num_workers: int = Field(
        default=0,
        description='Number of subprocesses for data loading. Keep at 0 on Windows: the worker '
                    'processes must pickle the transform, and a nested function cannot be pickled.'
    )

    # ==== Concepts ====
    overwriting_concept: str = Field(
        ...,
        description='The concept the forget images are pushed towards, as it would be written in a '
                    'prompt. The forget term distils the prediction under the image caption onto the '
                    'prediction under this concept, so it is what the model produces afterwards.'
    )

    # ==== Method ====
    train_method: str = Field(
        default='xattn',
        description="Which denoiser tensors are trained: 'xattn' for the cross-attention modules "
                    "only, 'full' for every parameter. The mask is always built over every parameter, "
                    "because its threshold is a global rank."
    )
    mask_threshold: float = Field(
        default=0.5,
        description='Fraction of the denoiser elements the saliency mask keeps, by global rank of '
                    'gradient magnitude. 1.0 disables the mask (every gradient passes).'
    )
    mask_precision: str = Field(
        default='bf16',
        description="Precision of the mask stage's forward and backward: 'fp32', 'bf16' or 'fp16'. "
                    "The gradient is accumulated in float32 regardless. bfloat16 exists because full "
                    "precision does not fit a 12 GB card at this model size; it perturbs the mask by "
                    "about as much as moving mask_threshold by 0.05."
    )
    mask_guidance: float = Field(
        default=7.5,
        description='Classifier-free guidance weight in the mask objective. The reference uses 7.5.'
    )
    alpha: float = Field(
        default=0.1,
        description='Weight of the retain term against the forget term. Higher keeps more of the '
                    'original model.'
    )

    # ==== Training ====
    per_device_train_batch_size: int = Field(default=1, description='Batch size per device.')
    num_train_epochs: int = Field(
        default=5,
        description='Number of passes over the forget set. This is the strength dial: more passes '
                    'erase more.'
    )
    max_train_steps: Optional[int] = Field(None, description='If set, stop after this many optimizer steps regardless of the epoch count.')
    learning_rate: float = Field(default=1e-5, description='The initial learning rate for Adam.')
    seed: int = Field(default=42, description='Seed for the mask stage, the sampled timesteps and the sampling noise.')

    # ==== Output and evaluation ====
    output_dir: str = Field(
        default='./salun-output',
        description='Where the modified tensors and the mask are written.'
    )
    final_eval_prompts_forget: str | List[str] = Field([], description='Prompts for final evaluation on the forget dataset.')
    final_eval_prompts_retain: str | List[str] = Field([], description='Prompts for final evaluation on the retain dataset.')
    compute_runtimes: bool = Field(True, description='Whether to report the stage runtimes as evaluation records.')
    hub_model_id: Optional[str] = Field(None, description='Repository name to sync with `output_dir`. None for not push.')

    ##########################################
    # Configuration
    ##########################################
    def _mask_dtype(self) -> torch.dtype:
        if self.mask_precision == 'fp16':
            return torch.float16
        if self.mask_precision == 'bf16':
            return torch.bfloat16
        if self.mask_precision == 'fp32':
            return torch.float32
        raise ValueError(f'Unsupported mask_precision: {self.mask_precision!r}')

    def _select_parameter_names(self, unet: torch.nn.Module) -> List[str]:
        '''Names of the denoiser parameters this train method trains, in module order.

        The reference selects by name substring over the parameters themselves
        (`random_label.py`), so this does the same rather than selecting by module class.
        '''
        if self.train_method == 'full':
            selected = [name for name, _ in unet.named_parameters()]
        elif self.train_method == 'xattn':
            selected = [name for name, _ in unet.named_parameters() if 'attn2' in name]
        else:
            raise ValueError(f'Unsupported train_method: {self.train_method!r}')

        if not selected:
            raise ValueError(
                f'No trainable parameters were selected for train_method={self.train_method!r}. '
                'The checkpoint is not shaped the way this method expects.'
            )
        return selected

    ##########################################
    # Data
    ##########################################
    def _image_transforms(self) -> Any:
        return transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _dataloaders(self, tokenizer: Any) -> Tuple[Any, Any]:
        '''The forget and retain loaders.

        The forget examples carry two tokenizations: their own caption, and the same caption with the
        overwriting concept substituted, which is the target the forget term distils onto. The retain
        examples carry only their own caption.
        '''
        forget = load_dataset(self.dataset_forget_name, self.dataset_forget_config_name, cache_dir=self.cache_dir)
        retain = load_dataset(self.dataset_retain_name, self.dataset_retain_config_name, cache_dir=self.cache_dir)

        columns = forget['train'].column_names
        for column in (self.image_column, self.caption_column):
            if column not in columns:
                raise ValueError(f"Column '{column}' is not one of: {', '.join(columns)}")

        image_transforms = self._image_transforms()
        if self.max_train_samples is not None:
            forget['train'] = forget['train'].shuffle(seed=self.seed).select(range(self.max_train_samples))

        forget_dataset = forget['train'].with_transform(
            lambda examples: preprocess_train(
                examples, tokenizer, self.caption_column, self.image_column, image_transforms,
                concept_overwrite=self.overwriting_concept,
            )
        )
        retain_dataset = retain['train'].with_transform(
            lambda examples: preprocess_train(
                examples, tokenizer, self.caption_column, self.image_column, image_transforms,
            )
        )
        logger.info(f'Number of training examples = {len(forget_dataset)} forget + {len(retain_dataset)} retain')

        forget_loader = torch.utils.data.DataLoader(
            forget_dataset, shuffle=True, collate_fn=collate_fn,
            batch_size=self.per_device_train_batch_size, num_workers=self.dataloader_num_workers,
        )
        retain_loader = torch.utils.data.DataLoader(
            retain_dataset, shuffle=True, collate_fn=collate_fn,
            batch_size=self.per_device_train_batch_size, num_workers=self.dataloader_num_workers,
        )
        return forget_loader, retain_loader

    @staticmethod
    def _cycle(loader: Any) -> Iterator[Any]:
        '''Iterate a loader forever, re-creating its iterator when it is exhausted.

        The retain set is traversed, not sampled with replacement: `next(iter(loader))` inside a step
        loop would return the same first batch at every step, which is a defect the port of this
        method in one published benchmark does have.
        '''
        while True:
            for batch in loader:
                yield batch

    ##########################################
    # The saliency mask
    ##########################################
    def _accumulate_saliency(self, pipeline: Any, forget_loader: Any) -> Dict[str, torch.Tensor]:
        '''``|sum of gradients|`` over the forget set, per parameter, in float32 on the processor.

        THE BACKWARD IS DECOMPOSED, AND THE DECOMPOSITION IS EXACT. The objective differentiates
        through two forward passes, so evaluating it directly holds two activation graphs alive at
        once, which does not fit a 12 GB card at this model size. Since

            r = noise - (1 + g) * F + g * N,   L = -mean(r ** 2)

        and ``r`` needs only the *values* of ``F`` and ``N``, the combined backward splits into two
        backwards with explicit output gradients, ``2(1+g)/M * r`` and ``-2g/M * r``, each freed
        before the next graph is built. Autograd accumulates both contributions into ``.grad`` exactly
        as one combined backward would. Verified against the undecomposed form on a small denoiser:
        the largest relative gradient difference was 2.6e-06 and the masks the two induce were
        identical, element for element.

        No optimizer is constructed and no step is taken, matching the reference, which builds one
        only to call ``zero_grad``.
        '''
        unet = pipeline.unet
        was_training = unet.training
        unet.eval()
        for parameter in unet.parameters():
            parameter.requires_grad_(True)

        accumulated = {
            name: torch.zeros(parameter.shape, dtype=torch.float32)
            for name, parameter in unet.named_parameters()
        }

        generator = torch.Generator(device='cpu').manual_seed(self.seed)
        # Built from the config the pipeline already carries, not fetched again: the training
        # scheduler is the same noise schedule as the pipeline's, and reloading it would make
        # this method require a network round trip to run at all.
        scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        batches = 0
        for batch in forget_loader:
            unet.zero_grad(set_to_none=True)
            latents, noise, timesteps, noisy = self._noised_latents(pipeline, batch, generator, scheduler)
            conditioned = self._encode(pipeline, batch['input_ids'])
            unconditioned = self._encode_empty(pipeline, batch['input_ids'].shape[0])

            with torch.no_grad():
                forget_value = unet(noisy, timesteps, encoder_hidden_states=conditioned).sample
                null_value = unet(noisy, timesteps, encoder_hidden_states=unconditioned).sample
                residual = (
                    noise.float()
                    - (1.0 + self.mask_guidance) * forget_value.float()
                    + self.mask_guidance * null_value.float()
                )
                elements = residual.numel()

            forget_out = unet(noisy, timesteps, encoder_hidden_states=conditioned).sample
            forget_out.backward(
                gradient=((2.0 * (1.0 + self.mask_guidance) / elements) * residual).to(forget_out.dtype)
            )
            del forget_out

            null_out = unet(noisy, timesteps, encoder_hidden_states=unconditioned).sample
            null_out.backward(gradient=((-2.0 * self.mask_guidance / elements) * residual).to(null_out.dtype))
            del null_out

            for name, parameter in unet.named_parameters():
                if parameter.grad is not None:
                    accumulated[name] += parameter.grad.detach().float().cpu()
            batches += 1
            device_utils.empty_cache()

        unet.zero_grad(set_to_none=True)
        if was_training:
            unet.train()
        if batches == 0:
            raise ValueError('The forget dataloader produced no batches, so no saliency was measured.')
        logger.info(f'Saliency accumulated over {batches} forget batches')

        for tensor in accumulated.values():
            tensor.abs_()
        return accumulated

    @staticmethod
    def _global_threshold(magnitudes: Dict[str, torch.Tensor], keep: int) -> Tuple[float, int]:
        '''The value of the ``keep``-th largest magnitude, by bisection on the float bit pattern.

        The reference concatenates every parameter element and sorts twice, which for this model is
        roughly 17 GB of processor memory. Non-negative float32 values order identically to their bit
        patterns read as int32, so bisecting the integer range finds the exact same cut in 32 counting
        scans, each of which iterates the tensors one at a time and never concatenates them.

        Returns the boundary value and the number of elements strictly above it.
        '''
        def count_above(value: float) -> int:
            return sum(int(torch.sum(tensor > value)) for tensor in magnitudes.values())

        low, high = 0, 0x7F7FFFFF  # 0.0 up to the largest finite float32
        while low < high:
            middle = (low + high) // 2
            candidate = float(torch.tensor([middle], dtype=torch.int32).view(torch.float32).item())
            if count_above(candidate) <= keep:
                high = middle
            else:
                low = middle + 1
        boundary = float(torch.tensor([low], dtype=torch.int32).view(torch.float32).item())
        return boundary, count_above(boundary)

    def _build_mask(self, magnitudes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''The binary mask: the top ``mask_threshold`` fraction of elements by global rank.

        Ties at the boundary are real -- on this model they run to six figures -- so the shortfall
        left after keeping everything strictly above the boundary is filled by the tied elements in
        the order the parameters are traversed, which is what makes this reproduce a stable rank cut
        element for element rather than merely in count.
        '''
        total = sum(int(tensor.numel()) for tensor in magnitudes.values())
        keep = int(total * self.mask_threshold)
        boundary, above = self._global_threshold(magnitudes, keep)
        remaining = keep - above

        masks: Dict[str, torch.Tensor] = {}
        for name, tensor in magnitudes.items():
            flat = (tensor > boundary).flatten()
            if remaining > 0:
                tied = torch.nonzero((tensor == boundary).flatten(), as_tuple=False).flatten()
                take = tied[:remaining]
                if take.numel() > 0:
                    flat[take] = True
                    remaining -= int(take.numel())
            masks[name] = flat.reshape(tensor.shape)

        kept = sum(int(torch.sum(mask)) for mask in masks.values())
        logger.info(
            f'Saliency mask keeps {kept} of {total} denoiser elements '
            f'({100.0 * kept / total:.2f} %) at threshold {self.mask_threshold}'
        )
        return masks

    ##########################################
    # Training
    ##########################################
    def _total_steps(self, forget_loader: Any) -> int:
        steps = self.num_train_epochs * len(forget_loader)
        return min(steps, self.max_train_steps) if self.max_train_steps is not None else steps

    def _encode(self, pipeline: Any, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return pipeline.text_encoder(input_ids.to(self.device))[0]

    def _encode_empty(self, pipeline: Any, batch_size: int) -> torch.Tensor:
        '''Embeddings of the empty caption, which is the unconditional branch of the mask objective.'''
        tokenizer = pipeline.tokenizer
        tokens = tokenizer(
            [''] * batch_size, padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt',
        ).input_ids
        return self._encode(pipeline, tokens)

    def _noised_latents(
        self, pipeline: Any, batch: Dict[str, torch.Tensor], generator: torch.Generator,
        scheduler: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Encode the batch's images and add noise at a sampled timestep.

        The timestep and the noise are drawn from this run's own generator so the run is reproducible
        independently of any global seeding a caller may or may not have done.
        '''
        with torch.no_grad():
            pixel_values = batch['pixel_values'].to(self.device, dtype=pipeline.vae.dtype)
            latents = pipeline.vae.encode(pixel_values).latent_dist.sample(generator=None)
            latents = latents * pipeline.vae.config.scaling_factor
            latents = latents.to(pipeline.unet.dtype)

            noise = torch.randn(latents.shape, generator=generator).to(latents.device, latents.dtype)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), generator=generator,
            ).long().to(latents.device)
            noisy = scheduler.add_noise(latents, noise, timesteps)
        return latents, noise, timesteps, noisy

    def _fit(self, pipeline: Any, mask: Dict[str, torch.Tensor], forget_loader: Any, retain_loader: Any) -> Dict[str, torch.Tensor]:
        '''The masked fine-tune. Returns the trained tensors, keyed by their name in the denoiser.'''
        unet = pipeline.unet
        trained_names = self._select_parameter_names(unet)
        trained = {name: unet.get_parameter(name) for name in trained_names}
        for parameter in unet.parameters():
            parameter.requires_grad_(False)
        for parameter in trained.values():
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(list(trained.values()), lr=self.learning_rate)
        # Built from the config the pipeline already carries, not fetched again: the training
        # scheduler is the same noise schedule as the pipeline's, and reloading it would make
        # this method require a network round trip to run at all.
        scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        generator = torch.Generator(device='cpu').manual_seed(self.seed)
        device_mask = {name: tensor.to(self.device) for name, tensor in mask.items() if name in trained}

        unet.train()
        total = self._total_steps(forget_loader)
        retain_batches = self._cycle(retain_loader)
        step = 0
        losses: List[float] = []
        while step < total:
            for forget_batch in forget_loader:
                if step >= total:
                    break
                optimizer.zero_grad(set_to_none=True)

                # THE TWO TERMS ARE BACKWARD-ED SEPARATELY, NOT SUMMED FIRST. Each needs its own
                # activation graph, and holding both alive at once does not fit this model on a 12 GB
                # card. Gradients accumulate into `.grad`, so the result is the gradient of the sum.
                retain_batch = next(retain_batches)
                retain_loss = self._retain_loss(pipeline, retain_batch, generator, scheduler)
                (self.alpha * retain_loss).backward()
                retain_value = float(retain_loss.detach())
                del retain_loss

                forget_loss = self._forget_loss(pipeline, forget_batch, generator, scheduler)
                forget_loss.backward()
                forget_value = float(forget_loss.detach())
                del forget_loss

                for name, parameter in trained.items():
                    if parameter.grad is not None:
                        parameter.grad.mul_(device_mask[name])

                optimizer.step()
                losses.append(forget_value + self.alpha * retain_value)
                step += 1
                if step % 10 == 0 or step == total:
                    logger.info(f'step {step}/{total} forget {forget_value:.6f} retain {retain_value:.6f}')

        unet.eval()
        return {name: parameter.detach().cpu().contiguous() for name, parameter in trained.items()}

    def _retain_loss(self, pipeline: Any, batch: Dict[str, torch.Tensor], generator: torch.Generator, scheduler: Any) -> torch.Tensor:
        '''The ordinary diffusion loss on a retain batch under its own captions.'''
        _, noise, timesteps, noisy = self._noised_latents(pipeline, batch, generator, scheduler)
        conditioned = self._encode(pipeline, batch['input_ids'])
        predicted = pipeline.unet(noisy, timesteps, encoder_hidden_states=conditioned).sample
        return F.mse_loss(predicted.float(), noise.float())

    def _forget_loss(self, pipeline: Any, batch: Dict[str, torch.Tensor], generator: torch.Generator, scheduler: Any) -> torch.Tensor:
        '''Distil the prediction under the image's caption onto the prediction under the substitute.

        The same noised latent and the same timestep feed both branches, and the substitute branch is
        detached: it is a target, not something to be pushed around.
        '''
        _, _, timesteps, noisy = self._noised_latents(pipeline, batch, generator, scheduler)
        conditioned = self._encode(pipeline, batch['input_ids'])
        substitute = self._encode(pipeline, batch['forget_ids'])

        with torch.no_grad():
            target = pipeline.unet(noisy, timesteps, encoder_hidden_states=substitute).sample.detach()
        predicted = pipeline.unet(noisy, timesteps, encoder_hidden_states=conditioned).sample
        return F.mse_loss(predicted.float(), target.float())

    def train(self) -> List[EvalResult]:
        '''Build the mask, fine-tune under it, save, evaluate, and return the evaluation records.

        The orchestration is the same as every other unlearner in this library: load, fit, save,
        evaluate. The steps are separate methods so that a test can drive one of them.
        '''
        t0 = time.time()
        assert self.pretrained_model_name_or_path, 'Pretrained model path must not be empty.'
        assert self.overwriting_concept, 'overwriting_concept must not be empty.'
        assert 0.0 < self.mask_threshold <= 1.0, 'mask_threshold must be in (0, 1].'
        assert self.alpha >= 0.0, 'alpha must not be negative.'
        if isinstance(self.final_eval_prompts_retain, str):
            raise NotImplementedError('final_eval_prompts_retain should be a list of prompts, not a string.')
        if isinstance(self.final_eval_prompts_forget, str):
            raise NotImplementedError('final_eval_prompts_forget should be a list of prompts, not a string.')

        os.makedirs(self.output_dir, exist_ok=True)

        # ==== The mask stage, at its own precision ====
        mask_pipeline = self._load_pipeline(self._mask_dtype())
        forget_loader, retain_loader = self._dataloaders(mask_pipeline.tokenizer)
        t1 = time.time()
        magnitudes = self._accumulate_saliency(mask_pipeline, forget_loader)
        mask = self._build_mask(magnitudes)
        del magnitudes, mask_pipeline
        device_utils.empty_cache()
        t2 = time.time()

        # ==== The fine-tune, always in full precision ====
        pipeline = self._load_pipeline(torch.float32)
        trained_tensors = self._fit(pipeline, mask, forget_loader, retain_loader)
        t3 = time.time()

        self._save_weights(trained_tensors)
        self._save_mask(mask)
        del pipeline, mask
        device_utils.empty_cache()

        eval_results, eval_images = self.evaluate()
        t4 = time.time()

        metric_common_attributes = {
            'task_type': 'text-to-image',
            'dataset_type': 'forget-and-retain-together',
            'dataset_name': f'{self.dataset_forget_name} (forget)',
        }
        if self.compute_runtimes:
            for metric_name, seconds in [
                ('Runtime init seconds (~↓)', t1 - t0),
                ('Runtime mask seconds (↓)', t2 - t1),
                ('Runtime training seconds (↓)', t3 - t2),
                ('Runtime eval seconds (~↓)', t4 - t3),
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
            dataset_forget_name=self.dataset_forget_name,
            dataset_retain_name=self.dataset_retain_name,
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

    def _load_pipeline(self, dtype: torch.dtype) -> Any:
        '''Load the base pipeline at a given precision and freeze everything that is never trained.'''
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        # The autoencoder overflows the float16 range and returns not-a-number latents without
        # raising, so it stays in float32 whatever the rest of the pipeline runs in.
        pipeline.vae.to(torch.float32)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)
        return pipeline

    ##########################################
    # Artifacts
    ##########################################
    def _save_weights(self, tensors: Dict[str, torch.Tensor]) -> None:
        '''Save only the modified denoiser tensors, keyed by their name in the denoiser.'''
        os.makedirs(self.output_dir, exist_ok=True)
        save_file(
            tensors,
            os.path.join(self.output_dir, SALUN_WEIGHTS_FILENAME),
            metadata={
                'base_model_id': self.pretrained_model_name_or_path,
                'train_method': self.train_method,
                'overwriting_concept': self.overwriting_concept,
                'mask_threshold': str(self.mask_threshold),
                'mask_precision': self.mask_precision,
                'mask_guidance': str(self.mask_guidance),
                'alpha': str(self.alpha),
                'num_train_epochs': str(self.num_train_epochs),
                'learning_rate': str(self.learning_rate),
                'seed': str(self.seed),
            },
        )
        logger.info(f'Saved {len(tensors)} SalUn weight tensors to {self.output_dir}')

    def _save_mask(self, mask: Dict[str, torch.Tensor]) -> None:
        '''Save the saliency mask beside the weights.

        Stored as uint8 rather than bool because the safetensors format has no boolean dtype.
        '''
        save_file(
            {name: tensor.to(torch.uint8).contiguous() for name, tensor in mask.items()},
            os.path.join(self.output_dir, SALUN_MASK_FILENAME),
            metadata={
                'mask_threshold': str(self.mask_threshold),
                'mask_precision': self.mask_precision,
                'seed': str(self.seed),
            },
        )
        logger.info(f'Saved the saliency mask ({len(mask)} tensors) to {self.output_dir}')

    @staticmethod
    def get_pipeline_from_modified_weights(pretrained_model_name_or_path: str, device: str | torch.device, output_dir: str) -> DiffusionPipeline:
        '''Rebuild the unlearned pipeline: the base model with the saved tensors applied.

        Same signature as `UCE.get_pipeline_from_modified_weights` and `ESD`'s, because all three
        methods produce the same kind of artifact and the benchmark loads them through one code path.
        '''
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

        salun_state_dict = load_file(os.path.join(output_dir, SALUN_WEIGHTS_FILENAME))
        logger.debug(f'Loaded {len(salun_state_dict)} SalUn weight tensors')

        applied = 0
        with torch.no_grad():
            for name, param in pipe.unet.named_parameters():  # type: ignore[union-attr]
                if name in salun_state_dict:
                    param.copy_(salun_state_dict[name])
                    applied += 1

        if applied != len(salun_state_dict):
            raise ValueError(
                f'{SALUN_WEIGHTS_FILENAME} holds {len(salun_state_dict)} tensors but only {applied} matched a '
                'parameter of the denoiser. The checkpoint and the saved weights do not correspond.'
            )

        return pipe

    ##########################################
    # Evaluation
    ##########################################
    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        '''Score the unlearned model against the original one on the final evaluation prompts.

        Public and called by `train()`, mirroring `UCE` and `ESD`. Two of the records it returns carry
        the metric names the I-CARE benchmark's equalization procedure tunes on.
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
            pipeline_original=cast(StableDiffusionPipeline, pipeline_original),
            pipeline_unlearned=cast(StableDiffusionPipeline, pipeline_unlearned),
            pipeline_learned=None,
            prompts_forget=self.final_eval_prompts_forget,
            prompts_retain=self.final_eval_prompts_retain,
            metric_clip=MetricImageTextSimilarity(metrics=['clip']),
            compute_runtimes=self.compute_runtimes,
            plot_show=self.plot_show,
        )

        eval_result, eval_images = evaluator.evaluate()

        return eval_result, eval_images
