import os
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from vision_unlearning.datasets.others import jsonl_dump
from vision_unlearning.unlearner.lora import unlearn_lora


@dataclass(frozen=True)
class GenerationRuntime:
    '''How the pipeline is built and how much memory it is allowed to use, as opposed to what image
    it is asked for.

    Every field defaults to what `generate_dataset` did before this class existed, so a caller that
    passes no runtime gets byte-identical behaviour. The defaults are the Stable Diffusion 1.x
    settings the existing artifacts were produced with; the non-defaults exist because Stable
    Diffusion XL does not fit on a 12 GB card without them.

    @param attention_slice_size: when set, the denoiser computes attention in slices of this size
        instead of all at once. 1 is the most aggressive and the slowest.
    @param vae_slicing: decode one image at a time rather than a whole batch.
    @param vae_tiling: decode the image in tiles. NOTE that enabling this alone may do nothing: the
        autoencoder only tiles when the latent is larger than its own `tile_latent_min_size`, which
        is derived from the checkpoint's `sample_size` -- 1024 for Stable Diffusion XL, giving a
        threshold of 128 latent units that a 768-pixel image (96 units) never crosses. Set
        `vae_tile_sample_min_size` to force it.
    @param vae_tile_sample_min_size: tile edge in pixels. The matching latent threshold is derived
        as one eighth of it, which is the autoencoder's own downsampling factor.
    @param deterministic_algorithms: the default, True, reproduces the historical behaviour --
        deterministic kernels are requested whenever a graphics card is present, which is what makes
        two runs of the same seed pixel-identical. It must be turned OFF for Stable Diffusion XL
        above 512 pixels on ROCm, where the deterministic convolution algorithm fails with
        `HIP error: unspecified launch failure`. Turning it off does not make generation random: the
        seed still determines the image, and the measured difference between two generations of the
        same seed in one process is 0.0003 of 255.
    @param device_map: passed to `from_pretrained`. "balanced" places each component on the graphics
        card as it is read instead of assembling the whole pipeline in system memory first, which on
        a 16 GB machine is the difference between a load that completes and one that does not. When
        set, the pipeline is NOT moved with `.to(device)` afterwards, because placement is then the
        loader's responsibility.
    '''
    attention_slice_size: Optional[int] = None
    vae_slicing: bool = False
    vae_tiling: bool = False
    vae_tile_sample_min_size: Optional[int] = None
    deterministic_algorithms: bool = True
    device_map: Optional[str] = None


def _apply_runtime(pipeline: Any, runtime: GenerationRuntime) -> None:
    '''Applies the memory settings that are configured on an already-built pipeline.

    Placement (`device_map`) is not here: it can only be chosen while the pipeline is being loaded.
    '''
    if runtime.attention_slice_size is not None:
        pipeline.enable_attention_slicing(runtime.attention_slice_size)
    if runtime.vae_slicing:
        pipeline.enable_vae_slicing()
    if runtime.vae_tiling:
        pipeline.enable_vae_tiling()
        if runtime.vae_tile_sample_min_size is not None:
            pipeline.vae.tile_sample_min_size = runtime.vae_tile_sample_min_size
            pipeline.vae.tile_latent_min_size = runtime.vae_tile_sample_min_size // 8


def generate_dataset(
    model_base_name: Optional[str],
    lora_name: Optional[str],
    prompts: List[str],
    output_path: str,
    filenames: Optional[List[str]] = None,
    batch_size: int = 4,
    device: Union[int, str, torch.device] = 'cuda',
    lora_requires_inversion: bool = False,
    model_pipeline: Optional[AutoPipelineForText2Image] = None,
    seeds: Optional[List[int]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    variant: Optional[str] = None,
    guidance_scale: Optional[float] = None,
    original_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Optional[Tuple[int, int]] = None,
    runtime: Optional[GenerationRuntime] = None,
) -> List[Dict[str, str]]:
    '''
    Generate images for the given prompts and save them to output_path.

    When seeds is provided (recommended for reproducibility):
      - For each seed, the function sets torch/numpy/random global state and passes a
        seeded torch.Generator to the pipeline call.  This guarantees that running with
        the same model weights and the same seed produces pixel-identical images.
      - filenames may optionally be provided.  When provided, the caller must supply
        exactly ``len(seeds) * len(prompts)`` filenames in seed-major order:
        ``[seed0_prompt0, seed0_prompt1, ..., seed1_prompt0, seed1_prompt1, ...]``.
      - When seeds is provided but filenames is None: filenames are auto-generated as
        ``{seed}_{prompt}.png`` (no prefix) for each (seed, prompt) pair.
      - metadata.jsonl is written once after all seeds are processed.

    When seeds is None (legacy mode):
      - filenames may be provided explicitly (one per prompt).
      - The pipeline is called once per batch without seeding — non-deterministic.
      - This path is kept for backward compatibility only.

    @param model_base_name: HF model name or local path.  Ignored if model_pipeline given.
    @param lora_name: LoRA adapter path.  If set, model_base_name is also required.
    @param prompts: Text prompts to generate images for.
    @param output_path: Directory where images and metadata.jsonl are saved.
    @param filenames: Explicit filenames (optional).
        - Legacy mode (seeds=None): one filename per prompt.
        - Seeded mode (seeds provided): len(seeds) * len(prompts) filenames in seed-major
          order.  If None, filenames are auto-generated as ``{seed}_{prompt}.png``.
    @param batch_size: Number of prompts per pipeline call.
    @param device: Torch device.
    @param lora_requires_inversion: Passed to unlearn_lora if lora_name is set.
    @param model_pipeline: Pre-loaded pipeline (skips loading if provided).
    @param seeds: List of integer seeds.  When provided the generation loop is seeded.
    @param height: Image height in pixels.  When None (the default) the argument is not
        passed to the pipeline at all, so the pipeline's own default applies: 512 for
        Stable Diffusion 1.x, 1024 for Stable Diffusion XL.  Both height and width must
        be given together.
    @param width: Image width in pixels.  See height.
    @param variant: Weight variant to load, e.g. "fp16".  When None (the default) the argument
        is not passed, so the checkpoint's full-precision weights are read and then cast to
        float16.  That is fine for Stable Diffusion 1.x (~4 GB) and impossible for Stable
        Diffusion XL on a 16 GB machine: its full-precision denoiser alone is 10.3 GB, and the
        read fills system memory before the cast can shrink it.
    @param guidance_scale: how strongly the sampler follows the prompt.  When None (the default) the
        argument is not passed and the pipeline's own default applies (7.5 for Stable Diffusion 1.x,
        5.0 for Stable Diffusion XL).
    @param original_size: Stable Diffusion XL only, and one of the arguments that decides whether a
        render is usable.  The model was conditioned during training on the size of each training
        image, so it associates a small original size with the kind of picture a small file usually
        is -- flat, low-detail, often clip art.  The pipeline defaults it to `(height, width)`, so
        asking for 512 pixels silently also asks for a 512-pixel-quality picture.  Passing
        (1024, 1024) declares the model's native size instead.  Ignored by Stable Diffusion 1.x
        pipelines, which have no such conditioning, so it must not be passed to them.
    @param target_size: Stable Diffusion XL only; the size the image is presented as, read together
        with original_size.  Same default and same reasoning.
    @param crops_coords_top_left: Stable Diffusion XL only; declares the crop offset the image
        should look like it was taken from.  (0, 0) means uncropped and centred.
    @param runtime: how the pipeline is built and how much memory it may use -- see
        `GenerationRuntime`.  When None (the default) every setting is the historical one, so a call
        that omits it behaves exactly as it did before this parameter existed.

    The four Stable Diffusion XL conditioning arguments and every field of `runtime` follow the same
    rule as height/width and variant: an argument that is not asked for is not passed at all, so a
    call that omits it reaches the pipeline with exactly the arguments it did before the parameter
    existed.  That is what keeps existing Stable Diffusion 1.4 artifacts comparable.
    '''
    # --- parameter validation ---
    if seeds is not None and filenames is not None:
        expected_count = len(seeds) * len(prompts)
        if len(filenames) != expected_count:
            raise ValueError(
                f"When seeds and filenames are both provided, filenames must have "
                f"len(seeds) * len(prompts) = {expected_count} entries in seed-major order "
                f"(seed0_prompt0, seed0_prompt1, ..., seed1_prompt0, ...); "
                f"got {len(filenames)}."
            )

    if (height is None) != (width is None):
        raise ValueError(
            f"height and width must be given together or not at all; got height={height}, width={width}."
        )

    # Kept empty unless the caller asked for an explicit resolution, so that every call which does
    # not pass one reaches the pipeline with exactly the arguments it did before this parameter existed.
    resolution_kwargs: Dict[str, Any] = {} if height is None else {"height": height, "width": width}
    # Same construction, same reason: a call that names no variant loads exactly what it always did.
    variant_kwargs: Dict[str, Any] = {} if variant is None else {"variant": variant}

    # The sampler arguments, built the same way and for the same reason. The three size arguments
    # exist only on Stable Diffusion XL pipelines; passing them to a Stable Diffusion 1.x pipeline
    # would be an error, which is why none of them has a default.
    conditioning_kwargs: Dict[str, Any] = {}
    if guidance_scale is not None:
        conditioning_kwargs["guidance_scale"] = guidance_scale
    if original_size is not None:
        conditioning_kwargs["original_size"] = original_size
    if target_size is not None:
        conditioning_kwargs["target_size"] = target_size
    if crops_coords_top_left is not None:
        conditioning_kwargs["crops_coords_top_left"] = crops_coords_top_left
    call_kwargs: Dict[str, Any] = {**resolution_kwargs, **conditioning_kwargs}

    runtime = runtime if runtime is not None else GenerationRuntime()
    device_map_kwargs: Dict[str, Any] = ({} if runtime.device_map is None
                                         else {"device_map": runtime.device_map})

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # --- load model (done once, shared across all seeds) ---
    # Typed loosely on purpose: the three branches below return three different pipeline
    # classes, and the diffusers stubs do not agree on which of them carries `.to`.
    pipeline: Any
    if lora_name:
        assert model_base_name is not None, "model_base_name must be provided if lora_name is used"
        _, _, pipeline = unlearn_lora(
            model_base_name,
            lora_name,
            device=str(device),
            weight_name="pytorch_lora_weights.safetensors",
            requires_inversion=lora_requires_inversion,
            return_original=False,
            return_learned=False,
            variant=variant,
            device_map=runtime.device_map,
        )
    elif model_base_name:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_base_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            **variant_kwargs,
            **device_map_kwargs,
        )
        # A pipeline loaded with a device map has already been placed by the loader, and moving it
        # afterwards is both unnecessary and unsupported.
        if runtime.device_map is None:
            pipeline = pipeline.to(device)
    elif model_pipeline:
        pipeline = model_pipeline if runtime.device_map is not None else model_pipeline.to(device)  # type: ignore
    else:
        raise ValueError("Either model_base_name or model_pipeline must be provided")

    _apply_runtime(pipeline, runtime)

    metadata: List[Dict[str, str]] = []

    if seeds is not None:
        # --- seeded generation mode ---

        # Enable deterministic CUDA ops for pixel-identical reproducibility.
        # CUBLAS_WORKSPACE_CONFIG must be set before the first CUBLAS call; setting
        # it here (before the first pipeline call in the loop) is sufficient when
        # the pipeline is freshly loaded.  On AMD ROCm this is the key flag that
        # eliminates non-deterministic GEMM/attention kernel selection.
        # runtime.deterministic_algorithms exists because this block is impossible on some
        # hardware: on ROCm, Stable Diffusion XL above 512 pixels fails inside a convolution with
        # `HIP error: unspecified launch failure` when deterministic kernels are requested.
        deterministic = torch.cuda.is_available() and runtime.deterministic_algorithms
        if deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            for seed_idx, seed in enumerate(seeds):
                # Seed all global RNG sources for full determinism.
                # SEEDING SITE, PAIRED: these five lines and the generator below are duplicated
                # deliberately in benchmarks/I_care/similarity.py::UnetLatentSimilarity._run_seed,
                # which captures the final denoised latent of these very images and verifies each
                # capture against the image written here. The two must move together: changing the
                # seeding convention on this side alone leaves that capture computing latents for
                # images that no longer exist. The matching comment is at the other site.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                generator = torch.Generator(device=device).manual_seed(seed)

                for start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[start:start + batch_size]
                    batch_outputs = pipeline(  # type: ignore
                        batch_prompts,
                        generator=generator,
                        **call_kwargs,
                    ).images

                    for i, image in enumerate(batch_outputs):
                        idx = start + i
                        prompt_text = prompts[idx]
                        # Filename: caller-supplied (seed-major order) or auto-generated.
                        if filenames is not None:
                            image_name = filenames[seed_idx * len(prompts) + idx]
                        else:
                            image_name = f"{seed}_{prompt_text}.png"
                        image.save(os.path.join(output_path, image_name), "PNG")
                        metadata.append({"file_name": image_name, "text": prompt_text})
        finally:
            # Restore deterministic algorithms setting to avoid affecting other code -- but only if
            # this call is what turned it on, so that a caller who deliberately runs without it is
            # not handed a different global state than it started with.
            if deterministic:
                torch.use_deterministic_algorithms(False)

    else:
        # --- legacy mode: no seeding, caller supplies filenames ---
        if filenames is not None:
            assert len(filenames) == len(prompts), \
                "filenames must have the same length as prompts"
            assert all(isinstance(fn, str) for fn in filenames), \
                "all filenames must be strings"
            assert all(fn.lower().endswith('.png') for fn in filenames), \
                "all filenames must end with .png"

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_outputs = pipeline(batch_prompts, **call_kwargs).images  # type: ignore

            for i, image in enumerate(batch_outputs):
                idx = start + i
                image_name = filenames[idx] if filenames is not None else f"{idx}.png"
                image_prompt = prompts[idx]

                image.save(os.path.join(output_path, image_name), "PNG")
                metadata.append({"file_name": image_name, "text": image_prompt})

    # Save metadata at the end
    jsonl_dump(metadata, os.path.join(output_path, "metadata.jsonl"))

    return metadata
