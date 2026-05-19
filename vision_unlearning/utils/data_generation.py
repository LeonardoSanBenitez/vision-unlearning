import os
import random
from typing import List, Dict, Optional, Union, Literal
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from vision_unlearning.datasets.others import jsonl_dump
from vision_unlearning.unlearner.lora import unlearn_lora


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
    lora_state: Optional[Literal['on', 'off']] = None,
) -> List[Dict[str, str]]:
    '''
    Generate images for the given prompts and save them to output_path.

    When seeds is provided (recommended for reproducibility):
      - lora_state must also be provided (used in the auto-generated filename prefix).
      - filenames must be None (mutually exclusive with seeds).
      - For each seed, the function sets torch/numpy/random global state and passes a
        seeded torch.Generator to the pipeline call.  This guarantees that running with
        the same model weights and the same seed produces pixel-identical images.
      - Filenames are auto-generated as ``{lora_state}_{seed:02d}_{prompt}.png`` for
        each (seed, prompt) pair — matching the convention in
        vision_unlearning.datasets.testbed.get_generated_dataset_file().
      - metadata.jsonl is written once after all seeds are processed.

    When seeds is None (legacy mode):
      - filenames may be provided explicitly (one per prompt).
      - The pipeline is called once per batch without seeding — non-deterministic.
      - This path is kept for backward compatibility only.

    @param model_base_name: HF model name or local path.  Ignored if model_pipeline given.
    @param lora_name: LoRA adapter path.  If set, model_base_name is also required.
    @param prompts: Text prompts to generate images for.
    @param output_path: Directory where images and metadata.jsonl are saved.
    @param filenames: Explicit filenames (legacy mode, mutually exclusive with seeds).
    @param batch_size: Number of prompts per pipeline call.
    @param device: Torch device.
    @param lora_requires_inversion: Passed to unlearn_lora if lora_name is set.
    @param model_pipeline: Pre-loaded pipeline (skips loading if provided).
    @param seeds: List of integer seeds.  When provided the generation loop is seeded.
    @param lora_state: 'on' or 'off' — required when seeds is provided, used in filenames.
    '''
    # --- parameter validation ---
    if seeds is not None and filenames is not None:
        raise ValueError("seeds and filenames are mutually exclusive; "
                         "when using seeds, filenames are auto-generated.")
    if seeds is not None and lora_state is None:
        raise ValueError("lora_state ('on' or 'off') is required when seeds is provided "
                         "so that filenames can be auto-generated.")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # --- load model (done once, shared across all seeds) ---
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
        )
    elif model_base_name:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_base_name,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
    elif model_pipeline:
        pipeline = model_pipeline.to(device)  # type: ignore
    else:
        raise ValueError("Either model_base_name or model_pipeline must be provided")

    metadata: List[Dict[str, str]] = []

    if seeds is not None:
        # --- seeded generation mode ---
        # lora_state is guaranteed non-None by the check above.
        ls: Literal['on', 'off'] = lora_state  # type: ignore[assignment]

        # Enable deterministic CUDA ops for pixel-identical reproducibility.
        # CUBLAS_WORKSPACE_CONFIG must be set before the first CUBLAS call; setting
        # it here (before the first pipeline call in the loop) is sufficient when
        # the pipeline is freshly loaded.  On AMD ROCm this is the key flag that
        # eliminates non-deterministic GEMM/attention kernel selection.
        if torch.cuda.is_available():
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            for seed in seeds:
                # Seed all global RNG sources for full determinism.
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
                    ).images

                    for i, image in enumerate(batch_outputs):
                        idx = start + i
                        prompt_text = prompts[idx]
                        # Filename matches get_generated_dataset_file() convention:
                        # {lora_state}_{seed:02d}_{prompt}.png
                        image_name = f"{ls}_{seed:02d}_{prompt_text}.png"
                        image.save(os.path.join(output_path, image_name), "PNG")
                        metadata.append({"file_name": image_name, "text": prompt_text})
        finally:
            # Restore deterministic algorithms setting to avoid affecting other code.
            if torch.cuda.is_available():
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
            batch_outputs = pipeline(batch_prompts).images  # type: ignore

            for i, image in enumerate(batch_outputs):
                idx = start + i
                image_name = filenames[idx] if filenames is not None else f"{idx}.png"
                image_prompt = prompts[idx]

                image.save(os.path.join(output_path, image_name), "PNG")
                metadata.append({"file_name": image_name, "text": image_prompt})

    # Save metadata at the end
    jsonl_dump(metadata, os.path.join(output_path, "metadata.jsonl"))

    return metadata
