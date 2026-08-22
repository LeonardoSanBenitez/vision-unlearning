"""Generate a U-Care answer set with the UnlearnCanvas UCE sampling protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from tqdm import tqdm

from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.generated_dataset import GeneratedDataset


def answer_set_prompts() -> List[str]:
    """Return the complete 51 x 20 prompt grid in stable order."""
    return [
        cfg.answer_set_prompt(theme, object_class)
        for theme in cfg.STYLE_ENTITIES
        for object_class in cfg.OBJECT_ENTITIES
    ]


def _make_pipeline(model_path: str, device: str) -> StableDiffusionPipeline:
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _write_metadata(folder: Path, seeds: Iterable[int], prompts: Iterable[str]) -> None:
    metadata_path = folder / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            for prompt in prompts:
                handle.write(json.dumps({"seed": seed, "prompt": prompt}) + "\n")


def generate_answer_set(
    model_path: str,
    output_folder: str,
    seeds: List[int],
    device: str = "cuda",
    prompts: Optional[List[str]] = None,
    overwrite: bool = False,
) -> str:
    """Generate baseline images using one CPU-seeded latent per prompt and seed."""
    prompt_list = prompts if prompts is not None else answer_set_prompts()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GeneratedDataset(base_folder=str(output_dir.parent.parent))
    if not overwrite and dataset.exists(seeds, prompt_list):
        return str(output_dir)

    pipe = _make_pipeline(model_path, device)
    started = time.perf_counter()
    for seed in tqdm(seeds, desc="Seeds"):
        for prompt in tqdm(prompt_list, desc=f"Seed {seed}", leave=False):
            filename = f"off_{seed:02d}_{prompt}.png"
            image_path = output_dir / filename
            if image_path.exists() and not overwrite:
                continue

            cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, 64, 64),
                generator=cpu_generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=device, dtype=torch.float16)
            result = pipe(
                prompt=prompt,
                latents=latents,
                width=512,
                height=512,
                num_inference_steps=100,
                guidance_scale=9.0,
            )
            result.images[0].save(image_path)

    _write_metadata(output_dir, seeds, prompt_list)
    elapsed = time.perf_counter() - started
    count = len(seeds) * len(prompt_list)
    print(f"Generated {count} images in {elapsed:.2f}s ({elapsed / count:.3f}s/image).")
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-folder", default="assets/datasets/generated_baseline_sd_style50")
    parser.add_argument("--seed", type=int, nargs="+", default=[188])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_answer_set(
        model_path=args.model_path,
        output_folder=args.output_folder,
        seeds=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
