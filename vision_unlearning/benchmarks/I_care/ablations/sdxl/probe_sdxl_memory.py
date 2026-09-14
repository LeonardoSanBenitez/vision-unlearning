"""Feasibility spike: can SPARE unlearning of SDXL run on this machine at all?

The question this answers is a go/no-go one, and it is answered with measurements rather than
arithmetic on parameter counts: SDXL's denoiser is 2.6 B parameters against Stable Diffusion 1.4's
0.86 B, its text conditioning comes from two encoders instead of one, and the SPARE training step
runs the denoiser four times (with and without the adapter, on the forget and on the retain batch)
before a single backward pass. Whether that fits in 12 GB of video memory, and how long one step
takes, decides the shape of the whole experiment.

Four stages, each writing one JSON into ``assets/`` so a later stage never has to be re-run to
recover an earlier number:

``env``       machine inventory: video memory, system memory, library versions. No download.
``load``      loads the two text encoders, the fp16-fix variational autoencoder and the denoiser,
              adds the LoRA adapter, and reports resident video memory. Downloads ~7 GB on the
              first run.
``step``      the real measurement: one full SPARE training step at a given resolution and batch
              size -- four denoiser forward passes, the combined loss, the backward pass and the
              optimizer step -- repeated a few times, reporting peak video memory and the time per
              step separated into the first (allocator warm-up) and the steady-state ones.
``generate``  one image from the SDXL pipeline at a given resolution, for the generation half of
              the budget; the campaign generates hundreds of these.

Run from this directory with PYTHONPATH at the repository root::

    PY=".../sd-interpretability/.venv/Scripts/python.exe"
    HF_HUB_DISABLE_XET=1 "$PY" probe_sdxl_memory.py --stage env
    HF_HUB_DISABLE_XET=1 "$PY" probe_sdxl_memory.py --stage load
    HF_HUB_DISABLE_XET=1 "$PY" probe_sdxl_memory.py --stage step --resolution 512 --batch-size 1
    HF_HUB_DISABLE_XET=1 "$PY" probe_sdxl_memory.py --stage generate --resolution 1024

``HF_HUB_DISABLE_XET=1`` is mandatory for anything that really talks to HuggingFace in this
environment: the xet transfer backend hangs indefinitely on some downloads here.

This script deliberately does not import ``vision_unlearning``: the interpreter that has torch and
diffusers does not have the package installed, and the measurement does not need it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_OUT = Path(__file__).resolve().parent / "assets"
_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"


def _gb(value: float) -> float:
    return round(value / 1024 ** 3, 3)


def _memory_snapshot() -> Dict[str, Any]:
    """Video and system memory, both as free/total, at the moment of the call."""
    import psutil
    import torch

    snapshot: Dict[str, Any] = {}
    virtual = psutil.virtual_memory()
    snapshot["system_memory_total_gb"] = _gb(virtual.total)
    snapshot["system_memory_available_gb"] = _gb(virtual.available)
    if torch.cuda.is_available():
        free_video, total_video = torch.cuda.mem_get_info(0)
        snapshot["video_memory_total_gb"] = _gb(total_video)
        snapshot["video_memory_free_gb"] = _gb(free_video)
        snapshot["video_memory_torch_allocated_gb"] = _gb(torch.cuda.memory_allocated())
        snapshot["video_memory_torch_reserved_gb"] = _gb(torch.cuda.memory_reserved())
    return snapshot


def _write(name: str, payload: Dict[str, Any]) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"WROTE {path}")
    return path


def stage_env(_: argparse.Namespace) -> Dict[str, Any]:
    import torch

    payload: Dict[str, Any] = {
        "stage": "env",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload["device_name"] = properties.name
        payload["device_arch"] = getattr(properties, "gcnArchName", None)
        payload["device_total_memory_gb"] = _gb(properties.total_memory)
    for module_name in ("diffusers", "transformers", "peft", "accelerate", "safetensors", "datasets"):
        try:
            module = __import__(module_name)
            payload[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            payload[f"{module_name}_version"] = None
    payload.update(_memory_snapshot())
    return payload


def _load_models(
    resolution: int,
    weight_dtype: Any,
    lora_rank: int,
    lora_alpha: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load every SDXL component the SPARE step needs, adapter attached, and report memory.

    Mirrors the component set and the dtype policy of the SDXL LoRA training example: the denoiser,
    the two text encoders and the variational autoencoder are frozen and live in ``weight_dtype``;
    the fp16-fix autoencoder is used so that a half-precision autoencoder does not produce
    not-a-number latents. Only the adapter parameters are trainable, and they are cast back to
    float32.
    """
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    timings: Dict[str, Any] = {}
    device = "cuda"

    t0 = time.time()
    tokenizer_one = AutoTokenizer.from_pretrained(_MODEL_ID, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(_MODEL_ID, subfolder="tokenizer_2", use_fast=False)
    scheduler = DDPMScheduler.from_pretrained(_MODEL_ID, subfolder="scheduler")
    timings["load_tokenizers_and_scheduler_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    text_encoder_one = CLIPTextModel.from_pretrained(_MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(_MODEL_ID, subfolder="text_encoder_2", variant="fp16", torch_dtype=weight_dtype)
    timings["load_text_encoders_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(_VAE_ID, torch_dtype=weight_dtype)
    timings["load_vae_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    unet = UNet2DConditionModel.from_pretrained(_MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=weight_dtype)
    timings["load_unet_s"] = round(time.time() - t0, 1)

    for model in (unet, vae, text_encoder_one, text_encoder_two):
        model.requires_grad_(False)

    parameter_counts = {
        "unet_parameters": sum(p.numel() for p in unet.parameters()),
        "vae_parameters": sum(p.numel() for p in vae.parameters()),
        "text_encoder_one_parameters": sum(p.numel() for p in text_encoder_one.parameters()),
        "text_encoder_two_parameters": sum(p.numel() for p in text_encoder_two.parameters()),
    }

    t0 = time.time()
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    torch.cuda.synchronize()
    timings["move_to_device_s"] = round(time.time() - t0, 1)

    unet.add_adapter(LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    if weight_dtype != torch.float32:
        cast_training_params([unet], dtype=torch.float32)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    parameter_counts["adapter_trainable_parameters"] = sum(p.numel() for p in trainable)
    parameter_counts["adapter_trainable_tensors"] = len(trainable)

    bundle = {
        "unet": unet,
        "vae": vae,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "scheduler": scheduler,
        "trainable": trainable,
        "device": device,
        "weight_dtype": weight_dtype,
        "resolution": resolution,
    }
    report = {"timings": timings, "parameter_counts": parameter_counts, "after_load": _memory_snapshot()}
    return bundle, report


def stage_load(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    dtype = getattr(torch, args.dtype)
    before = _memory_snapshot()
    _, report = _load_models(args.resolution, dtype, args.lora_rank, args.lora_alpha)
    return {
        "stage": "load",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": args.dtype,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "before_load": before,
        **report,
    }


def _encode_prompt(text_encoders: List[Any], token_ids: List[Any]) -> Tuple[Any, Any]:
    """SDXL prompt conditioning: penultimate hidden states of both encoders, concatenated.

    The pooled projection of the second encoder is a separate output and is passed to the denoiser
    as micro-conditioning, not as the cross-attention sequence. This is the same computation the
    SDXL training example performs.
    """
    import torch

    embeds_list = []
    pooled = None
    for encoder, ids in zip(text_encoders, token_ids):
        output = encoder(ids.to(encoder.device), output_hidden_states=True, return_dict=False)
        pooled = output[0]
        embeds = output[-1][-2]
        embeds_list.append(embeds)
    prompt_embeds = torch.concat(embeds_list, dim=-1)
    assert pooled is not None
    return prompt_embeds, pooled.view(prompt_embeds.shape[0], -1)


def stage_step(args: argparse.Namespace) -> Dict[str, Any]:
    """One full SPARE training step, repeated, with peak video memory and time per step.

    The step is the one the method actually performs: encode both image batches with the
    autoencoder, sample noise and a timestep for each, run the denoiser four times (adapter on and
    adapter off, for the forget batch under the replacer caption and for the retain batch under its
    own caption), combine the two mean-squared-error losses and take one optimizer step.
    """
    import torch
    import torch.nn.functional as F

    dtype = getattr(torch, args.dtype)
    before = _memory_snapshot()
    bundle, load_report = _load_models(args.resolution, dtype, args.lora_rank, args.lora_alpha)

    unet = bundle["unet"]
    vae = bundle["vae"]
    scheduler = bundle["scheduler"]
    device = bundle["device"]
    resolution = args.resolution
    batch_size = args.batch_size
    encoders = [bundle["text_encoder_one"], bundle["text_encoder_two"]]

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.vae_tiling:
        # The 1024 failures of the first pass were both inside the autoencoder -- the encoder on the
        # training side, the decoder on the generation side -- so tiling it is the first lever, not
        # gradient checkpointing.
        vae.enable_tiling()
        vae.enable_slicing()

    optimizer = torch.optim.AdamW(bundle["trainable"], lr=1e-4)

    def tokenize(text: str) -> List[Any]:
        ids = []
        for tokenizer in (bundle["tokenizer_one"], bundle["tokenizer_two"]):
            ids.append(tokenizer(
                [text] * batch_size,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids)
        return ids

    ids_forget = tokenize("An image of Mark Philippoussis")
    ids_override = tokenize("An image of a child")
    ids_retain = tokenize("An image of Tim Henman")

    # Synthetic pixel batches: the measurement is of shapes and dtypes, not of image content.
    pixels_forget = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)
    pixels_retain = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)

    def time_ids() -> Any:
        row = [resolution, resolution, 0, 0, resolution, resolution]
        return torch.tensor([row] * batch_size, device=device, dtype=dtype)

    added_time_ids = time_ids()

    # With --precompute, the autoencoder and both text encoders are used once and then evicted from
    # the device: every caption in this experiment is fixed, and the training images are square, so
    # their latents do not vary between steps either. Only the denoiser stays resident.
    precomputed: Dict[str, Any] = {}
    if args.precompute:
        with torch.no_grad():
            precomputed["latents_forget"] = vae.encode(pixels_forget).latent_dist.sample() * vae.config.scaling_factor
            precomputed["latents_retain"] = vae.encode(pixels_retain).latent_dist.sample() * vae.config.scaling_factor
            precomputed["forget"] = _encode_prompt(encoders, ids_forget)
            precomputed["override"] = _encode_prompt(encoders, ids_override)
            precomputed["retain"] = _encode_prompt(encoders, ids_retain)
        vae.to("cpu")
        for encoder in encoders:
            encoder.to("cpu")
        torch.cuda.empty_cache()
        print("precomputed latents and text embeddings; autoencoder and text encoders evicted", flush=True)

    def one_step() -> Tuple[float, float]:
        if args.precompute:
            latents_forget = precomputed["latents_forget"]
            latents_retain = precomputed["latents_retain"]
            embeds_forget, pooled_forget = precomputed["forget"]
            embeds_override, pooled_override = precomputed["override"]
            embeds_retain, pooled_retain = precomputed["retain"]
        else:
            with torch.no_grad():
                latents_forget = vae.encode(pixels_forget).latent_dist.sample() * vae.config.scaling_factor
                latents_retain = vae.encode(pixels_retain).latent_dist.sample() * vae.config.scaling_factor
                embeds_forget, pooled_forget = _encode_prompt(encoders, ids_forget)
                embeds_override, pooled_override = _encode_prompt(encoders, ids_override)
                embeds_retain, pooled_retain = _encode_prompt(encoders, ids_retain)

        noise_forget = torch.randn_like(latents_forget)
        noise_retain = torch.randn_like(latents_retain)
        steps_forget = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        steps_retain = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_forget = scheduler.add_noise(latents_forget, noise_forget, steps_forget)
        noisy_retain = scheduler.add_noise(latents_retain, noise_retain, steps_retain)

        conditions_forget = {"time_ids": added_time_ids, "text_embeds": pooled_forget}
        conditions_override = {"time_ids": added_time_ids, "text_embeds": pooled_override}
        conditions_retain = {"time_ids": added_time_ids, "text_embeds": pooled_retain}

        def forget_half() -> Any:
            prediction_new = unet(noisy_forget, steps_forget, embeds_forget, added_cond_kwargs=conditions_forget, return_dict=False)[0]
            unet.disable_adapters()
            with torch.no_grad():
                prediction_old = unet(noisy_forget, steps_forget, embeds_override, added_cond_kwargs=conditions_override, return_dict=False)[0]
            unet.enable_adapters()
            return F.mse_loss(prediction_new.float(), prediction_old.float(), reduction="mean")

        def retain_half() -> Any:
            prediction_new = unet(noisy_retain, steps_retain, embeds_retain, added_cond_kwargs=conditions_retain, return_dict=False)[0]
            unet.disable_adapters()
            with torch.no_grad():
                prediction_old = unet(noisy_retain, steps_retain, embeds_retain, added_cond_kwargs=conditions_retain, return_dict=False)[0]
            unet.enable_adapters()
            return F.mse_loss(prediction_new.float(), prediction_old.float(), reduction="mean")

        if args.split_backward:
            # The gradient of a weighted sum is the weighted sum of the gradients, and gradients
            # accumulate additively into .grad, so backward on each half in turn is the same update
            # as backward on the combined loss -- while only one half's activation graph is alive at
            # a time.
            loss_forget = forget_half()
            (args.forget_weight * loss_forget).backward()
            loss_retain = retain_half()
            (args.retain_weight * loss_retain).backward()
        else:
            loss_forget = forget_half()
            loss_retain = retain_half()
            loss = args.forget_weight * loss_forget + args.retain_weight * loss_retain
            loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle["trainable"], 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return float(loss_forget.detach()), float(loss_retain.detach())

    torch.cuda.reset_peak_memory_stats()
    per_step_seconds: List[float] = []
    losses: List[Tuple[float, float]] = []
    for index in range(args.steps):
        t0 = time.time()
        losses.append(one_step())
        per_step_seconds.append(round(time.time() - t0, 2))
        print(f"step {index}: {per_step_seconds[-1]}s  loss_forget={losses[-1][0]:.3e}  loss_retain={losses[-1][1]:.3e}", flush=True)

    peak = {
        "peak_torch_allocated_gb": _gb(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_gb": _gb(torch.cuda.max_memory_reserved()),
    }
    steady = per_step_seconds[1:] or per_step_seconds
    return {
        "stage": "step",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": args.dtype,
        "resolution": resolution,
        "batch_size": batch_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "vae_tiling": args.vae_tiling,
        "split_backward": args.split_backward,
        "precompute": args.precompute,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "steps": args.steps,
        "before_load": before,
        "load": load_report,
        "first_step_seconds": per_step_seconds[0],
        "steady_state_seconds_per_step": round(sum(steady) / len(steady), 2),
        "per_step_seconds": per_step_seconds,
        "loss_forget_first_and_last": [losses[0][0], losses[-1][0]],
        "loss_retain_first_and_last": [losses[0][1], losses[-1][1]],
        **peak,
        "after_steps": _memory_snapshot(),
    }


def stage_generate(args: argparse.Namespace) -> Dict[str, Any]:
    """One SDXL image at the requested resolution, with peak video memory and wall clock."""
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    dtype = getattr(torch, args.dtype)
    before = _memory_snapshot()
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(_VAE_ID, torch_dtype=dtype)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        _MODEL_ID, vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True,
    )
    pipeline = pipeline.to("cuda")
    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=False)
    load_seconds = round(time.time() - t0, 1)

    torch.cuda.reset_peak_memory_stats()
    per_image_seconds: List[float] = []
    for index in range(args.images):
        generator = torch.Generator(device="cuda").manual_seed(42)
        t0 = time.time()
        image = pipeline(
            prompt="An image of Mark Philippoussis",
            generator=generator,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.inference_steps,
        ).images[0]
        torch.cuda.synchronize()
        per_image_seconds.append(round(time.time() - t0, 1))
        out_path = _OUT / f"probe_generate_{args.resolution}_{index}.png"
        image.save(out_path)
        print(f"image {index}: {per_image_seconds[-1]}s -> {out_path}", flush=True)

    steady = per_image_seconds[1:] or per_image_seconds
    return {
        "stage": "generate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": args.dtype,
        "resolution": args.resolution,
        "inference_steps": args.inference_steps,
        "images": args.images,
        "vae_tiling": args.vae_tiling,
        "pipeline_load_seconds": load_seconds,
        "first_image_seconds": per_image_seconds[0],
        "steady_state_seconds_per_image": round(sum(steady) / len(steady), 1),
        "per_image_seconds": per_image_seconds,
        "before_load": before,
        "peak_torch_allocated_gb": _gb(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_gb": _gb(torch.cuda.max_memory_reserved()),
        "after_generate": _memory_snapshot(),
    }


_STAGES: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {
    "env": stage_env,
    "load": stage_load,
    "step": stage_step,
    "generate": stage_generate,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(_STAGES), required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--forget-weight", type=float, default=0.3)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true",
                        help="Tile and slice the autoencoder, the component both 1024 stages died in.")
    parser.add_argument("--split-backward", action="store_true",
                        help="Backward each half of the loss separately; same update, one activation graph at a time.")
    parser.add_argument("--precompute", action="store_true",
                        help="Encode latents and captions once, then evict the autoencoder and text encoders.")
    parser.add_argument("--tag", default="", help="Appended to the output filename, so variants coexist.")
    args = parser.parse_args()

    if os.environ.get("HF_HUB_DISABLE_XET") != "1" and args.stage != "env":
        print("REFUSING: set HF_HUB_DISABLE_XET=1 before any stage that downloads from HuggingFace.")
        return 2

    payload = _STAGES[args.stage](args)
    suffix = f"_{args.tag}" if args.tag else ""
    _write(f"probe_{args.stage}{suffix}.json", payload)
    print(f"PROBE_OK {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
