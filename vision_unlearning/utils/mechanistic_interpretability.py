"""
mechanistic_interpretability.py — UNet cross-attention activation fingerprinting.

Provides utilities for computing per-entity "activation fingerprints" from the
original Stable Diffusion 1.4 UNet.  The fingerprint is the concatenation of
mean cross-attention (attn2) hidden-state outputs across all cross-attention
layers, averaged over denoising steps and generation seeds.

The resulting fingerprints are L2-normalized and can be compared via cosine
similarity to define an activation-based entity similarity metric (type_s='act')
coherent with the rest of the I-CARE methodology.

Design goals:
- All heavy computation (model loading, forward passes) lives here, not in RTs.
- RTs use `load_act_fingerprints` + `compute_cosine_similarity_matrix` (cheap).
- The standalone runner `3b_compute_act_fingerprints.py` calls
  `compute_act_fingerprints_for_task` + `save_act_fingerprints`.

File naming convention (follows assets/datasets/ layout):
    act_fingerprints_{task}_{model}.json
    e.g. act_fingerprints_scenes_sd1.4.json

Actual fingerprint dimensionality for SD 1.4 (16 cross-attn layers):
    down[0]: 2×320=640, down[1]: 2×640=1280, down[2]: 2×1280=2560
    mid: 1×1280=1280
    up[1]: 3×1280=3840, up[2]: 3×640=1920, up[3]: 3×320=960
    Total: 12480 dims  (SD1.4 has 3 CrossAttnDownBlocks, not 2)

"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from vision_unlearning.utils.logger import get_logger
from vision_unlearning.utils.determinism import set_determinism
from vision_unlearning.utils.noise import type_noise

logger = get_logger('mechanistic_interpretability')


# ---------------------------------------------------------------------------
# SD model name for sd1.4
# ---------------------------------------------------------------------------
SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# Fingerprint file path template — kept consistent with assets/datasets/ convention
_FINGERPRINT_FILENAME = "act_fingerprints_{task}_{model}.json"


def get_act_fingerprints_path(task: str, model: str, base_folder: str) -> str:
    """Local path for the activation fingerprints JSON file."""
    return os.path.join(
        base_folder,
        "datasets",
        _FINGERPRINT_FILENAME.format(task=task, model=model),
    )


# ---------------------------------------------------------------------------
# Hook-based activation extractor
# ---------------------------------------------------------------------------

def _attn2_modules(pipeline: Any) -> Dict[str, Any]:
    """Every cross-attention (``attn2``) module of the denoiser, by qualified name, in model order."""
    modules: Dict[str, Any] = {
        name: module for name, module in pipeline.unet.named_modules() if name.endswith('.attn2')
    }
    if not modules:
        raise RuntimeError(
            "No cross-attention (attn2) modules found in pipeline.unet. "
            "Verify that the pipeline uses a UNet with cross-attention layers "
            "(expected for Stable Diffusion)."
        )
    return modules


def _per_sample_means(hidden: Any, batch_size: int) -> np.ndarray:
    """One ``[dim]`` vector per prompt of the batch, from one attn2 output.

    Under classifier-free guidance the denoiser sees each prompt twice, unconditioned then
    conditioned, so the output holds ``2 * batch_size`` rows: ``[uncond_0 .. uncond_{B-1},
    cond_0 .. cond_{B-1}]``. A prompt's vector is the mean over both of its rows and every spatial
    position, which for a batch of one is exactly the mean over dimensions 0 and 1 that this
    capture has always taken.
    """
    rows = int(hidden.shape[0])
    if rows % batch_size != 0:
        raise RuntimeError(f"attn2 output has {rows} rows, not a multiple of the batch size {batch_size}")
    grouped = hidden.detach().float().reshape(rows // batch_size, batch_size, *hidden.shape[1:])
    result: np.ndarray = grouped.mean(dim=(0, 2)).cpu().numpy()
    return result  # [batch_size, dim]


def capture_act_fingerprints(
    pipeline: Any,        # StableDiffusionPipeline
    prompts: List[str],
    n_steps: int,
    seeds: List[int],
    device: str,
    noise: Optional[type_noise] = None,
    batch_size: int = 1,
) -> Tuple[List[np.ndarray], Dict[int, Dict[int, str]]]:
    """
    The activation fingerprint of every prompt, and the hash of every initial latent it was given.

    Runs SD inference for the prompts, seed by seed, in batches of `batch_size` in prompt order,
    and collects the cross-attention activations (output of all attn2 modules) of every prompt
    separately. The fingerprint of a prompt is computed as:
      1. At each denoising step, each attn2 hook stores, per prompt of the batch, the hidden-state
         output averaged over spatial positions and over the unconditioned and conditioned rows
         (see `_per_sample_means`): a [dim] vector per layer per step per seed.
      2. Average across steps and seeds, then concatenate across layers.
      3. L2-normalize the result.

    The whole capture runs under the determinism regime of
    `vision_unlearning.utils.data_generation.generate_dataset` -- deterministic algorithms,
    `CUBLAS_WORKSPACE_CONFIG`, and the global torch / numpy / random state seeded once per seed --
    so a fingerprint is bit-reproducible for a given seed on one card model.

    @param noise: how each image's initial noise is drawn. None (the default) is this capture's
        historical behaviour: a fresh generator on `device` seeded to the seed for every pipeline
        call, so every prompt of a call starts from the card's draw for that seed.
        ``'per_image_processor'`` draws each image's latent with
        `vision_unlearning.utils.noise.initial_latents` and passes it as ``latents=``, which is how
        the benchmark's images are generated, and records the hash of each latent handed over.
        ``'stream'`` is refused: this capture never advanced one generator across the prompt list.
    @param batch_size: prompts per pipeline call.
    @return: (one L2-normalized fingerprint of shape (D,) per prompt, in prompt order;
        ``{prompt index: {seed: initial-latent hash}}``, empty unless noise is
        ``'per_image_processor'``).
    """
    import random
    import torch
    from vision_unlearning.utils.noise import initial_latents

    if noise == 'stream':
        raise ValueError("the activation capture has no 'stream' convention; use None or 'per_image_processor'")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    attn2_modules = _attn2_modules(pipeline)
    attn2_names = list(attn2_modules)
    # activations[prompt index][layer name] -> one [dim] vector per step and seed
    activations: Dict[int, Dict[str, List[np.ndarray]]] = {
        index: defaultdict(list) for index in range(len(prompts))
    }
    current: Dict[str, Any] = {'indices': []}

    def _make_hook(layer_name: str) -> Any:
        """Return a forward hook that stores the output hidden states of every prompt in the call."""
        def _hook(module: Any, _input: Any, output: Any) -> None:
            # output can be a Tensor or (Tensor, ...) depending on diffusers version
            hidden = output[0] if isinstance(output, tuple) else output
            indices: List[int] = current['indices']
            for index, vector in zip(indices, _per_sample_means(hidden, len(indices))):
                activations[index][layer_name].append(vector)
        return _hook

    latent_hashes: Dict[int, Dict[int, str]] = {}
    handles: List[Any] = [
        attn2_modules[name].register_forward_hook(_make_hook(name)) for name in attn2_names
    ]
    set_determinism(True)
    try:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start:start + batch_size]
                current['indices'] = list(range(start, start + len(batch)))
                noise_kwargs: Dict[str, Any]
                if noise == 'per_image_processor':
                    latents, hashes = initial_latents(pipeline, seed, len(batch))
                    noise_kwargs = {'latents': latents}
                    for offset, latent_hash in enumerate(hashes):
                        latent_hashes.setdefault(start + offset, {})[seed] = latent_hash
                else:
                    noise_kwargs = {'generator': torch.Generator(device=device).manual_seed(seed)}
                with torch.no_grad():
                    pipeline(
                        prompt=batch,
                        num_inference_steps=n_steps,
                        output_type="np",   # skip PIL conversion overhead
                        **noise_kwargs,
                    )
    finally:
        for h in handles:
            h.remove()
        set_determinism(False)

    fingerprints: List[np.ndarray] = []
    for index, prompt in enumerate(prompts):
        # Aggregate: for each layer, average all stored [dim] vectors (across steps x seeds)
        per_layer_means: List[np.ndarray] = []
        for name in attn2_names:
            vecs = activations[index][name]
            if not vecs:
                raise RuntimeError(
                    f"No activations captured for layer {name}. "
                    "The hook may not have fired — check that the layer is called during inference."
                )
            per_layer_means.append(np.stack(vecs).mean(axis=0))  # [dim]

        # Concatenate across layers → [D]
        fingerprint: np.ndarray = np.concatenate(per_layer_means)

        # L2-normalize
        norm = float(np.linalg.norm(fingerprint))
        if norm < 1e-8:
            raise RuntimeError(
                f"Near-zero fingerprint norm ({norm:.2e}) for prompt '{prompt}'. "
                "This is unexpected — check that the UNet is producing non-trivial outputs."
            )
        fingerprints.append(fingerprint / norm)

    return fingerprints, latent_hashes


def extract_unet_crossattn_activations(
    pipeline: Any,        # StableDiffusionPipeline
    prompt: str,
    n_steps: int,
    seeds: List[int],
    device: str,
) -> np.ndarray:
    """
    The L2-normalized activation fingerprint of one prompt, shape (D,), where D is the sum of
    hidden dims across all 16 cross-attention layers (12480 for SD 1.4).

    One prompt at a time, with this capture's historical noise; see `capture_act_fingerprints`,
    which this calls, for how the fingerprint is computed.

    @param pipeline: Loaded StableDiffusionPipeline with UNet and tokenizer.
    @param prompt: Text prompt, e.g. "An image of an abbey scene".
    @param n_steps: Number of denoising steps per inference (small: 10).
    @param seeds: List of integer seeds for reproducible generation.
    @param device: Torch device string, e.g. 'cuda' or 'cpu'.
    @return: L2-normalized fingerprint array of shape (D,).
    """
    fingerprints, _ = capture_act_fingerprints(pipeline, [prompt], n_steps, seeds, device)
    return fingerprints[0]


# ---------------------------------------------------------------------------
# Batch fingerprint computation for a full task
# ---------------------------------------------------------------------------

def compute_act_fingerprints_for_task(
    task: str,
    pipeline: Any,       # StableDiffusionPipeline
    entity_prompts: Dict[str, str],
    device: str,
    n_steps: int = 10,
    seeds: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute activation fingerprints for all entities in a task, one prompt at a time, with this
    capture's historical noise. `capture_act_fingerprints` is the same capture with a choice of
    noise and batch size, and the initial-latent hashes returned.

    @param task: Task name ('scenes', 'people', 'breeds').
    @param pipeline: Loaded StableDiffusionPipeline (original model, no LoRA).
    @param entity_prompts: Mapping from entity_name to text prompt string.
                           e.g. {'abbey': 'An image of an abbey scene', ...}
    @param device: Torch device string.
    @param n_steps: Number of denoising steps per entity per seed.
    @param seeds: Seed list; defaults to [42, 43, 44, 45] if None.
    @return: Dict mapping entity_name -> L2-normalized fingerprint array.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45]

    fingerprints: Dict[str, np.ndarray] = {}
    total = len(entity_prompts)
    for idx, (entity_name, prompt) in enumerate(entity_prompts.items()):
        logger.info(
            "[%d/%d] Computing fingerprint for entity: %s",
            idx + 1, total, entity_name,
        )
        fingerprints[entity_name] = extract_unet_crossattn_activations(
            pipeline=pipeline,
            prompt=prompt,
            n_steps=n_steps,
            seeds=seeds,
            device=device,
        )

    logger.info("Computed %d fingerprints for task '%s'.", len(fingerprints), task)
    return fingerprints

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_act_fingerprints(
    fingerprints: Dict[str, np.ndarray],
    task: str,
    model: str,
    base_folder: str,
    n_steps: int,
    seeds: List[int],
    model_name: str = SD_MODEL_NAME,
    noise: Optional[str] = None,
    batch_size: Optional[int] = None,
    entity_prompts: Optional[Dict[str, str]] = None,
    initial_latent_sha256: Optional[Dict[str, Dict[str, str]]] = None,
    pairing: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save entity activation fingerprints to disk.

    Follows the assets/datasets/ naming convention used throughout I-CARE.

    @param fingerprints: Dict mapping entity_name -> fingerprint array.
    @param task: Task name.
    @param model: Model short name (e.g. 'sd1.4').
    @param base_folder: Base folder for assets (e.g. 'assets').
    @param n_steps: Denoising steps used during computation.
    @param seeds: Seeds used during computation.
    @param model_name: Full HF model name (informational).
    @param noise: the noise convention of the capture, recorded when given (see
        `capture_act_fingerprints`).
    @param batch_size: prompts per pipeline call, recorded when given.
    @param entity_prompts: the prompt each entity was captured with, recorded when given.
    @param initial_latent_sha256: ``{entity: {seed: hash}}`` of the initial latents the pipeline was
        handed, recorded when given. With the three above it is what lets a reader check, key by
        key, that the capture started from the same noise as the images it is compared with.
    @param pairing: the capture's pairing record (`vision_unlearning.utils.noise.capture_pairing_record`),
        recorded when given; `vision_unlearning.benchmarks.I_care.pairing.check` certifies it against the
        task baseline's with the class ``'act'``.
    @return: Path to the saved JSON file.
    """
    output_path = get_act_fingerprints_path(task, model, base_folder)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine layers_captured from any fingerprint (all have same structure)
    # We store fingerprint_dim as the only reliable structural info at this level
    sample_fp = next(iter(fingerprints.values()))
    fingerprint_dim = int(sample_fp.shape[0])

    records = [
        {"entity": name, "fingerprint": fp.tolist()}
        for name, fp in fingerprints.items()
    ]

    output: Dict[str, Any] = {
        "metadata": {
            "task": task,
            "model": model,
            "model_name": model_name,
            "n_steps": n_steps,
            "seeds": seeds,
            "fingerprint_dim": fingerprint_dim,
            "n_entities": len(fingerprints),
        },
        "fingerprints": records,
    }
    # Written only when given, so a call that passes none of them writes exactly the file it did
    # before these parameters existed.
    capture_record: Dict[str, Any] = {
        "noise": noise,
        "batch_size": batch_size,
        "entity_prompts": entity_prompts,
        "initial_latent_sha256": initial_latent_sha256,
        "pairing": pairing,
    }
    output["metadata"].update({key: value for key, value in capture_record.items() if value is not None})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    logger.info("Saved %d fingerprints to %s", len(records), output_path)
    return output_path


def load_act_fingerprints(
    task: str,
    model: str,
    base_folder: str,
) -> Dict[str, np.ndarray]:
    """
    Load activation fingerprints from disk.

    @param task: Task name.
    @param model: Model short name (e.g. 'sd1.4').
    @param base_folder: Base folder for assets (e.g. 'assets').
    @return: Dict mapping entity_name -> fingerprint array (L2-normalized).
    @raises FileNotFoundError: If the fingerprint file does not exist.
    """
    path = get_act_fingerprints_path(task, model, base_folder)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Activation fingerprints not found at {path}. "
            f"Run 3b_compute_act_fingerprints.py --task {task} first."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fingerprints: Dict[str, np.ndarray] = {}
    for entry in data["fingerprints"]:
        fingerprints[entry["entity"]] = np.array(entry["fingerprint"], dtype=np.float32)

    logger.info(
        "Loaded %d fingerprints for task '%s' (dim=%d).",
        len(fingerprints),
        task,
        data["metadata"].get("fingerprint_dim", -1),
    )
    return fingerprints


# ---------------------------------------------------------------------------
# Similarity matrix from fingerprints
# ---------------------------------------------------------------------------

def compute_cosine_similarity_matrix(
    fingerprints: Dict[str, np.ndarray],
    entity_list: List[str],
) -> pd.DataFrame:
    """
    Build an N×N cosine similarity matrix from pre-computed L2-normalized fingerprints.

    Since fingerprints are already L2-normalized, cosine similarity reduces to the
    dot product.

    @param fingerprints: Dict mapping entity_name -> L2-normalized fingerprint array.
    @param entity_list: Ordered list of entity names; defines rows and columns.
    @return: N×N DataFrame with entity_list as both index and columns.
    @raises KeyError: If any entity in entity_list is not in fingerprints.
    """
    missing = [e for e in entity_list if e not in fingerprints]
    if missing:
        raise KeyError(
            f"Fingerprints missing for {len(missing)} entities: {missing[:5]}..."
            if len(missing) > 5
            else f"Fingerprints missing for entities: {missing}"
        )

    # Build matrix: rows = entity_list, cols = fingerprint dim
    mat = np.stack([fingerprints[e] for e in entity_list])  # [N, D]

    # Cosine similarity = dot product of unit vectors
    sim_matrix = mat @ mat.T  # [N, N]

    return pd.DataFrame(sim_matrix, index=entity_list, columns=entity_list)
