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

Run from: unlearning/unlearning-analysis/
Interpreter: C:/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from vision_unlearning.utils.logger import get_logger

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

def extract_unet_crossattn_activations(
    pipeline: Any,        # StableDiffusionPipeline
    prompt: str,
    n_steps: int,
    seeds: List[int],
    device: str,
) -> np.ndarray:
    """
    Run SD inference for `prompt` using `pipeline` and collect UNet cross-attention
    activations (output of all attn2 modules) across denoising steps and seeds.

    Returns a L2-normalized 1-D fingerprint vector of shape (D,) where D is
    the sum of hidden dims across all 16 cross-attention layers (12480 for SD 1.4).

    The fingerprint is computed as:
      1. For each seed, run `pipeline(prompt, num_inference_steps=n_steps)`.
      2. At each denoising step, each attn2 hook fires and stores the hidden-state
         output (shape [batch, seq, dim]).  Spatial (seq) positions are averaged to
         get a [dim] vector per layer per step.
      3. Average across steps, then across seeds, then concatenate across layers.
      4. L2-normalize the result.

    @param pipeline: Loaded StableDiffusionPipeline with UNet and tokenizer.
    @param prompt: Text prompt, e.g. "An image of an abbey scene".
    @param n_steps: Number of denoising steps per inference (small: 10).
    @param seeds: List of integer seeds for reproducible generation.
    @param device: Torch device string, e.g. 'cuda' or 'cpu'.
    @return: L2-normalized fingerprint array of shape (D,).
    """
    import torch

    # Identify all cross-attention (attn2) modules in the UNet.
    # Pattern: any module whose full qualified name ends with '.attn2'.
    attn2_names: List[str] = []
    attn2_modules: Dict[str, Any] = {}
    for name, module in pipeline.unet.named_modules():
        if name.endswith('.attn2'):
            attn2_names.append(name)
            attn2_modules[name] = module

    if not attn2_names:
        raise RuntimeError(
            "No cross-attention (attn2) modules found in pipeline.unet. "
            "Verify that the pipeline uses a UNet with cross-attention layers "
            "(expected for Stable Diffusion)."
        )

    # Storage: layer_name -> list of per-step per-seed [dim] arrays
    # After collection: seed_activations[name] = list of np.ndarray[dim], one per step×seed
    all_activations: Dict[str, List[np.ndarray]] = defaultdict(list)

    def _make_hook(layer_name: str) -> Any:
        """Return a forward hook that captures and stores the output hidden states."""
        def _hook(module: Any, _input: Any, output: Any) -> None:
            # output can be a Tensor or (Tensor, ...) depending on diffusers version
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # hidden shape: [batch, seq, dim]
            # Mean over batch (=1) and seq (spatial positions) → [dim]
            vec: np.ndarray = hidden.detach().float().mean(dim=(0, 1)).cpu().numpy()
            all_activations[layer_name].append(vec)
        return _hook

    # Register hooks
    handles: List[Any] = []
    for name in attn2_names:
        h = attn2_modules[name].register_forward_hook(_make_hook(name))
        handles.append(h)

    # ------------------------------------------------------------------ #
    # TODO (determinism gap — fix the NEXT time these fingerprints are     #
    # recomputed, so we can verify nothing shifts): this loop passes a     #
    # per-seed torch.Generator, but — unlike                              #
    # vision_unlearning.utils.data_generation.generate_dataset — it does   #
    # NOT enable torch.use_deterministic_algorithms(True), does NOT set    #
    # CUBLAS_WORKSPACE_CONFIG=':4096:8', and does NOT seed the global      #
    # torch / numpy / random state. On AMD ROCm, kernel selection for      #
    # GEMM/attention is non-deterministic without that regime, so the raw  #
    # fingerprints here are NOT bit-reproducible for a given seed. It is    #
    # tolerable for `act` only because the fingerprint is averaged over     #
    # spatial positions, denoising steps, and seeds before cosine          #
    # similarity, which washes out most kernel-level noise. Any per-entity  #
    # LATENT captured here (e.g. a future z_0 similarity) is a single       #
    # un-averaged tensor and would NOT be protected by that averaging, so   #
    # adopt generate_dataset's full determinism regime before trusting it.  #
    # ------------------------------------------------------------------ #
    try:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                pipeline(
                    prompt=prompt,
                    num_inference_steps=n_steps,
                    generator=generator,
                    output_type="np",   # skip PIL conversion overhead
                )
    finally:
        for h in handles:
            h.remove()

    # Aggregate: for each layer, average all stored [dim] vectors (across steps × seeds)
    per_layer_means: List[np.ndarray] = []
    for name in attn2_names:
        vecs = all_activations[name]
        if not vecs:
            raise RuntimeError(
                f"No activations captured for layer {name}. "
                "The hook may not have fired — check that the layer is called during inference."
            )
        layer_mean: np.ndarray = np.stack(vecs).mean(axis=0)  # [dim]
        per_layer_means.append(layer_mean)

    # Concatenate across layers → [D]
    fingerprint: np.ndarray = np.concatenate(per_layer_means)

    # L2-normalize
    norm = float(np.linalg.norm(fingerprint))
    if norm < 1e-8:
        raise RuntimeError(
            f"Near-zero fingerprint norm ({norm:.2e}) for prompt '{prompt}'. "
            "This is unexpected — check that the UNet is producing non-trivial outputs."
        )
    fingerprint = fingerprint / norm

    return fingerprint  # shape (D,)


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
    Compute activation fingerprints for all entities in a task.

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
        fp = extract_unet_crossattn_activations(
            pipeline=pipeline,
            prompt=prompt,
            n_steps=n_steps,
            seeds=seeds,
            device=device,
        )
        fingerprints[entity_name] = fp

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
