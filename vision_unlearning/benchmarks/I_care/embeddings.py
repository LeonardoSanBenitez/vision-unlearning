"""DINOv2 embedding utilities for the I-CARE benchmark.

This module provides:
  - embed_forgetting_session(): embed all images from one forgetting session (entity or baseline)
  - load_dino_model(): load DINOv2 vits14 and return (model, transform, device) triple
  - embed_image_with_dino(): embed a single image using a pre-loaded DINOv2 model
  - compute_dino_image_similarity(): DINOv2 cosine similarity between two PIL images (for script 3)
  - compute_mean_embeddings_by_prompt(): {prompt: mean_embedding} from an embedding file dict
  - compute_dino_diff_for_emitter(): {prompt: dino_diff} from on/off embedding files (for 3b script)

Design notes:
  - Heavy GPU imports (torch, torchvision, PIL) are deferred to function call time
    so this module is safe to import in CPU-only environments.
  - embed_image_fn is injectable in embed_forgetting_session() for unit testing without GPU.
  - TODO: refactor embed_image_with_dino() into batched DataLoader for throughput.
  - TODO: add torch.compile() support.
  - TODO: add fp16 (half-precision) support for throughput.
  - TODO: add DataLoader parallelism (num_workers > 0).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# Model constants
EMBEDDING_MODEL = "dinov2_vits14"
EMBEDDING_DIM = 384


def load_dino_model(
    model_name: str = EMBEDDING_MODEL,
    force_device: Optional[str] = None,
) -> "Tuple[Any, Any, str]":
    """Load DINOv2 model, transform pipeline, and device.

    Heavy imports (torch, torchvision) happen here, not at module load.

    Args:
        model_name: DINOv2 model variant (default: 'dinov2_vits14' → 384-dim CLS).
        force_device: If set, use this device string instead of auto-detecting.

    Returns:
        (model, transform, device) tuple.
        model: DINOv2 PyTorch model in eval mode, on device.
        transform: torchvision.transforms pipeline (resize → crop → normalize).
        device: device string ('cuda' or 'cpu').
    """
    import torch
    import torchvision.transforms as T

    if force_device is not None:
        device: str = force_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    logger.info("Loading DINOv2 %s ...", model_name)
    dino_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_model = dino_model.to(device)  # type: ignore[attr-defined]
    dino_model.eval()  # type: ignore[attr-defined]
    logger.info("DINOv2 loaded.")

    dino_transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),  # type: ignore[attr-defined]
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return dino_model, dino_transform, device


def embed_image_with_dino(
    image_path: str,
    model: "Any",
    transform: "Any",
    device: str,
) -> List[float]:
    """Embed a single image using a pre-loaded DINOv2 model.

    Args:
        image_path: Path to a PNG/JPEG image on disk.
        model: DINOv2 model (from load_dino_model()).
        transform: torchvision transform (from load_dino_model()).
        device: device string ('cuda' or 'cpu').

    Returns:
        384-dim CLS embedding as a plain Python list of floats.
        TODO: refactor into batched DataLoader for throughput (currently single-image).
    """
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # type: ignore[attr-defined]
    with torch.no_grad():
        feat = model(tensor)  # type: ignore[operator]
    return feat.squeeze().tolist()  # type: ignore[attr-defined]


def embed_forgetting_session(
    dataset_folder: str,
    seeds: List[int],
    prompts: List[str],
    metadata_filtered: List[Dict[str, Any]],
    lora_state: Literal["on", "off"],
    task: str,
    embed_image_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Embed all images from one forgetting session (entity or baseline).

    Iterates over all (seed, prompt) combinations and embeds each matching image.
    Images that do not exist on disk are skipped with a warning.

    Args:
        dataset_folder: Local directory containing the generated images.
        seeds: List of generation seeds (e.g. [0, 1, 2, 3]).
        prompts: Full prompt strings (e.g. "An image of Colin Powell").
        metadata_filtered: Metadata list used to map prompt index → entity name.
                           metadata_filtered[i]['name'] corresponds to prompts[i].
        lora_state: 'on' for unlearned model images, 'off' for baseline images.
        task: Task name, passed to get_target_preprocessed().
        embed_image_fn: Injectable embedding function (image_path → [float]).
                        Required — there is no default. Pass embed_image_with_dino
                        (partially applied) or a test stub.

    Returns:
        List of records:
        [
            {
                'prompted_entity': str,   # entity name (preprocessed)
                'seed': int,
                'prompt': str,
                'embedding': List[float], # 384-dim CLS embedding
            },
            ...
        ]
    """
    from vision_unlearning.datasets.testbed import (
        get_target_preprocessed,
        get_generated_dataset_file,
    )

    if embed_image_fn is None:
        raise ValueError(
            "embed_image_fn is required — pass embed_image_with_dino "
            "(partially applied with model/transform/device) or a test stub."
        )

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        for i, prompt in enumerate(prompts):
            prompted_entity = get_target_preprocessed(task, metadata_filtered[i]["name"])  # type: ignore[arg-type]
            filename = get_generated_dataset_file(lora_state, seed, prompt)
            image_path = os.path.join(dataset_folder, filename)
            if not os.path.exists(image_path):
                logger.warning("Image not found, skipping: %s", image_path)
                continue
            embedding = embed_image_fn(image_path)
            records.append(
                {
                    "prompted_entity": prompted_entity,
                    "seed": seed,
                    "prompt": prompt,
                    "embedding": embedding,
                }
            )
    return records


def embed_forgetting_session_batched(
    dataset_folder: str,
    seeds: List[int],
    prompts: List[str],
    metadata_filtered: List[Dict[str, Any]],
    lora_state: Literal["on", "off"],
    task: str,
    model: "Any",
    transform: "Any",
    device: str,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Embed all images for one forgetting session using batched GPU inference.

    More efficient than embed_forgetting_session() for large image sets.
    Collects all (path, metadata) pairs first, then processes in batches via
    a simple loop, amortising Python overhead and maximising GPU utilisation.

    Args:
        dataset_folder: Local directory containing the generated images.
        seeds: List of generation seeds used.
        prompts: Full prompt strings.
        metadata_filtered: Metadata list: metadata_filtered[i]['name'] → prompts[i].
        lora_state: 'on' for unlearned model, 'off' for baseline.
        task: Task name, passed to get_target_preprocessed().
        model: DINOv2 model (from load_dino_model()), on device, in eval mode.
        transform: torchvision transform pipeline (from load_dino_model()).
        device: Torch device string ('cuda' or 'cpu').
        batch_size: Number of images per GPU forward pass (default 32).
                    TODO: tune based on VRAM; 32 images × 224×224 ≈ 220MB VRAM.

    Returns:
        Same structure as embed_forgetting_session().
    """
    import torch
    from PIL import Image

    from vision_unlearning.datasets.testbed import (
        get_target_preprocessed,
        get_generated_dataset_file,
    )

    # Collect all (image_path, metadata) tuples
    items: List[Dict[str, Any]] = []
    for seed in seeds:
        for i, prompt in enumerate(prompts):
            prompted_entity = get_target_preprocessed(task, metadata_filtered[i]["name"])  # type: ignore[arg-type]
            filename = get_generated_dataset_file(lora_state, seed, prompt)
            image_path = os.path.join(dataset_folder, filename)
            if not os.path.exists(image_path):
                logger.warning("Image not found, skipping: %s", image_path)
                continue
            items.append(
                {
                    "image_path": image_path,
                    "prompted_entity": prompted_entity,
                    "seed": seed,
                    "prompt": prompt,
                }
            )

    if not items:
        return []

    # Batch inference
    records: List[Dict[str, Any]] = []
    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start: batch_start + batch_size]
        tensors: List["Any"] = []
        for item in batch_items:
            img = Image.open(item["image_path"]).convert("RGB")
            tensors.append(transform(img))

        batch_tensor = torch.stack(tensors).to(device)  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            feats = model(batch_tensor)  # shape: (B, embedding_dim)

        for j, item in enumerate(batch_items):
            records.append(
                {
                    "prompted_entity": item["prompted_entity"],
                    "seed": item["seed"],
                    "prompt": item["prompt"],
                    "embedding": feats[j].tolist(),
                }
            )

    return records


def compute_dino_image_similarity(
    img_off: "Any",
    img_on: "Any",
    model: "Any",
    transform: "Any",
    device: str,
) -> float:
    """Compute DINOv2 cosine similarity between two PIL images (off and on).

    Used in 3_compute_caused_interferences.py to compute dino_diff for each
    (emitter, receiver, seed) triple from the raw generated images.

    Args:
        img_off: PIL Image — receiver image from original model (baseline).
        img_on:  PIL Image — receiver image from emitter's unlearned model.
        model:   DINOv2 model (from load_dino_model()), on device, in eval mode.
        transform: torchvision transform pipeline (from load_dino_model()).
        device:  Torch device string ('cuda' or 'cpu').

    Returns:
        Cosine similarity in [−1, 1]; typically in [0, 1] for natural images.
        1.0 = identical DINOv2 representations (no interference).
        Lower = more semantic drift = more interference.
    """
    import numpy as np
    import torch

    def _embed(pil_img: "Any") -> "Any":
        tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)  # type: ignore[attr-defined]
        with torch.no_grad():
            feat = model(tensor)  # type: ignore[operator]
        return feat.squeeze()  # type: ignore[attr-defined]

    emb_off = _embed(img_off)
    emb_on = _embed(img_on)
    # Cosine similarity via dot product of L2-normalised vectors
    emb_off_np = emb_off.cpu().numpy()  # type: ignore[attr-defined]
    emb_on_np = emb_on.cpu().numpy()  # type: ignore[attr-defined]
    norm_off = float(np.linalg.norm(emb_off_np))
    norm_on = float(np.linalg.norm(emb_on_np))
    if norm_off == 0.0 or norm_on == 0.0:
        return float('nan')
    return float(np.dot(emb_off_np, emb_on_np) / (norm_off * norm_on))


def compute_mean_embeddings_by_prompt(
    embedding_file: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Return {prompt: L2-normalised mean embedding} from an embedding file dict.

    Aggregates all records that share the same ``prompt`` field (across seeds),
    computes the element-wise mean, and L2-normalises the result.

    Uses ``prompt`` (not ``prompted_entity``) as the canonical key, per
    ICARE guidelines (prompted_entity has inconsistent formatting across tasks).

    Args:
        embedding_file: Parsed JSON dict with an ``"embeddings"`` list of records.
                        Each record must have ``"prompt"`` and ``"embedding"`` fields.

    Returns:
        Dict mapping prompt string → 384-dim L2-normalised mean embedding.
        Prompts with all-zero mean (degenerate) are omitted.
    """
    import numpy as np

    buckets: Dict[str, List[List[float]]] = defaultdict(list)
    for entry in embedding_file.get("embeddings", []):
        # Support two historical file formats:
        #   new format: {"prompt": "An image of X", "prompted_entity": "X", ...}
        #   old format: {"entity": "An image of X", "lora_state": "...", ...}
        prompt_key: str = entry.get("prompt") or entry.get("entity") or ""
        if not prompt_key:
            logger.warning("Embedding record missing both 'prompt' and 'entity' fields; skipping.")
            continue
        buckets[prompt_key].append(entry["embedding"])

    result: Dict[str, List[float]] = {}
    for prompt, vecs in buckets.items():
        arr = np.array(vecs, dtype=np.float32)
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            result[prompt] = (mean_vec / norm).tolist()
    return result


def compute_dino_diff_for_emitter(
    on_embedding_file: Dict[str, Any],
    off_embedding_file: Dict[str, Any],
) -> Dict[str, float]:
    """Compute dino_diff for all receiver prompts using pre-computed embedding files.

    dino_diff(emitter=E, receiver_prompt=p) =
        cosine_similarity(mean_on_embedding[p], mean_off_embedding[p])

    where mean_on_embedding comes from the per-entity embedding file (images generated
    by E's unlearned model for all receiver prompts) and mean_off_embedding comes from
    the baseline file (same prompts generated by the original SD).

    This function is the CPU-only path used by 3b_compute_caused_interferences_dino.py
    to backfill dino_diff in already-computed interference files without re-running
    image generation or DINOv2 inference.

    Note: this computes cosine_similarity(mean_on, mean_off) — similarity of averaged
    embeddings — which differs slightly from mean(cosine_similarity(on_s, off_s) for s
    in seeds), which is what 3_compute_caused_interferences.py computes from raw images.
    The difference is negligible for analysis purposes.

    Args:
        on_embedding_file:  Parsed JSON dict for the emitter's per-entity embedding file.
                            Contains embeddings of ALL receiver prompts run under E's
                            unlearned model.
        off_embedding_file: Parsed JSON dict for the baseline embedding file.
                            Contains embeddings of ALL receiver prompts under original SD.

    Returns:
        Dict mapping prompt string → dino_diff float.
        Prompts absent from either file are omitted (not included, not NaN).
    """
    import numpy as np

    on_embs = compute_mean_embeddings_by_prompt(on_embedding_file)
    off_embs = compute_mean_embeddings_by_prompt(off_embedding_file)

    result: Dict[str, float] = {}
    for prompt, on_vec in on_embs.items():
        if prompt not in off_embs:
            logger.warning("Prompt not found in baseline file, skipping: %s", prompt)
            continue
        off_vec = off_embs[prompt]
        on_arr = np.array(on_vec, dtype=np.float32)
        off_arr = np.array(off_vec, dtype=np.float32)
        # Both vectors are already L2-normalised by compute_mean_embeddings_by_prompt
        sim = float(np.dot(on_arr, off_arr))
        result[prompt] = sim
    return result
