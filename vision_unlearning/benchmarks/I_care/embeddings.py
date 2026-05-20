"""DINOv2 embedding utilities for the I-CARE benchmark.

This module provides:
  - embed_forgetting_session(): embed all images from one forgetting session (entity or baseline)
  - load_dino_model(): load DINOv2 vits14 and return (model, transform, device) triple
  - embed_image_with_dino(): embed a single image using a pre-loaded DINOv2 model

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
            prompted_entity = get_target_preprocessed(task, metadata_filtered[i]["name"])
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
