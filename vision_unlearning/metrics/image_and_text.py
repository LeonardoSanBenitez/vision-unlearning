from typing import List, Dict, Union, Optional, Callable, Literal
from functools import partial

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore

from vision_unlearning.metrics.base import Metric


class MetricImageTextSimilarity(Metric):
    metrics: List[Literal['clip']]
    _clip_metric: Optional[CLIPScore] = None

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        # infer device if not already set
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'clip' in self.metrics:
            self._clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            self._clip_metric.to(self.device)

    def _load_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        if isinstance(image, str):
            image_obj = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_obj = image
        else:
            # np.ndarray
            image_obj = Image.fromarray(image.astype(np.uint8))

        image_np = np.array(image_obj)
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # convert to tensor and move to device
        img_tensor = torch.from_numpy(image_np).to(self.device, non_blocking=True)
        # reorder to (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def score(self, image: Union[Image.Image, np.ndarray, str], text: str) -> Dict[str, float]:
        assert self._clip_metric is not None

        image_tensor = self._load_image(image)

        # CLIPScore expects either a batch tensor (N, C, H, W) or list of (C, H, W).
        # We wrap single image into batch
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Move text (string) as is: CLIPScore handles tokenization internally
        with torch.inference_mode():
            score_tensor = self._clip_metric(image_tensor, text)

        return {"clip": float(score_tensor.detach().cpu().item())}

    def score_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        texts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Warning: this function don't improve performance. The underlying libraries still work serially.
        Returns per-pair results in the same order.
        """
        assert len(images) == len(texts), "images and texts must have same length"
        assert 'clip' in self.metrics, "score_batch is only implemented for 'clip'"
        assert self._clip_metric is not None
    
        results: List[Dict[str, float]] = []
        for img, txt in zip(images, texts):
            results.append(self.score(img, txt))
    
        # Ensure output length matches input length
        assert len(results) == len(images)
        return results
