from functools import partial
from typing import Literal, List, Dict, Tuple, Optional, Callable, Union
from PIL import Image
import numpy as np

import torch
import torch.utils.checkpoint
from torchmetrics.functional.multimodal import clip_score
from libs.metrics.base import Metric


class MetricImageTextSimilarity(Metric):
    metrics: List[Literal['clip']]
    _clip_score_fn: Optional[Callable] = None

    def model_post_init(self, __context: dict) -> None:
        # Download the models, if required
        if 'clip' in self.metrics:
            self._clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def score(self, image: Union[Image.Image, np.ndarray], text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Preprocess
        image_np: np.ndarray
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        image_int = (image_np * 255).astype("uint8")

        # Calculate
        for metric in self.metrics:
            if metric == 'clip':
                assert self._clip_score_fn is not None
                scores[metric] = float(self._clip_score_fn(torch.from_numpy(image_int), text).detach())
        return scores
