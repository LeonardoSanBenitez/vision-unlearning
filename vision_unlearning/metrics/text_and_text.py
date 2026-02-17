from typing import Literal, List, Dict, Tuple, Optional, Callable, Union
from functools import partial
from torchmetrics.functional.multimodal import clip_score

from vision_unlearning.metrics.base import Metric


class MetricTextTextSimilarity(Metric):
    metrics: List[Literal['clip_text']]
    _clip_score_fn: Optional[Callable] = None

    def model_post_init(self, __context: Optional[dict]) -> None:
        # Download the models, if required
        if 'clip_text' in self.metrics:
            self._clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def score(self, text1: str, text2: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Calculate
        for metric in self.metrics:
            if metric == 'clip_text':
                assert self._clip_score_fn is not None
                scores[metric] = float(self._clip_score_fn(text1, text2).detach())
        return scores
