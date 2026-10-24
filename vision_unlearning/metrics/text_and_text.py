from typing import Literal, List, Dict, Tuple, Optional, Callable, Union
from functools import partial
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor

from vision_unlearning.metrics.base import Metric


class MetricTextTextSimilarity(Metric):
    metrics: List[Literal['clip_text']]
    _clip_score_fn: Optional[Callable] = None

    def model_post_init(self, __context: Optional[dict]) -> None:
        # Download the models, if required
        if 'clip_text' in self.metrics:
            # Loaded once and handed to clip_score as a callable: given a model name, clip_score loads the
            # model and its processor from disk on every call, which made a 100 x 100 similarity matrix
            # take hours on one processor core. The model, the processor and the computation are the same.
            model, processor = _get_clip_model_and_processor("openai/clip-vit-base-patch16")
            self._clip_score_fn = partial(clip_score, model_name_or_path=lambda: (model, processor))

    def score(self, text1: str, text2: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Calculate
        for metric in self.metrics:
            if metric == 'clip_text':
                assert self._clip_score_fn is not None
                scores[metric] = float(self._clip_score_fn(text1, text2).detach())
        return scores
