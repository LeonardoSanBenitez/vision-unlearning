"""The text-text CLIP score loads its model once, not once per pair.

Fails on the code before this change: the scorer handed ``clip_score`` a model name, and given a name
``clip_score`` loads the model and its processor from disk on every call -- a 100 x 100 similarity
matrix spent hours on one processor core doing little else. Heavy tier: imports torchmetrics.
"""
from typing import Any, List

import pytest
import torch

from vision_unlearning.metrics import text_and_text


def test_the_model_is_loaded_once_and_every_score_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: List[str] = []
    model, processor = object(), object()

    def load(name: str) -> Any:
        loads.append(name)
        return model, processor

    seen: List[Any] = []

    def fake_clip_score(source: str, target: str, model_name_or_path: Any) -> torch.Tensor:
        seen.append(model_name_or_path())
        return torch.tensor(42.0)

    monkeypatch.setattr(text_and_text, '_get_clip_model_and_processor', load)
    monkeypatch.setattr(text_and_text, 'clip_score', fake_clip_score)
    scorer = text_and_text.MetricTextTextSimilarity(metrics=['clip_text'])
    scores = [scorer.score('a dog', 'a cat')['clip_text'] for _ in range(3)]
    assert loads == ['openai/clip-vit-base-patch16']
    assert seen == [(model, processor)] * 3
    assert scores == [42.0] * 3
