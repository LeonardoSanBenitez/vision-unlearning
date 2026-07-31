import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "vision_unlearning" / "metrics"

vision_unlearning_pkg = ModuleType("vision_unlearning")
vision_unlearning_pkg.__path__ = [str(ROOT / "vision_unlearning")]
sys.modules.setdefault("vision_unlearning", vision_unlearning_pkg)

metrics_pkg = ModuleType("vision_unlearning.metrics")
metrics_pkg.__path__ = [str(METRICS_DIR)]
sys.modules.setdefault("vision_unlearning.metrics", metrics_pkg)

base_spec = importlib.util.spec_from_file_location("vision_unlearning.metrics.base", METRICS_DIR / "base.py")
base_module = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_module
assert base_spec.loader is not None
base_spec.loader.exec_module(base_module)

MODULE_PATH = METRICS_DIR / "image.py"
SPEC = importlib.util.spec_from_file_location("vision_unlearning.metrics.image", MODULE_PATH)
image_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = image_module
assert SPEC.loader is not None
SPEC.loader.exec_module(image_module)


class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1024, 2)

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 2, dtype=torch.float32)
        logits[:, 1] = 1.0
        return logits


def test_metric_image_classifier_score(monkeypatch, tmp_path):
    fake_timm = SimpleNamespace(create_model=lambda *args, **kwargs: DummyClassifier())
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    checkpoint_path = tmp_path / "classifier.pt"
    model = DummyClassifier()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    metric = image_module.MetricImageClassifier(
        checkpoint_path=str(checkpoint_path),
        labels=["cat", "dog"],
        device="cpu",
    )

    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = metric.score(image)

    assert result["predicted_label"] == "dog"
    assert result["probabilities"]["dog"] > result["probabilities"]["cat"]
    assert result["probabilities"]["dog"] > 0.5
