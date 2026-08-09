from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Dict, List, Literal, ClassVar
import tempfile
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
import torch
from torchvision import transforms

try:
    from transformers import (
        pipeline,
        AutoImageProcessor,
        SiglipForImageClassification,
    )
    from transformers.pipelines.image_classification import ImageClassificationPipeline
except ImportError:  # pragma: no cover - optional in lightweight environments
    pipeline = None  # type: ignore[assignment]
    AutoImageProcessor = None  # type: ignore[assignment]
    SiglipForImageClassification = None  # type: ignore[assignment]
    ImageClassificationPipeline = Any  # type: ignore[assignment]

try:
    import piq
except ImportError:  # pragma: no cover - optional in lightweight environments
    piq = None

from vision_unlearning.metrics.base import Metric


# TODO take these pseudo tests and examples and transform into automated test


class MetricImage(Metric, ABC):
    '''
    Based only on the image itself
    e.g., image quality, painting style
    '''
    @abstractmethod
    def score(self, image: Image.Image) -> Dict[str, Any]:
        pass


class MetricPaintingStyle(MetricImage):
    metrics: List[Literal['is_desired_style', 'desired_style_confidence']] = []  # TODO: this is currently ignored
    desired_style: str
    top_k: int = 5
    model_path: str
    device: Optional[torch.device | str | int] = 'cuda'
    _pipeline: Optional[ImageClassificationPipeline] = None

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        if pipeline is None:
            raise ImportError("transformers is required for MetricPaintingStyle")
        self._pipeline = pipeline('image-classification', model=self.model_path, device=self.device)

    def score(self, image: Image.Image) -> Dict[str, bool | float]:
        assert self._pipeline is not None
        scores = {
            'is_desired_style': False,
            'desired_style_confidence': 0.0
        }
        predictions: list = self._pipeline(image, top_k=self.top_k)
        for p in predictions:
            if p['label'] == self.desired_style:
                scores['is_desired_style'] = True
                scores['desired_style_confidence'] = float(p['score'])
        return scores


# Pseudo test
# import torch
# from PIL import Image
#
# #image = Image.open('assets/Diffusion-MU-Attack/files/dataset/vangogh/imgs/35_0.png')
# image = Image.open('assets/Diffusion-MU-Attack/files/dataset/i2p_nude/imgs/1011_0.png')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# metric_painting_style = MetricPaintingStyle(desired_style='vincent-van-gogh', top_k=3, model_path='assets/models_pretrained/style_classifier/results/checkpoint-2800', device=device)
# result = metric_painting_style.score(image)
# print(result)


class MetricRace(MetricImage):
    """
    Race classification using Hugging Face model:
    syntheticbot/clip-face-attribute-classifier

    Requires the following additional dependencies:
    * tf_keras = "~2.19.0"
    * tensorrt = "~10.13.2"
    * blinker = "~1.9.0"
    """
    # TODO: if we could do this with a HF model model be better, no need for additional libs

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        try:
            from deepface import DeepFace  # noqa
            self.DeepFace = DeepFace
        except ImportError as e:
            raise ImportError("DeepFace library is required for MetricRace. Please install it via 'pip install deepface'. Recommended version: deepfaces = '~0.0.95', tf_keras = '~2.19.0', tensorrt = '~10.13.2'") from e

    def score(self, image: Image.Image) -> Dict[str, str]:
        results = self.DeepFace.analyze(
            np.array(image.convert('RGB')),
            actions=['race'],
            enforce_detection=False,
        )

        # DeepFace may return list if multiple faces
        if isinstance(results, list):
            results = results[0]

        return {
            "race": results.get("dominant_race"),
        }


# Example usage
'''
img = Image.open("assets/datasets/lfw_splits/George_W_Bush/train_forget/George_W_Bush_0001.jpg")
metric_race = MetricRace()
print(metric_race.score(img))
'''


class MetricGender(MetricImage):
    device: Optional[torch.device | str | int] = 'cpu'
    _model_name: str = "prithivMLmods/Realistic-Gender-Classification"
    _id2label = {0: 'female', 1: 'male'}
    _model: Any
    _processor: Any

    def model_post_init(self, __context):
        if AutoImageProcessor is None or SiglipForImageClassification is None:
            raise ImportError("transformers is required for MetricGender")
        # load processor & model from HF
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = SiglipForImageClassification.from_pretrained(self._model_name)
        self._model.to(self.device).eval()

    def score(self, image: Image.Image) -> Dict[str, Union[Literal['male', 'female'], float]]:
        # ensure RGB and prepare batch
        img = image.convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            idx = int(torch.argmax(probs))
            label: Literal['male', 'female'] = self._id2label[idx]  # type: ignore
            confidence: float = float(probs[idx])

        return {
            'gender': label,
            'gender_confidence': confidence
        }


# This is how to use it
'''
import torch
from PIL import Image

image = Image.open('assets/male.jpg')
#image = Image.open('assets/female.jpg')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric_gender = MetricGender(device=device)
result = metric_gender.score(image)
print(result)
'''


class MetricQuality(MetricImage):
    def _load_image(self, img: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        img_obj: PILImage
        if isinstance(img, str):
            img_obj = Image.open(img)

        elif isinstance(img, np.ndarray):
            arr = img
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            img_obj = Image.fromarray(arr)

        else:
            img_obj = img

        tensor: torch.Tensor = torch.from_numpy(
            np.array(img_obj, copy=True)
        ).float()

        assert tensor.ndim == 3
        assert tensor.shape[2] in {1, 3}

        tensor = tensor.permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W]
        return tensor.to(self.device, non_blocking=True)

    def score(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, float]:
        image_tensor: torch.Tensor = self._load_image(image).unsqueeze(0)

        assert image_tensor.ndim == 4
        assert image_tensor.dtype == torch.float32
        assert 0.0 <= float(image_tensor.min())
        assert float(image_tensor.max()) <= 1.0

        if piq is None:
            raise ImportError("piq is required for MetricQuality")

        with torch.no_grad():
            score: torch.Tensor = piq.brisque(
                image_tensor, data_range=1.0
            )

        return {"brisque": float(score.item())}

    def score_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
    ) -> List[Dict[str, float]]:
        tensors: List[torch.Tensor] = [self._load_image(img) for img in images]

        shapes = {t.shape for t in tensors}
        if len(shapes) != 1:
            raise ValueError(
                "All images must have identical shape for batching. "
                f"Found shapes: {shapes}"
            )

        batch: torch.Tensor = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)  # [N, C, H, W]

        assert batch.ndim == 4
        assert batch.dtype == torch.float32
        assert 0.0 <= float(batch.min())
        assert float(batch.max()) <= 1.0

        if piq is None:
            raise ImportError("piq is required for MetricQuality")

        with torch.no_grad():
            scores: torch.Tensor = piq.brisque(  # [N]
                batch,
                data_range=1.0,
                reduction="none",
            )

        assert scores.ndim == 1, f"Return shape is {scores.ndim}"
        assert scores.shape[0] == batch.shape[0]

        results: List[Dict[str, float]] = [
            {"brisque": float(v.item())} for v in scores
        ]

        assert len(results) == len(images)
        return results


# This is how to use it
'''
import torch
from PIL import Image

image = Image.open('assets/male.jpg')
metric_quality = MetricQuality()
result = metric_quality.score(image)
print(result)
'''


class MetricImageClassifier(MetricImage):
    """Classify an image with a fine-tuned timm backbone whose head was replaced.

    Loads a checkpoint saved as {"model_state_dict": ...} over
    timm.create_model(backbone, pretrained=True) with model.head replaced by
    Linear(head_in_features, len(labels)). Transform and softmax/argmax reproduce
    UnlearnCanvas' accuracy.py exactly: Resize((224,224)) -> ToTensor() -> Normalize([0.5],[0.5]).
    """

    from vision_unlearning.benchmarks.u_care.configuration import STYLE_ENTITIES, OBJECT_ENTITIES

    checkpoint_path: str
    labels: List[str] = STYLE_ENTITIES + OBJECT_ENTITIES
    backbone: str = "vit_large_patch16_224.augreg_in21k"
    head_in_features: int = 1024
    _model: Optional[torch.nn.Module] = None
    _transform: Optional[transforms.Compose] = None
    STYLE_ENTITIES: ClassVar[List[str]] = STYLE_ENTITIES
    OBJECT_ENTITIES: ClassVar[List[str]] = OBJECT_ENTITIES

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        import timm

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = str(device)

        model = timm.create_model(self.backbone, pretrained=True)
        model.head = torch.nn.Linear(self.head_in_features, len(self.labels))
        checkpoint = torch.load(self.checkpoint_path, map_location=device_str)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device_str).eval()

        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def score(self, image: Image.Image) -> Dict[str, Any]:
        """Return {"predicted_label": str, "probabilities": Dict[str, float]}."""
        assert self._model is not None
        assert self._transform is not None

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = str(device)

        image_tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(device_str)

        with torch.no_grad():
            logits = self._model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())

        probabilities = {}
        for label, prob in zip(self.labels, probs.cpu().tolist()):
            if isinstance(prob, torch.Tensor):
                value = float(prob.item())
            else:
                value = float(prob)
            probabilities[label] = value
        return {
            "predicted_label": self.labels[predicted_idx],
            "probabilities": probabilities,
        }

