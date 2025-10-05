from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Dict, List, Literal
import numpy as np
from PIL import Image

import torch
from transformers import (
    pipeline,
    AutoImageProcessor,
    SiglipForImageClassification,
)
from transformers.pipelines.image_classification import ImageClassificationPipeline
import piq

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
    device: Union[int, str, torch.device] = 'cuda'
    _pipeline: Optional[ImageClassificationPipeline] = None

    def model_post_init(self, __context: Optional[dict] = None) -> None:
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
    device: Union[int, str, torch.device] = 'cpu'
    _model_name: str = "prithivMLmods/Realistic-Gender-Classification"
    _id2label = {0: 'female', 1: 'male'}
    _model: Any
    _processor: Any

    def model_post_init(self, __context):
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
    '''
    https://ieeexplore.ieee.org/document/6272356
    '''
    def score(self, image: Image.Image) -> Dict[str, float]:
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        return {
            'brisque': float(piq.brisque(image_tensor, data_range=1.0).item()),
        }


# This is how to use it
'''
import torch
from PIL import Image

image = Image.open('assets/male.jpg')
metric_quality = MetricQuality()
result = metric_quality.score(image)
print(result)
'''
