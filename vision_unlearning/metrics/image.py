from typing import Union, Optional, Any, Dict, List, Literal
from PIL import Image
import torch
from transformers import pipeline
from transformers.pipelines.image_classification import ImageClassificationPipeline
from libs.metrics.base import Metric


class MetricPaintingStyle(Metric):
    metrics: List[Literal['is_desired_style', 'desired_style_confidence']] = []  # TODO: this is currently ignored 
    desired_style: str
    top_k: int = 5
    model_path: str
    device: Union[int, str, torch.device] = 'cuda'
    _pipeline: Optional[ImageClassificationPipeline] = None

    def model_post_init(self, __context):
        self._pipeline = pipeline('image-classification', model=self.model_path, device=self.device)

    def score(self, image: Image) -> Dict[str, bool|float]:
        scores = {
            'is_desired_style': False,
            'desired_style_confidence': 0.0
        }
        predictions: list = self._pipeline(image, top_k=self.top_k)
        for p in predictions:
            if p['label']==self.desired_style:
                scores['is_desired_style'] = True
                scores['desired_style_confidence'] = float(p['score'])
        return scores


# Pseudo test
'''
import torch
from PIL import Image

#image = Image.open('assets/Diffusion-MU-Attack/files/dataset/vangogh/imgs/35_0.png')
image = Image.open('assets/Diffusion-MU-Attack/files/dataset/i2p_nude/imgs/1011_0.png')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric_painting_style = MetricPaintingStyle(desired_style='vincent-van-gogh', top_k=3, model_path='assets/models_pretrained/style_classifier/results/checkpoint-2800', device=device)
result = metric_painting_style.score(image)
print(result)
''';
