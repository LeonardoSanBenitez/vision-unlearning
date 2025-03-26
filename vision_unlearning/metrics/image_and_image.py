from typing import List, Literal, Dict
from PIL import Image
from image_similarity_measures.evaluate import evaluation
import lpips
from torchvision import transforms
from libs.metrics.base import Metric


class MetricImageImage(Metric):
    _loss_alex: lpips.lpips.LPIPS
    _loss_vgg: lpips.lpips.LPIPS
    metrics: List[Literal['rmse', 'psnr', 'ssim', 'fsim', 'issm', 'sre', 'sam', 'uiq', 'lpips_alex', 'lpips_vgg']]

    def __init__(self, metrics: List[Literal['rmse', 'psnr', 'ssim', 'fsim', 'issm', 'sre', 'sam', 'uiq', 'lpips_alex', 'lpips_vgg']]):
        # TODO: use pydantic's constructor, and initialize the models as post init
        self.metrics = metrics
        # Download the models for the LPIPS metrics, if required
        if 'lpips_alex' in metrics:
            self._loss_alex = lpips.LPIPS(net='alex')
        if 'lpips_vgg' in metrics:
            self._loss_vgg = lpips.LPIPS(net='vgg')

    def _evaluate_lpips(self, org_img_path: str, pred_img_path: str, loss_fn: lpips.lpips.LPIPS) -> float:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        img_real_tensor = transform(Image.open(org_img_path)).unsqueeze(0)
        img_fake_tensor = transform(Image.open(pred_img_path)).unsqueeze(0)
        d = loss_fn(img_real_tensor, img_fake_tensor)
        return float(d.item())

    def score(self, org_img_path: str, pred_img_path: str) -> Dict[str, float]:
        distances = {}
        metrics_remaining = self.metrics.copy()
        if 'lpips_alex' in metrics_remaining:
            distances['lpips_alex'] = self._evaluate_lpips(org_img_path, pred_img_path, self._loss_alex)
            metrics_remaining.remove('lpips_alex')
        if 'lpips_vgg' in metrics_remaining:
            distances['lpips_vgg'] = self._evaluate_lpips(org_img_path, pred_img_path, self._loss_vgg)
            metrics_remaining.remove('lpips_vgg')
        if len(metrics_remaining) > 0:
            distances.update(evaluation(org_img_path, pred_img_path, metrics_remaining))
        assert len(distances) == len(self.metrics)
        # TODO: ensure distances are float
        return distances
