from typing import List, Literal, Dict, Optional
from PIL import Image
from image_similarity_measures.evaluate import evaluation
import lpips
from torchvision import transforms
from vision_unlearning.metrics.base import Metric

import tempfile
from typing import Union
from PIL import Image

class MetricImageImage(Metric):
    _loss_alex: lpips.lpips.LPIPS
    _loss_vgg: Optional[lpips.lpips.LPIPS]
    metrics: List[Literal['rmse', 'psnr', 'ssim', 'fsim', 'issm', 'sre', 'sam', 'uiq', 'lpips_alex', 'lpips_vgg']]
    # SSIM interpertation: 1.0 → Images are identical. -1.0 → Images are totaly different
    def model_post_init(self, __context: dict) -> None:
        if 'lpips_alex' in self.metrics:
            self._loss_alex = lpips.LPIPS(net='alex')
        if 'lpips_vgg' in self.metrics:
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

    def _score_from_paths(self, org_img_path: str, pred_img_path: str) -> Dict[str, float]:
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
        return distances  # TODO: ensure distances are float
    
    def score(self, org_img: Union[str, Image.Image], pred_img: Union[str, Image.Image]) -> Dict[str, float]:
        if isinstance(org_img, str) and isinstance(pred_img, str):
            return self._score_from_paths(org_img, pred_img)
    
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_org, tempfile.NamedTemporaryFile(suffix=".png") as tmp_pred:
            if isinstance(org_img, Image.Image):
                org_img.save(tmp_org, format="PNG")
                tmp_org.flush()
                org_path = tmp_org.name
            else:
                org_path = org_img
    
            if isinstance(pred_img, Image.Image):
                pred_img.save(tmp_pred, format="PNG")
                tmp_pred.flush()
                pred_path = tmp_pred.name
            else:
                pred_path = pred_img
    
            return self._score_from_paths(org_path, pred_path)