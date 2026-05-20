from typing import List, Literal, Dict, Optional, Union
from PIL import Image
from image_similarity_measures.evaluate import evaluation
import lpips
from torchvision import transforms
from vision_unlearning.metrics.base import Metric

import os
import tempfile
import numpy as np


class MetricImageImage(Metric):
    _loss_alex: Optional[lpips.lpips.LPIPS]
    _loss_vgg: Optional[lpips.lpips.LPIPS]
    metrics: List[
        Literal[
            "rmse",
            "psnr",
            "ssim",
            "fsim",
            "issm",
            "sre",
            "sam",
            "uiq",
            "lpips_alex",
            "lpips_vgg",
        ]
    ]

    def model_post_init(self, __context: Optional[dict] = None) -> None:
        # initialize LPIPS if requested
        if "lpips_alex" in self.metrics:
            self._loss_alex = lpips.LPIPS(net="alex")
        else:
            self._loss_alex = None

        if "lpips_vgg" in self.metrics:
            self._loss_vgg = lpips.LPIPS(net="vgg")
        else:
            self._loss_vgg = None

    def _evaluate_lpips(
        self, org_img_path: str, pred_img_path: str, loss_fn: lpips.lpips.LPIPS
    ) -> float:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor [0,1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Normalize to [-1, 1]
            ]
        )
        img_real_tensor = transform(Image.open(org_img_path)).unsqueeze(0)
        img_fake_tensor = transform(Image.open(pred_img_path)).unsqueeze(0)
        d = loss_fn(img_real_tensor, img_fake_tensor)
        return float(d.item())

    def _score_from_paths(self, org_img_path: str, pred_img_path: str) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        metrics_remaining = self.metrics.copy()

        if "lpips_alex" in metrics_remaining:
            assert self._loss_alex is not None
            distances["lpips_alex"] = self._evaluate_lpips(
                org_img_path, pred_img_path, self._loss_alex
            )
            metrics_remaining.remove("lpips_alex")

        if "lpips_vgg" in metrics_remaining:
            assert self._loss_vgg is not None
            distances["lpips_vgg"] = self._evaluate_lpips(
                org_img_path, pred_img_path, self._loss_vgg
            )
            metrics_remaining.remove("lpips_vgg")

        if len(metrics_remaining) > 0:
            distances.update(evaluation(org_img_path, pred_img_path, metrics_remaining))

        assert len(distances) == len(self.metrics)
        return {k: float(v) for k, v in distances.items()}

    @staticmethod
    def _pil_or_array_to_path(img: Union[str, Image.Image, "np.ndarray"]) -> "tuple[str, bool]":  # type: ignore[name-defined]
        """Return (file_path, is_temp).  Caller must unlink when is_temp is True."""
        if isinstance(img, str):
            return img, False
        # Convert numpy array to PIL if needed.
        if isinstance(img, Image.Image):
            image_obj = img
        else:
            arr = img
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            image_obj = Image.fromarray(arr)
        # Windows: NamedTemporaryFile holds an exclusive lock while open, so PIL
        # cannot write to it by name.  Use delete=False, close immediately, then
        # clean up manually in the caller.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image_obj.save(tmp.name, format="PNG")
        return tmp.name, True

    def score(
        self, org_img: Union[str, Image.Image], pred_img: Union[str, Image.Image]
    ) -> Dict[str, float]:
        # Fast path: both are already file paths
        if isinstance(org_img, str) and isinstance(pred_img, str):
            return self._score_from_paths(org_img, pred_img)

        # Otherwise, ensure we have file paths for the underlying libraries by
        # saving non-path inputs to temporary files.
        org_path, org_is_temp = self._pil_or_array_to_path(org_img)
        pred_path, pred_is_temp = self._pil_or_array_to_path(pred_img)
        try:
            return self._score_from_paths(org_path, pred_path)
        finally:
            if org_is_temp:
                os.unlink(org_path)
            if pred_is_temp:
                os.unlink(pred_path)

    def score_batch(
        self,
        org_imgs: List[Union[str, Image.Image]],
        pred_imgs: List[Union[str, Image.Image]],
    ) -> List[Dict[str, float]]:
        '''
        Warning: this function don't improve performance. The underlying libraries still work serially.
        Returns per-pair results in the same order.
        '''
        assert len(org_imgs) == len(pred_imgs), "org_imgs and pred_imgs must have the same length"

        results: List[Dict[str, float]] = []
        for org, pred in zip(org_imgs, pred_imgs):
            results.append(self.score(org, pred))

        assert len(results) == len(org_imgs)
        return results
