import pytest
from PIL import Image
import numpy as np
import tempfile

from vision_unlearning.metrics.image_and_image import MetricImageImage


def random_image(size=(64, 64, 3)):
    return (np.random.rand(*size) * 255).astype(np.uint8)


def test_score_consistency():
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    
    img1 = Image.fromarray(random_image())
    img2 = Image.fromarray(random_image())
    
    score1 = metric.score(img1, img2)
    score2 = metric.score(img1, img2)
    
    # Should be deterministic
    for k in score1:
        assert k in score2
        assert isinstance(score2[k], float)


def test_score_from_file():
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    
    img1 = random_image()
    img2 = random_image()
    
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp1, tempfile.NamedTemporaryFile(suffix=".png") as tmp2:
        Image.fromarray(img1).save(tmp1, format="PNG")
        tmp1.flush()
        Image.fromarray(img2).save(tmp2, format="PNG")
        tmp2.flush()
        
        score_from_path = metric.score(tmp1.name, tmp2.name)
        score_from_image = metric.score(Image.fromarray(img1), Image.fromarray(img2))
        
        # Values should be the same
        for k in score_from_path:
            assert abs(score_from_path[k] - score_from_image[k]) < 1e-6


def test_score_batch_consistency():
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    
    images1 = [Image.fromarray(random_image()) for _ in range(3)]
    images2 = [Image.fromarray(random_image()) for _ in range(3)]
    
    batch_scores = metric.score_batch(images1, images2)
    
    assert len(batch_scores) == len(images1)
    for score in batch_scores:
        for k in ['rmse', 'ssim']:
            assert k in score
            assert isinstance(score[k], float)


def test_score_batch_from_mixed_input():
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    
    img_arr = random_image()
    img_pil = Image.fromarray(random_image())
    
    images1 = [img_arr, img_pil]
    images2 = [img_pil, img_arr]
    
    batch_scores = metric.score_batch(images1, images2)
    assert len(batch_scores) == 2
    for score in batch_scores:
        for k in ['rmse', 'ssim']:
            assert isinstance(score[k], float)


def test_score_batch_from_file_and_array():
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    
    img_arr1 = random_image()
    img_arr2 = random_image()
    img_pil2 = Image.fromarray(img_arr2)
    
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(img_arr1).save(tmp, format="PNG")
        tmp.flush()
        
        images1 = [tmp.name, img_pil2]
        images2 = [img_pil2, img_arr1]
        
        batch_scores = metric.score_batch(images1, images2)
        assert len(batch_scores) == 2
        for score in batch_scores:
            for k in ['rmse', 'ssim']:
                assert isinstance(score[k], float)
