import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from vision_unlearning.metrics.image_and_image import MetricImageImage


def random_image(size=(64, 64, 3)):
    return (np.random.rand(*size) * 255).astype(np.uint8)


def make_temp_png(img_array: np.ndarray) -> str:
    """Write a numpy array as PNG to a named temp file and return its path.

    Uses delete=False so Windows does not hold a lock that prevents PIL from
    writing to the same path.  Caller is responsible for os.unlink().
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(img_array).save(tmp.name, format="PNG")
    return tmp.name


# ---------------------------------------------------------------------------
# Regression: score() must accept PIL.Image objects on Windows without crashing
# (NamedTemporaryFile with default delete=True held an exclusive lock on Windows,
#  making PIL's save-by-name fail with PermissionError)
# ---------------------------------------------------------------------------

def test_score_pil_images_no_crash():
    """score(PIL.Image, PIL.Image) must not raise on any platform (regression)."""
    metric = MetricImageImage(metrics=['rmse', 'ssim'])
    img1 = Image.fromarray(random_image())
    img2 = Image.fromarray(random_image())
    result = metric.score(img1, img2)
    assert set(result.keys()) == {'rmse', 'ssim'}
    for v in result.values():
        assert isinstance(v, float)


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

    # Use delete=False so Windows doesn't hold an exclusive lock.
    tmp1 = make_temp_png(img1)
    tmp2 = make_temp_png(img2)
    try:
        score_from_path = metric.score(tmp1, tmp2)
        score_from_image = metric.score(Image.fromarray(img1), Image.fromarray(img2))

        # Values should be the same
        for k in score_from_path:
            assert abs(score_from_path[k] - score_from_image[k]) < 1e-6
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)


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

    tmp = make_temp_png(img_arr1)
    try:
        images1 = [tmp, img_pil2]
        images2 = [img_pil2, img_arr1]

        batch_scores = metric.score_batch(images1, images2)
        assert len(batch_scores) == 2
        for score in batch_scores:
            for k in ['rmse', 'ssim']:
                assert isinstance(score[k], float)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# The pixel range rmse and ssim normalise by. This fails on the code before the change, which always
# used the underlying library's 12-bit default of 4095 and so reported a full black-to-white
# difference on 8-bit images as 255 / 4095 = 0.0623.
# ---------------------------------------------------------------------------

def test_a_full_scale_difference_is_one_on_the_eight_bit_scale():
    black = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
    eight_bit = MetricImageImage(metrics=['rmse', 'ssim'], max_pixel_value=255).score(black, white)
    assert eight_bit['rmse'] == pytest.approx(1.0)
    assert eight_bit['ssim'] == pytest.approx(0.0, abs=1e-3)


def test_the_default_keeps_the_library_scale():
    black = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
    assert MetricImageImage(metrics=['rmse']).score(black, white)['rmse'] == pytest.approx(255 / 4095, rel=1e-5)
