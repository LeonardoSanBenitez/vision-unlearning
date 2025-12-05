import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
from vision_unlearning.metrics import MetricImageTextSimilarity

# Helper: create random image
def random_image(width=64, height=64):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return arr

def test_score_consistency():
    metric = MetricImageTextSimilarity(metrics=['clip'])

    arr = random_image()
    img_pil = Image.fromarray(arr)
    text = "A test prompt"

    # Score from Image.Image
    score_img = metric.score(img_pil, text)

    # Score from np.ndarray
    score_arr = metric.score(arr, text)

    # They should be equal (or nearly equal)
    for k in score_img:
        assert abs(score_img[k] - score_arr[k]) < 1e-5

def test_score_from_file():
    metric = MetricImageTextSimilarity(metrics=['clip'])

    arr = random_image()
    text = "A test prompt"

    # Save temporary image
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img_pil = Image.fromarray(arr)
        img_pil.save(tmp, format="PNG")
        tmp.flush()

        score_file = metric.score(tmp.name, text)
        score_arr = metric.score(arr, text)

        for k in score_file:
            assert abs(score_file[k] - score_arr[k]) < 1e-5

def test_score_batch_consistency():
    metric = MetricImageTextSimilarity(metrics=['clip'])

    texts = ["prompt one", "prompt two", "prompt three"]
    images = [Image.fromarray(random_image()) for _ in texts]

    # Single score vs batch score
    single_scores = [metric.score(img, txt) for img, txt in zip(images, texts)]
    batch_scores = metric.score_batch(images, texts)

    for s, b in zip(single_scores, batch_scores):
        for k in s:
            assert abs(s[k] - b[k]) < 1e-5

def test_score_batch_from_mixed_input():
    metric = MetricImageTextSimilarity(metrics=['clip'])

    texts = ["prompt one", "prompt two"]
    # First image: np.ndarray, second: PIL.Image
    arr = random_image()
    img_pil = Image.fromarray(random_image())
    images = [arr, img_pil]

    batch_scores = metric.score_batch(images, texts)
    single_scores = [metric.score(img, txt) for img, txt in zip(images, texts)]

    for s, b in zip(single_scores, batch_scores):
        for k in s:
            assert abs(s[k] - b[k]) < 1e-5

def test_score_batch_from_file_and_array():
    metric = MetricImageTextSimilarity(metrics=['clip'])

    texts = ["prompt one", "prompt two"]

    arr1 = random_image()
    arr2 = random_image()
    img2 = Image.fromarray(arr2)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(arr1).save(tmp, format="PNG")
        tmp.flush()
        images = [tmp.name, img2]

        batch_scores = metric.score_batch(images, texts)
        single_scores = [metric.score(img, txt) for img, txt in zip(images, texts)]

        for s, b in zip(single_scores, batch_scores):
            for k in s:
                assert abs(s[k] - b[k]) < 1e-5
