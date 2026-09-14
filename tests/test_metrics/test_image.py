'''
Every test here scores an image with `MetricQuality`, which calls `piq.brisque`. On its first call
in a fresh environment, `piq` downloads its support-vector-regression weights from a GitHub release
(`piq/brisque.py::_score_svr` -> `torch.hub.load_state_dict_from_url`) into the torch hub cache. The
download is inside the dependency, not in this file or in `vision_unlearning`, so it cannot be
avoided without vendoring third-party weights.

That makes these tests genuinely network-dependent, and they fail when GitHub's release endpoint is
briefly unavailable -- observed as `HTTP Error 503` and `RemoteDisconnected` on Python 3.10 while the
identical code passed on 3.12 in the same run. They are therefore marked `network`: still run, still
reported, but not part of the merge gate. See CONTRIBUTING.md section 6.
'''
import pytest
from PIL import Image
import numpy as np
import tempfile
from vision_unlearning.metrics.image import MetricQuality

pytestmark = pytest.mark.network


def random_image(size=(64, 64, 3)):
    return (np.random.rand(*size) * 255).astype(np.uint8)


def test_score_consistency():
    metric = MetricQuality()

    img = Image.fromarray(random_image())
    score1 = metric.score(img)
    score2 = metric.score(img)

    # Should be deterministic
    assert "brisque" in score1
    assert isinstance(score1["brisque"], float)
    assert abs(score1["brisque"] - score2["brisque"]) < 1e-6


def test_score_from_ndarray():
    metric = MetricQuality()

    arr = random_image()
    img_pil = Image.fromarray(arr)

    score_from_arr = metric.score(arr)
    score_from_pil = metric.score(img_pil)

    # Should produce same results
    assert abs(score_from_arr["brisque"] - score_from_pil["brisque"]) < 1e-6


def test_score_from_file():
    metric = MetricQuality()

    arr = random_image()
    img_pil = Image.fromarray(arr)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img_pil.save(tmp, format="PNG")
        tmp.flush()

        score_from_path = metric.score(tmp.name)
        score_from_pil = metric.score(img_pil)

        assert abs(score_from_path["brisque"] - score_from_pil["brisque"]) < 1e-6


def test_score_batch_consistency():
    metric = MetricQuality()

    images = [Image.fromarray(random_image()) for _ in range(3)]
    batch_scores = metric.score_batch(images)

    assert len(batch_scores) == len(images)
    for score in batch_scores:
        assert "brisque" in score
        assert isinstance(score["brisque"], float)


def test_score_batch_mixed_inputs():
    metric = MetricQuality()

    arr = random_image()
    img_pil = Image.fromarray(random_image())

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(arr).save(tmp, format="PNG")
        tmp.flush()

        images = [tmp.name, img_pil, arr]
        batch_scores = metric.score_batch(images)

        assert len(batch_scores) == 3
        for score in batch_scores:
            assert "brisque" in score
            assert isinstance(score["brisque"], float)
