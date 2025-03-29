import warnings
import pytest
import torch
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodNone


def test_none():
    method = GradientWeightingMethodNone()
    grads_forget = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    grads_retain = [torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])]
    result = method.weight_grads(grads_forget, grads_retain, None)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0]))
