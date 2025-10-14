from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
import copy
from diffusers import DiffusionPipeline
import os


class Unlearner(BaseModel, ABC):
    """
    Performs the actual finetuning.

    One unlearner may have variations/parametrizations
    that correspond to different unlearning algorithms/methods.
    """

    # TODO: define what is shared among all unlearners

    @abstractmethod
    def train(self) -> None:
        """Abstract training method to be implemented by subclasses."""
        pass
