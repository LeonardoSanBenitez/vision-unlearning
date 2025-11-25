from __future__ import annotations
import os
import sys
import time
import copy
import logging
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any, cast
from PIL import Image


import torch  # noqa: F401
from torchvision import transforms
import torch.nn as nn
from pydantic import Field
from omegaconf import OmegaConf
from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from huggingface_hub.repocard_data import EvalResult
from huggingface_hub import upload_folder
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

from vision_unlearning.unlearner.base import Unlearner, logger
from vision_unlearning.evaluator import EvaluatorTextToImage
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.utils.model_management import save_model_card


class MACE(Unlearner):
    '''
    Mass Concept Editing for applying of more than 2 concepts at a time.
    Adapted From:-
        Github:- https://github.com/Shilin-LU/MACE/tree/main
        Arxiv:- https://arxiv.org/abs/2403.06135
        Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong (2024).
        MACE: Mass Concept Erasure in Diffusion Models
        Accepted by Computer Vision and Pattern Recognition(CVPR) 2024.
    uses finetuning framework for task of mass concept erasure.
    '''

