from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from huggingface_hub.repocard_data import EvalResult
from vision_unlearning.utils.logger import get_logger


logger = get_logger('unlearner')


class Unlearner(BaseModel, ABC):
    '''
    performs the actual finetuning

    One unlearner may have variations/parametrizations that correspond to different unlearning algorithms/methods
    '''

    @abstractmethod
    def train(self) -> List[EvalResult]:
        pass
