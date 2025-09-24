from abc import ABC, abstractmethod
from pydantic import BaseModel
from vision_unlearning.utils.logger import get_logger


logger = get_logger('unlearner')


class Unlearner(BaseModel, ABC):
    '''
    performs the actual finetuning

    One unlearner may have variations/parametrizations that correspond to different unlearning algorithms/methods
    '''

    @abstractmethod
    def train(self):
        pass
