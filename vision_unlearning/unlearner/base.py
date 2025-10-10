from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
import copy
from uce_sd_erase import uce_run 
from diffusers import DiffusionPipeline
import os





class Unlearner(BaseModel, ABC):
    '''
    performs the actual finetuning

    One unlearner may have variations/parametrizations that correspond to different unlearning algorithms/methods
    '''
    # TODO: what is shared among all unlearners?

    @abstractmethod
    def train(self):
        pass


        


        

                    
        
        

        


