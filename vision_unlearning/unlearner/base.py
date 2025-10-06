from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
from pathlib import Path
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

class UCEUnlearner(Unlearner):
    model_id:str = "CompVis/stable-diffusion-v1-4"
    device = 'cuda:0'
    erase_scale : int = 1
    preserve_scale : int = 1
    lamb : float = 0.5
    save_dir = '../uce_models'
    exp_name = None
    edit_concepts : str = None
    guide_concepts : str = None
    preserve_concepts : str = None
    concept_type : str = "object" #or "art"
    expand_prompts : str = "true" #or "false"

    def train(self):
        torch_dtype = torch.float32
        Path(self.save_dir).mkdir(parents = True,exists_ok = True)

        if self.exp_name == None:
            self.exp_name = 'uce_test'
        
        self.edit_concepts = [concept.strip() for concept in self.edit_concepts.split(';')]

        if self.guide_concepts == None:
            self.guide_concepts = ''
            if self.concept_type == "art":
                self.guide_concepts = "art"
        
        self.guide_concepts = [concept.strip() for concept in  self.guide_concepts.split(';')]

        if len(self.guide_concepts) == 1:
            self.guide_concepts = self.guide_concepts * len(self.edit_concepts)
        
        if len(self.guide_concepts) != len(self.edit_concepts):
            raise Exception("Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are seperated by ; and are of equal sizes")

        if self.preserve_concepts == None:
            self.preserve_concepts = []
        else:
            self.preserve_concepts = [concept.strip() for concept in self.preserve_concepts.split(';')]
        
        if self.expand_prompts == "true":
            edit_concepts_ = copy.deepcopy(self.edit_concepts)
            guide_concepts_ = copy.deepcopy(self.guide_concepts)

            for (concept,guide_concept) in zip(edit_concepts_, guide_concepts_):
                if self.concept_type == 'art':
                    self.edit_concepts.extend([f'painting by {concept}',
                                               f'art by {concept}',
                                               f'artwork by {concept}',
                                               f'picture by {concept}',
                                               f'style of {concept}'])
                    self.guide_concepts.extend([f'painting by {guide_concept}',
                                               f'art by {guide_concept}',
                                               f'artwork by {guide_concept}',
                                               f'picture by {guide_concept}',
                                               f'style of {guide_concept}'])
                
                else:
                    self.edit_concepts.extend([f'image of {concept}',
                                               f'photo of {concept}',
                                               f'portrait of {concept}',
                                               f'picture of {concept}',
                                               f'painting of {concept}'])
                    self.guide_concepts.extend([f'image of {guide_concept}',
                                               f'photo of {guide_concept}',
                                               f'portrait of {guide_concept}',
                                               f'picture of {guide_concept}',
                                               f'painting of {guide_concept}'])
        
        print(f"\n\nErasing: {self.edit_concepts}\n")
        print(f"Guiding: {self.guide_concepts}\n")
        print(f"Preserving: {self.preserve_concepts}\n")

        pipe = DiffusionPipeline.from_pretrained(self.model_id,torch_dtype = torch_dtype,
                                                 safety_checker = None,
                                                 vae = None).to(self.device)
        
        uce_run(pipe,self.edit_concepts,self.guide_concepts,self.preserve_concepts,self.erase_scale,self.preserve_scale,self.lamb,self.save_dir,self.exp_name,self.device,torch_dtype)

        return os.path.join(self.save_dir,f"{self.exp_name}.safetensors")


        

                    
        
        

        


