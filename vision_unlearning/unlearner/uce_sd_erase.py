'''
Taken the reference of Unified Concept Editing Github Repo:
Github repository:- https://github.com/rohitgandikota/unified-concept-editing
Arxiv Paper:- https://arxiv.org/pdf/2308.14761.pdf
'''


#Import formatted
import os
import time
import copy
import logging
from pathlib import Path
from enum import Enum
from typing import Optional
import torch #noqa: F401(imported but unused)
from pydantic import Field
from safetensors.torch import save_file
from diffusers import DiffusionPipeline

import base

class ConceptType(str, Enum):
    Object = "object"
    Art = "art"

class UCE(base.Unlearner):
    pretrained_model_name_or_path:str = Field(default="CompVis/stable-diffusion-v1-4", description="Path to pretrained model or model identifier from huggingface.co/models.") #noqa : E501
    device: str = 'cuda:0'
    erase_scale : int = 1
    preserve_scale : int = 1
    lamb : float = 0.5
    output_dir: str = Field(default='../uce_models', description="Output directory for model predictions and checkpoints.")
    edit_concepts : Optional[str] = None
    guide_concepts : Optional[str] = None
    preserve_concepts : Optional[str] = None
    concept_type : ConceptType = Field(default=ConceptType.Object, description="Type of concept to unlearn")
    expand_prompts : bool = True #or false

    def train(self) -> None:
        torch_dtype :torch.dtype = torch.float32
        Path(self.output_dir).mkdir(parents = True,exist_ok = True)

        if self.pretrained_model_name_or_path != "CompVis/stable-diffusion-v1-4" :
            logging.warning("UCE was not tested with this base model, we do not ensure correct working.")
        
        self.edit_concepts = [concept.strip() for concept in self.edit_concepts.split(';')]

        if self.guide_concepts == None:
            self.guide_concepts = ''
            if self.concept_type.value == "art":
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
        
        if self.expand_prompts == True:
            edit_concepts_ = copy.deepcopy(self.edit_concepts) #type : ignore
            guide_concepts_ = copy.deepcopy(self.guide_concepts)#type : ignore

            for (concept,guide_concept) in zip(edit_concepts_, guide_concepts_):
                if self.concept_type.value == 'art':
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

        pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path,torch_dtype = torch_dtype,
                                                 safety_checker = None,
                                                 vae = None).to(self.device) # type: ignore
        
        uce_run(pipe,self.edit_concepts,self.guide_concepts,self.preserve_concepts,self.erase_scale,self.preserve_scale,self.lamb,self.output_dir,self.device,torch_dtype) #noqa : E501



def collect_text_embeddings(pipe, concepts, device, torch_dtype)-> dict[str,torch.Tensor]:
    """Return dict {concept: last_token_embedding}."""
    uce_embeds = {}
    for e in concepts:
        if e in uce_embeds:
            continue
        t_emb = pipe.encode_prompt(
            prompt=e,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

        last_token_idx = (
            pipe.tokenizer(
                e,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['attention_mask']
        ).sum() - 2

        uce_embeds[e] = t_emb[0][:, last_token_idx, :] # noqa : E501
    return uce_embeds

def collect_guide_outputs(concepts, embeds, modules):
    """
    Collect cross attention outputs for guide/preserve concepts.
    Returns dict {concept: [outputs per module]}.
    """
    outputs = {}
    for g in concepts:
        if g in outputs:
            continue
        t_emb = embeds[g]
        for module in modules:
            outputs[g] = outputs.get(g, []) + [module(t_emb)]
    return outputs

def update_weights(original_modules, erase_embeds, guide_outputs,
                   edit_concepts, guide_concepts, preserve_concepts,
                   erase_scale, preserve_scale, lamb, device, torch_dtype):
    """Apply the UCE weight update to each module and return new modules."""
    uce_modules = copy.deepcopy(original_modules)# type : ignore

    for module_idx, module in enumerate(original_modules):
        w_old = module.weight

        mat1 = lamb * w_old
        mat2 = lamb * torch.eye(w_old.shape[1], device=device, dtype=torch_dtype)

        # Erase Concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = erase_embeds[erase_concept].T
            v_i_star = guide_outputs[guide_concept][module_idx].T

            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)

        # Preserve Concepts
        for preserve_concept in preserve_concepts:
            c_i = erase_embeds[preserve_concept].T
            v_i_star = guide_outputs[preserve_concept][module_idx].T

            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)

        uce_modules[module_idx].weight = torch.nn.Parameter(
            mat1 @ torch.inverse(mat2.float()).to(torch_dtype)
        )

    return uce_modules

def save_uce_weights(uce_modules, uce_module_names, save_dir):
    """Save updated module weights to a safetensors file."""
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name + '.weight'] = parameter.weight
    # save_file(uce_state_dict, os.path.join(save_dir, exp_name + '.safetensors'))

def uce_run(pipe, edit_concepts, guide_concepts, preserve_concepts,
        erase_scale, preserve_scale, lamb, save_dir,
        device="cuda:0", torch_dtype=torch.float32):
    torch.set_grad_enabled(False)
    start_time = time.time()

    # Find relevant modules
    uce_modules, uce_module_names = [], []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules) #type : ignore

    # 1. collect embeddings
    all_concepts = edit_concepts + guide_concepts + preserve_concepts
    erase_embeds = collect_text_embeddings(pipe, all_concepts, device, torch_dtype)

    # 2. collect guide outputs
    guide_outputs = collect_guide_outputs(guide_concepts + preserve_concepts,
                                          erase_embeds, original_modules)

    # 3. apply weight updates
    updated_modules = update_weights(
        original_modules, erase_embeds, guide_outputs,
        edit_concepts, guide_concepts, preserve_concepts,
        erase_scale, preserve_scale, lamb,
        device, torch_dtype
    )

    # 4. save weights
    save_uce_weights(updated_modules, uce_module_names, save_dir)

    end_time = time.time()
    print(f"\n\nErased concepts using UCE\nModel edited in {end_time-start_time:.2f} seconds\n")

def main():
    uce = UCE(model_id='CompVis/stable-diffusion-v1-4',edit_concepts='Van Gogh; Picasso',guide_concepts='art',preserve_concepts='Monet; Rembrandt; Warhol',device='cuda:0',concept_type=ConceptType.Art) 

    uce.train()


if __name__ == "__main__":
    main()

