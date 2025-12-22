# from __future__ import annotations
# import os
# import sys
# import time
# import copy
# import logging
# from pathlib import Path
# from enum import Enum
# from typing import Optional, List, Tuple, Dict, Any, cast
# from PIL import Image


# import torch  # noqa = F401
# from torchvision import transforms
# import torch.nn as nn
# from pydantic import Field
# from omegaconf import OmegaConf
# from inference import inference
# from safetensors.torch import save_file, load_file
# from diffusers import DiffusionPipeline, AutoPipelineForText2Image
# from huggingface_hub.repocard_data import EvalResult
# from huggingface_hub import upload_folder
# from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

# from vision_unlearning.unlearner.base import Unlearner, logger
# from vision_unlearning.evaluator import EvaluatorTextToImage
# from vision_unlearning.metrics import MetricImageTextSimilarity
# from vision_unlearning.mace_config import MACEConfig 
# from vision_unlearning.utils.model_management import save_model_card
# from vision_unlearning.utils.logger import get_logger

import os
import torch
import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from PIL import Image
import json
import random
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import cv2
from pydantic import Field
from argparse import Namespace
from pathlib import Path
import logging
import math
import copy
import numpy as np

from vision_unlearning.unlearner.base import Unlearner
from vision_unlearning.utils.logger import get_logger
from vision_unlearning.evaluator import EvaluatorTextToImage
from vision_unlearning.metrics import MetricImageTextSimilarity
from vision_unlearning.utils.attention_manipulation import *
from vision_unlearning.utils.segmentation import *
from vision_unlearning.utils.prompt_augmentation import mace_prompt_augmentation, text_augmentation, clean_prompt


from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image, AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from huggingface_hub.repocard_data import EvalResult
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


logger = get_logger("MACE")

# cfr_lora_training
def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    instance_prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["preserve_prompt_ids"] for example in examples]
        pixel_values += [example["preserve_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if masks[0] is not None: 
        # object/celebrity erasure
        masks = torch.stack(masks)
    else:
        # artistic style erasure
        masks = None

    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "masks": masks,
        "concept_positions": concept_positions,
    }
    return batch

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    if text_encoder_config.architectures is not None:
        model_class = text_encoder_config.architectures[0]
    else:
        print("Items model class is None")
        

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

# cfr_utils.py
def importance_sampling_fn(t, temperature=0.05):
    """Importance Sampling Function f(t)"""
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))

def prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, test_set,
                tokenizer, with_to_k=True, all_words=False, prepare_k_v_for_lora=False):

    with torch.no_grad():
        all_contexts, all_valuess = [], []
     
        for curr_item in test_set:
            gc.collect()
            torch.cuda.empty_cache()
   
            # restart LDM parameters
            num_ca_clip_layers = len(ca_layers)
            for idx_, l in enumerate(ca_layers):
                l.to_v = copy.deepcopy(og_matrices[idx_])
                projection_matrices[idx_] = l.to_v
                if with_to_k:
                    l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                    projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    
            old_embs, new_embs = [], []
            extended_old_indices, extended_new_indices = [], []
    
            # indetify corresponding destinations for each token in old_emb
            # Bulk tokenization
            texts_old = [item[0] for item in curr_item["old"]]
            texts_new = [item[0] for item in curr_item["new"]]
            texts_combined = texts_old + texts_new

            tokenized_inputs = tokenizer(
                texts_combined,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
    
            # Text embeddings
            text_embeddings = text_encoder(tokenized_inputs.input_ids.to(text_encoder.device))[0]
            old_embs.extend(text_embeddings[:len(texts_old)])
            new_embs.extend(text_embeddings[len(texts_old):])

            # Find matching indices
            for old_text, new_text in zip(texts_old, texts_new):
                tokens_a = tokenizer(old_text).input_ids
                tokens_b = tokenizer(new_text).input_ids
         
                old_indices, new_indices = find_matching_indices(tokens_a, tokens_b)
         
                if old_indices[-1] >= new_indices[-1]:
                    extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77)))
                    extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77 - (old_indices[-1] - new_indices[-1]))))
                else:
                    extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77)))
                    extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77 - (new_indices[-1] - old_indices[-1]))))

            # prepare batch: for each pair of setences, old context and new values
            contexts, valuess = [], []
            if not all_words:
                for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                    context = old_emb[extended_old_indices[idx]].detach()
                    values = []
                    for layer in projection_matrices:
                        values.append(layer(new_emb[extended_new_indices[idx]]).detach())
                    contexts.append(context)
                    valuess.append(values)
     
                all_contexts.append(contexts)
                all_valuess.append(valuess)
            else:
                if prepare_k_v_for_lora:
                    # prepare for lora, then no need to use new_emb
                    for idx, old_emb in enumerate(old_embs):
                        context = old_emb.detach()
                        values = []
                        for layer in projection_matrices:
                            values.append(layer(old_emb).detach())
                        contexts.append(context)
                        valuess.append(values)
                else:
                    # need to use new_emb
                    for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                        context = old_emb.detach()
                        values = []
                        for layer in projection_matrices:
                            values.append(layer(new_emb).detach())
                        contexts.append(context)
                        valuess.append(values)
    
                all_contexts.append(contexts)
                all_valuess.append(valuess)
    
        return all_contexts, all_valuess

def closed_form_refinement(projection_matrices, all_contexts=None, all_valuess=None, lamb=0.5,
                           preserve_scale=1, cache_dict=None, cache_dict_path=None, cache_mode=False):

    with torch.no_grad():
        if cache_dict_path is not None:
            cache_dict = torch.load(cache_dict_path, map_location=projection_matrices[0].weight.device)
    
        for layer_num in tqdm(range(len(projection_matrices))):
            gc.collect()
            torch.cuda.empty_cache()
     
            mat1 = lamb * projection_matrices[layer_num].weight
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=projection_matrices[layer_num].weight.device)
  
            total_for_mat1 = torch.zeros_like(projection_matrices[layer_num].weight)
            total_for_mat2 = torch.zeros_like(mat2)

            if all_contexts is not None and all_valuess is not None:
                for contexts, valuess in zip(all_contexts, all_valuess):
                    # Convert contexts and values to tensors
                    contexts_tensor = torch.stack(contexts, dim=2)
                    values_tensor = torch.stack([vals[layer_num] for vals in valuess], dim=2)
             
                    # Aggregate sums for mat1, mat2 using matrix multiplication
                    for_mat1 = torch.bmm(values_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0)
                    for_mat2 = torch.bmm(contexts_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0)
              
                    total_for_mat1 += for_mat1
                    total_for_mat2 += for_mat2

                del for_mat1, for_mat2
         
            if cache_mode:
                # cache the results
                if cache_dict[f'{layer_num}_for_mat1'] is None:
                    cache_dict[f'{layer_num}_for_mat1'] = total_for_mat1
                    cache_dict[f'{layer_num}_for_mat2'] = total_for_mat2
                else:
                    cache_dict[f'{layer_num}_for_mat1'] += total_for_mat1
                    cache_dict[f'{layer_num}_for_mat2'] += total_for_mat2
            else:
                # CFR calculation
                if cache_dict_path is not None or cache_dict is not None:
                    total_for_mat1 += preserve_scale * cache_dict[f'{layer_num}_for_mat1']
                    total_for_mat2 += preserve_scale * cache_dict[f'{layer_num}_for_mat2']
             
                total_for_mat1 += mat1
                total_for_mat2 += mat2
           
                projection_matrices[layer_num].weight.data = total_for_mat1 @ torch.inverse(total_for_mat2)
         
            del total_for_mat1, total_for_mat2

# dataset.py
# TODO:- MACE should be using class UnlearnDatasetSplit
class MACEDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        size=512,
        center_crop=False,
        use_pooler=False,
        multi_concept=None,
        mapping=None,
        augment=True,
        batch_size=None,
        with_prior_preservation=False,
        preserve_info=None,
        num_class_images=None,
        train_seperate=False,
        aug_length=50,
        prompt_len=250,
        input_data_path=None,
        use_gpt=False,
    ):
        self.with_prior_preservation = with_prior_preservation
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.batch_counter = 0
        self.batch_size = batch_size
        self.concept_number = 0
        self.train_seperate = train_seperate
        self.aug_length = aug_length

        self.all_concept_image_path = []
        self.all_concept_mask_path = []
        single_concept_images_path = []
        self.instance_prompt = []
        self.target_prompt = []

        self.num_instance_images = 0
        self.dict_for_close_form = []
        self.class_images_path = []

        for concept_idx, (data, mapping_concept) in enumerate(zip(multi_concept, mapping)):
            c, t = data
    
            if input_data_path is not None:
                p = Path(os.path.join(input_data_path, c.replace("-", " ")))
                if not p.exists():
                    raise ValueError(f"Instance {p} images root doesn't exists.")
           
                if t == "object":
                    p_mask = Path(os.path.join(input_data_path, c.replace("-", " ")).replace(f'{c.replace("-", " ")}', f'{c.replace("-", " ")} mask'))
                    if not p_mask.exists():
                        raise ValueError(f"Instance {p_mask} images root doesn't exists.")
            else:
                raise ValueError(f"Input data path is not provided.")    

            image_paths = sorted(list(p.iterdir()))
            single_concept_images_path = []
            single_concept_images_path += image_paths
            self.all_concept_image_path.append(single_concept_images_path)
     
            if t == "object":
                mask_paths = sorted(list(p_mask.iterdir()))
                single_concept_masks_path = []
                single_concept_masks_path += mask_paths
                self.all_concept_mask_path.append(single_concept_masks_path)
             
            erased_concept = c.replace("-", " ")
    
            if use_gpt:
                class_prompt_collection, mapping_prompt_collection = text_augmentation(erased_concept, mapping_concept, t, num_text_augmentations=self.aug_length)
                self.instance_prompt.append(class_prompt_collection)
                self.target_prompt.append(mapping_prompt_collection)
            else:
                sampled_indices = random.sample(range(0, prompt_len), self.aug_length)
                self.instance_prompt.append(mace_prompt_augmentation(erased_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
                self.target_prompt.append(mace_prompt_augmentation(mapping_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
       
            self.num_instance_images += len(single_concept_images_path)
    
            entry = {"old": self.instance_prompt[concept_idx], "new": self.target_prompt[concept_idx]}
            self.dict_for_close_form.append(entry)
    
        if with_prior_preservation:
            class_data_root = Path(preserve_info['dataset_retain_name'])
            if os.path.isdir(class_data_root):
                class_images_path = list(class_data_root.iterdir())
                class_prompt = [preserve_info["preserve_prompt"] for _ in range(len(class_images_path))]
            else:
                with open(class_data_root, "r") as f:
                    class_images_path = f.read().splitlines()
                with open(preserve_info["preserve_prompt"], "r") as f:
                    class_prompt = f.read().splitlines()
        
            class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])
             
        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
        self._concept_num = len(self.instance_prompt)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images // self._concept_num, self.num_class_images)
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
    
        if not self.train_seperate:
            if self.batch_counter % self.batch_size == 0:
                self.concept_number = random.randint(0, self._concept_num - 1)
            self.batch_counter += 1
        
        instance_image = Image.open(self.all_concept_image_path[self.concept_number][index % self._length])
    
        if len(self.all_concept_mask_path) == 0:
            # artistic style erasure
            binary_tensor = None
        else:
            # object/celebrity erasure
            instance_mask = Image.open(self.all_concept_mask_path[self.concept_number][index % self._length])
            instance_mask = instance_mask.convert('L')
            trans = transforms.ToTensor()
            binary_tensor = trans(instance_mask)
    
        prompt_number = random.randint(0, len(self.instance_prompt[self.concept_number]) - 1)
        instance_prompt, target_tokens = self.instance_prompt[self.concept_number][prompt_number]
    
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_masks"] = binary_tensor

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids           

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1] * len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]             

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["preserve_images"] = self.image_transforms(class_image)
            example["preserve_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
         
        return example


class MACEUnlearner(Unlearner):
    '''
    Mass Concept Editing for applying of more than 2 concepts at a time.
    Adapted From =-
        Github =- https =//github.com/Shilin-LU/MACE/tree/main
        Arxiv =- https =//arxiv.org/abs/2403.06135
        Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong (2024).
        MACE = Mass Concept Erasure in Diffusion Models
        Accepted by Computer Vision and Pattern Recognition(CVPR) 2024.
    uses finetuning framework for task of mass concept erasure.
    '''
    def __init__(self,multi_concept:List[List[Tuple[str,str]]], mapping_concept:List[str], device:str,userpooler:bool, train_batch_size:int, learning_rate:float,max_train_steps:int, train_preserve_scale:float, fuse_preserve_scale:float,augment:bool, lamb:float,rank:int, lora:bool, train_seperate:bool, importance_sampling:bool, max_memory:int, aug_length:int, prompt_len:int,all_words:bool, use_gpt:bool, prior_preservation_cache_path:str, domain_preservation_cache_path:str, preserve_weight:float, input_data_dir:str, output_dir:str, final_save_path:str, use_gsam_mask:bool, use_sam_hq:bool, grounded_config:Optional[str], grounded_checkpoint:Optional[str], sam_hq_checkpoint:Optional[str], sam_checkpoint:Optional[str], pretrained_model_name_or_path:str, preserve_prompt:List[str], forget_prompt:List[str], with_prior_preservation:bool, dataset_forget_name:str, dataset_retain_name:str, prior_loss_weight:float, with_uncond_loss:bool,negative_guidance:float, uncond_loss_weight:float, num_class_images:int, seed:int, resolution:int, revision:Optional[str], tokenizer_name:Optional[str], instance_prompt:Optional[str], concept_leyword:Optional[str], no_real_image:bool, center_crop:bool, train_text_encoder:bool, sample_batch_size:int, num_train_epochs:int, checkpointing_steps:int, resume_from_checkpoint:Optional[str], gradient_accumulation_steps:int, gradient_checkpointing:bool, scale_lr:bool, lr_Scheduler:str, lr_warmup_steps:int, lr_num_cycles:int, lr_power:float, use_8bit_adam:bool, dataloader_num_workers:int, adam_beta1:float, adam_beta2:float, adam_weight_decay:float, adam_epsilon:float, max_grad_norm:float, push_to_hub:bool, hub_token:Optional[str], hub_model_id:Optional[str], logging_dir:str, allow_tf32:bool,report_to:str, mixed_precision:Optional[str], prior_generation_precision:Optional[str], local_rank:int,enable_xformers_memory_efficient_attention:bool, set_grads_to_none:bool, save_entire_model:bool,generate_training_data:bool, compute_runtimes:bool):
        super().__init__()
        self.multi_concept = multi_concept
        self.mapping_concept = mapping_concept

        # -------------------------
        # DEVICE
        # -------------------------
        self.device = device

        # -------------------------
        # Primary Settings
        # -------------------------
        self.use_pooler = userpooler
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.train_preserve_scale = train_preserve_scale
        self.fuse_preserve_scale = fuse_preserve_scale
        self.augment = augment
        self.lamb = lamb
        self.rank = rank
        self.lora = lora
        self.train_separate = train_seperate
        self.importance_sampling = importance_sampling
        self.max_memory = max_memory
        self.aug_length = aug_length
        self.prompt_len = prompt_len
        self.all_words = all_words
        self.use_gpt = use_gpt

        # -------------------------
        # Cache / Preservation
        # -------------------------
        self.prior_preservation_cache_path = prior_preservation_cache_path
        self.domain_preservation_cache_path = domain_preservation_cache_path
        self.preserve_weight = preserve_weight

        # -------------------------
        # Paths
        # -------------------------
        self.input_data_dir = input_data_dir
        self.output_dir = output_dir
        self.final_save_path = final_save_path
        # -------------------------
        # Grounded-SAM
        # -------------------------
        self.use_gsam_mask = use_gsam_mask
        self.use_sam_hq = use_sam_hq
        self.grounded_config = grounded_config
        self.grounded_checkpoint = grounded_checkpoint
        self.sam_hq_checkpoint = sam_hq_checkpoint
        self.sam_checkpoint = sam_checkpoint

        # -------------------------
        # Diffusion / Model
        # -------------------------
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.with_prior_preservation = with_prior_preservation
        self.dataset_forget_path = dataset_forget_name
        self.dataset_retain_name = dataset_retain_name
        self.preserve_prompt = preserve_prompt
        self.forget_prompt = forget_prompt
        self.prior_loss_weight = prior_loss_weight
        self.with_uncond_loss = with_uncond_loss
        self.negative_guidance = negative_guidance
        self.uncond_loss_weight = uncond_loss_weight
        self.num_class_images = num_class_images
        self.seed = seed
        self.resolution = resolution
        self.revision = revision
        self.tokenizer_name = tokenizer_name
        self.instance_prompt = instance_prompt
        self.concept_keyword = concept_leyword
        self.no_real_image = no_real_image
        self.center_crop = center_crop
        self.train_text_encoder = train_text_encoder
        self.sample_batch_size = sample_batch_size
        self.num_train_epochs = num_train_epochs
        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_lr = scale_lr
        self.lr_scheduler = lr_Scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_num_cycles = lr_num_cycles
        self.lr_power = lr_power
        self.use_8bit_adam = use_8bit_adam
        self.dataloader_num_workers = dataloader_num_workers
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.push_to_hub = push_to_hub
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        self.logging_dir = logging_dir
        self.allow_tf32 = allow_tf32
        self.report_to = report_to
        self.mixed_precision = mixed_precision
        self.prior_generation_precision = prior_generation_precision
        self.local_rank = local_rank
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        self.set_grads_to_none = set_grads_to_none
        self.save_entire_model = save_entire_model
        self.generate_training_data = generate_training_data
        self.compute_runtimes = compute_runtimes

    '''
    def data_preparation(self) :
        logger.info("data_preparation function entered")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # generate 8 images per concept using the original model for performing erasure
        if self.generate_data :
            logger.info("Generating training images for MACE...")

            self.inference("CompVis/stable-diffusion-v1-4",True,self.multi_concept,self.device,30,self.input_data_dir)
            # inference({
            #     "pretrained_model_name_or_path" = "CompVis/stable-diffusion-v1-4",
            #     "multi_concept" = self.multi_concept,
            #     "generate_training_data" = True,
            #     "device" = self.device,
            #     "steps" = 30,
            #     "output_dir" = self.input_data_dir,
            # })
  
        # get and save masks for each image
        if self.use_gsam_mask :
            grounded_model = load_model(self.grounded_config, self.grounded_checkpoint, device = self.device)

            if self.use_sam_hq :
                predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.sam_hq_checkpoint).to(self.device))
            else :
                predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.sam_checkpoint).to(self.device))

            transform = transforms.ToTensor()
            for root,_,files in os.walk(self.input_data_dir) :
                mask_save_path = root.replace(f'{os.path.basename(root)}',f'{os.path.basename(root)} mask')
                os.makedirs(mask_save_path,exist_ok=True)
                for file in files :
                    file_path = os.path.join(root,file)
                    print(file_path)
                    # read images and get masks
                    image = Image.open(file_path)
                    if not image.mode == "RGB" :
                        image = image.convert("RGB")
                    tensor_image = transform(image).to(self.device)
                    GSAM_mask = get_mask(tensor_image,os.path.basename(root), grounded_model, predictor, self.device)
                    #save masks
                    GSAM_mask = (GSAM_mask.to(torch.uint8) * 255).squeeze()
                    save_mask = to_pil_image(GSAM_mask)
                    save_mask.save(f"{os.path.join(mask_save_path, file).replace('.jpg','_mask.jpg')}")

    def data_preparation_transformers(self) :
        logger.info("data_preparation_transformers function entered")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # generate 8 images per concept using the original model for performing erasure
        if self.generate_data :
            logger.info("Generating training images for MACE...")

            self.inference("CompVis/stable-diffusion-v1-4",True,self.multi_concept,self.device,30,self.input_data_dir)
            # inference({
            #     "pretrained_model_name_or_path" = "CompVis/stable-diffusion-v1-4",
            #     "multi_concept" = self.multi_concept,
            #     "generate_training_data" = True,
            #     "device" = self.device,
            #     "steps" = 30,
            #     "output_dir" = self.input_data_dir,
            # })
   
        #get and save mask for each image
        if self.use_gsam_mask :
            detector_id = "IDEA-Research/grounding-dino-base"
            segmenter_id = "facebook/sam-vit-huge"

            for root,_, files in os.walk(self.input_data_dir) :
                mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
                os.makedirs(mask_save_path, exist_ok=True)
                for file in files :
                    file_path = os.path.join(root, file)
                    print(file_path)
                    save_mask = grounded_segmentation(
                        image=file_path,
                        labels=os.path.basename(root),
                        threshold=0.3,
                        polygon_refinement=True,
                        detector_id=detector_id,
                        segmenter_id=segmenter_id
                    )
                    cv2.imwrite(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}", save_mask)
    '''
    def trasformer_gsam_util(self):
        logger.info("transformers_gasm function entered")
        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"

        transform = transforms.ToTensor()
        for root, _, files in os.walk(self.input_data_dir):
            mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
            os.makedirs(mask_save_path, exist_ok=True)
            for file in files:
                print(file, root)
                GSAM_mask = grounded_segmentation(
                    image=os.path.join(root, file),
                    labels='a person',
                    threshold=0.3,
                    polygon_refinement=True,
                    detector_id=detector_id,
                    segmenter_id=segmenter_id
                )

            cv2.imwrite(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}", GSAM_mask)

    def cfr_lora_training(self):
        logging_dir = Path(self.output_dir, self.logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_dir=logging_dir,
        )

        if self.train_text_encoder and self.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        # If passed along, set the training seed now.
        if self.seed is not None:
            set_seed(self.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

        # Load the tokenizer
        if self.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, revision=self.revision, use_fast=False)
        elif self.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.revision,
                use_fast=False,
            )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path, self.revision)

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.revision
        )
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision)
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet", revision=self.revision
        )
  
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)
  
        if self.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        if self.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
   
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.scale_lr:
            self.learning_rate = (
                self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
    
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        if self.with_prior_preservation:
            self.preservation_info = {
                    "preserve_prompt": self.preserve_prompt,
                    "preserve_data_dir": self.dataset_forget_name
                }
        else:
            self.preservation_info = None
 
        train_dataset = MACEDataset(
            tokenizer=tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
            use_pooler=self.use_pooler,
            multi_concept=self.multi_concept[0],
            mapping=self.mapping_concept,
            augment=self.augment,
            batch_size=self.train_batch_size,
            with_prior_preservation=self.with_prior_preservation,
            preserve_info=self.preservation_info,
            train_seperate=self.train_seperate,
            aug_length=self.aug_length,
            prompt_len=self.prompt_len,
            input_data_path=self.input_data_dir,
            use_gpt=self.use_gpt,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),
            num_workers=self.dataloader_num_workers,
        )
  
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
      
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
    
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)
       
        # Move vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # stage 1: closed-form refinement
        projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
 
        # to save memory
        CFR_dict = {}
        max_concept_num = self.max_memory  # the maximum number of concept that can be processed at once
        if len(train_dataset.dict_for_close_form) > max_concept_num:
      
            for layer_num in tqdm(range(len(projection_matrices))):
                CFR_dict[f'{layer_num}_for_mat1'] = None
                CFR_dict[f'{layer_num}_for_mat2'] = None
           
            for i in tqdm(range(0, len(train_dataset.dict_for_close_form), max_concept_num)):
                contexts_sub, valuess_sub = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                                        train_dataset.dict_for_close_form[i:i+5], tokenizer, all_words=self.all_words)
                closed_form_refinement(projection_matrices, contexts_sub, valuess_sub, cache_dict=CFR_dict, cache_mode=True)
                
                del contexts_sub, valuess_sub
                gc.collect()
                torch.cuda.empty_cache()
                
        else:
            for layer_num in tqdm(range(len(projection_matrices))):
                CFR_dict[f'{layer_num}_for_mat1'] = .0
                CFR_dict[f'{layer_num}_for_mat2'] = .0
      
            contexts, valuess = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices,
                                            train_dataset.dict_for_close_form, tokenizer, all_words=self.all_words)

        del ca_layers, og_matrices

        # Load cached prior knowledge for preserving
        if self.prior_preservation_cache_path:
            prior_preservation_cache_dict = torch.load(self.prior_preservation_cache_path, map_location=projection_matrices[0].weight.device)
        else:
            prior_preservation_cache_dict = {}
            for layer_num in self(range(len(projection_matrices))):
                prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
                prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
       
        # Load cached domain knowledge for preserving
        if self.domain_preservation_cache_path:
            domain_preservation_cache_dict = torch.load(self.domain_preservation_cache_path, map_location=projection_matrices[0].weight.device)
        else:
            domain_preservation_cache_dict = {}
            for layer_num in tqdm(range(len(projection_matrices))):
                domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
                domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0

        # integrate the prior knowledge, domain knowledge and closed-form refinement
        cache_dict = {}
        for key in CFR_dict:
            cache_dict[key] = self.train_preserve_scale * (prior_preservation_cache_dict[key] \
                            + self.preserve_weight * domain_preservation_cache_dict[key]) \
                            + CFR_dict[key]
 
        # closed-form refinement
        projection_matrices, _, _ = get_ca_layers(unet, with_to_k=True)
 
        if len(train_dataset.dict_for_close_form) > max_concept_num:
            closed_form_refinement(projection_matrices, lamb=self.lamb, preserve_scale=1, cache_dict=cache_dict)
        else:
            closed_form_refinement(projection_matrices, contexts, valuess, lamb=self.lamb, 
                                preserve_scale=self.train_preserve_scale, cache_dict=cache_dict)
        
        del contexts, valuess, cache_dict
        gc.collect()
        torch.cuda.empty_cache()
        
        # stage 2: multi-lora training
        for i in range(train_dataset._concept_num):  # the number of concept/lora
    
            attn_controller = AttnController()
            if i != 0:
                unet.set_default_attn_processor()
            for name, m in unet.named_modules():
                if name.endswith('attn2') or name.endswith('attn1'):
                    cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = unet.config.block_out_channels[block_id]

                    m.set_processor(LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=self.rank,
                        attn_controller=attn_controller,
                        module_name=name,
                        preserve_prior=self.with_prior_preservation,
                    ))

            # set lora
            # unet.set_attn_processor(lora_attn_procs)
            lora_attn_procs = {}
            for key, value in zip(unet.attn_processors.keys(), unet.attn_processors.values()):
                if key.endswith("attn2.processor"):
                    lora_attn_procs[f'{key}.to_k_lora'] = value.to_k_lora
                    lora_attn_procs[f'{key}.to_v_lora'] = value.to_v_lora
                    # lora_attn_procs[f'{key}.to_q_lora'] = value.to_q_lora
                    # lora_attn_procs[f'{key}.to_out_lora'] = value.to_out_lora
            
            lora_layers = AttnProcsLayers(lora_attn_procs)

            optimizer = optimizer_class(
                lora_layers.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.adam_weight_decay,
                eps=self.adam_epsilon,
            )
    
            lr_scheduler = get_scheduler(
                self.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
                num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
                num_cycles=self.lr_num_cycles,
                power=self.lr_power,
            )
    
            if self.train_text_encoder:
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoder, optimizer, train_dataloader, lr_scheduler
                )
            else:
                unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, optimizer, train_dataloader, lr_scheduler
                )

            # We need to initialize the trackers we use, and also store our configuration.
            # The trackers initializes automatically on the main process.
            if accelerator.is_main_process:
                accelerator.init_trackers("MACE")

            # Train
            total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
            logger.info(f"  Num Epochs = {self.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.max_train_steps}")
            global_step = 0
            first_epoch = 0

            # Potentially load in the weights and states from a previous save
            if self.resume_from_checkpoint:
                if self.resume_from_checkpoint != "latest":
                    path = os.path.basename(self.resume_from_checkpoint)
                else:
                    # Get the mos recent checkpoint
                    dirs = os.listdir(self.output_dir)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None

                if path is None:
                    accelerator.print(
                        f"Checkpoint '{self.resume_from_checkpoint}' does not exist. Starting a new training run."
                    )
                    self.resume_from_checkpoint = None
                else:
                    accelerator.print(f"Resuming from checkpoint {path}")
                    accelerator.load_state(os.path.join(self.output_dir, path))
                    global_step = int(path.split("-")[1])

                    resume_global_step = global_step * self.gradient_accumulation_steps
                    first_epoch = global_step // num_update_steps_per_epoch
                    resume_step = resume_global_step % (num_update_steps_per_epoch * self.gradient_accumulation_steps)

            if self.importance_sampling:
                print("""Using relation-focal importance sampling, which can make training more efficient
                    and is particularly beneficial in erasing mass concepts with overlapping terms.""")
  
                list_of_candidates = [
                    x for x in range(noise_scheduler.config.num_train_timesteps)
                ]
                prob_dist = [
                    importance_sampling_fn(x)
                    for x in list_of_candidates
                ]
                prob_sum = 0
                # normalize the prob_list so that sum of prob is 1
                for j in prob_dist:
                    prob_sum += j
                prob_dist = [x / prob_sum for x in prob_dist]

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
            progress_bar.set_description("Steps")

            debug_once = True
        
            if self.train_seperate:
                train_dataset.concept_number = i
            for epoch in range(first_epoch, self.num_train_epochs):
                unet.train()
                if self.train_text_encoder:
                    text_encoder.train()
        
                torch.cuda.empty_cache()
                gc.collect()
    
                for step, batch in enumerate(train_dataloader):
                    # Skip steps until we reach the resumed step        
                    if self.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                        if step % self.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                        continue

                    with accelerator.accumulate(unet):
                        # show
                        if debug_once:
                            print('==================================================================')
                            print(f'Concept {i}: {batch["instance_prompts"][0]}')
                            print('==================================================================')
                            debug_once = False
     
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
   
                        if self.importance_sampling:
                            timesteps = np.random.choice(
                                list_of_candidates,
                                size=bsz,
                                replace=True,
                                p=prob_dist)
                            timesteps = torch.tensor(timesteps).cuda()
                        else:
                            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                   
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        if self.no_real_image:
                            noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps) 
                        else:
                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                      
                        # set concept_positions for this batch
                        if self.use_gsam_mask:
                            GSAM_mask = batch['masks']
                        else:
                            GSAM_mask = None
 
                        attn_controller.set_concept_positions(batch["concept_positions"], GSAM_mask, use_gsam_mask=self.use_gsam_mask)

                        # Predict the noise residual
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                   
                        loss = attn_controller.loss()
                    
                        if self.with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)

                            # Compute prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        
                            # Add the prior loss to the instance loss.
                            loss = loss + self.prior_loss_weight * prior_loss
        
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # params_to_clip = params_to_optimize
                            params_to_clip = lora_layers.parameters()
                            accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=self.set_grads_to_none)
                        attn_controller.zero_attn_probs()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if global_step % self.checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= self.max_train_steps:
                        break

            # Create the pipeline using using the trained modules and save it.
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # save lora layers
                if self.train_seperate:
                    concepts, _ = self.multi_concept[0][i]
                else:
                    concepts = len(self.multi_concept[0])
        
                unet = accelerator.unwrap_model(unet).to(torch.float32)
                lora_path = f"{self.output_dir}/lora/{concepts}"
                os.makedirs(lora_path, exist_ok=True)
                unet.save_attn_procs(lora_path)
    
                if isinstance(self, Namespace):
                    with open(f"{self.output_dir}/my_self.json", "w") as f:
                        json.dump(vars(self), f, indent=4)

            accelerator.end_training()
    
            del lora_attn_procs, lora_layers, optimizer, lr_scheduler, attn_controller
            torch.cuda.empty_cache()

            if not self.train_seperate:
                break

        # save base initialized model
        pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=self.revision,
        )
        pipeline.save_pretrained(self.output_dir)
    
    def multi_lora_fusion(self):   
        model_id = f"{self.output_dir}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lora_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        lora_pipe.safety_checker = None
        lora_pipe.requires_safety_checker = False
  
        final_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
        final_pipe.safety_checker = None
        final_pipe.requires_safety_checker = False
        final_projection_matrices, _, _ = get_ca_layers(final_pipe.unet, with_to_k=True)
    
        train_dataset = MACEDataset(
            tokenizer=lora_pipe.tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
            use_pooler=self.use_pooler,
            multi_concept=self.multi_concept[0],
            mapping=self.mapping_concept,
            augment=self.augment,
            batch_size=self.train_batch_size,
            with_prior_preservation=self.with_prior_preservation,
            aug_length=self.aug_length,
            prompt_len=self.prompt_len,
            input_data_path=self.input_data_dir,
        )
   
        # to save memory
        CFR_dict = {}
        for layer_num in tqdm(range(len(final_projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = None
            CFR_dict[f'{layer_num}_for_mat2'] = None
    
        all_contexts = []
        all_valuess = []
        all_concepts = []
        max_concept_num = self.max_memory  # the maximum number of concept that can be processed at once
        count = 0
        for single_concept in train_dataset.dict_for_close_form:
            count += 1
            print(f"============================== Concept {count}: {single_concept['old'][0][1]} ==============================")
            all_concepts.append(single_concept['old'][0][1])
            lora_pipe.load_lora_weights(f"{model_id}/lora/{single_concept['old'][0][1].replace(' ', '-')}")
            lora_pipe.fuse_lora(lora_scale=1.0)
   
            lora_projection_matrices, lora_ca_layers, lora_og_matrices = get_ca_layers(lora_pipe.unet, with_to_k=True)

            contexts, valuess = prepare_k_v(lora_pipe.text_encoder, lora_projection_matrices, lora_ca_layers, lora_og_matrices,
                                            [single_concept], lora_pipe.tokenizer, all_words=True, prepare_k_v_for_lora=True)

            # if the number of concept is too large, we need to use cache mode to save memory
            if len(train_dataset.dict_for_close_form) > max_concept_num:
                closed_form_refinement(lora_projection_matrices, contexts, valuess, cache_dict=CFR_dict, cache_mode=True)

                del contexts, valuess
                gc.collect()
                torch.cuda.empty_cache()
            else:
                all_contexts.append(contexts[0])
                all_valuess.append(valuess[0])
    
            lora_pipe.unfuse_lora()
            lora_pipe.unload_lora_weights()

        del lora_pipe
        gc.collect()
        torch.cuda.empty_cache()
  
        # Load cached prior knowledge for preserving
        if self.prior_preservation_cache_path:
            prior_preservation_cache_dict = torch.load(self.prior_preservation_cache_path, map_location=device)
        else:
            prior_preservation_cache_dict = {}
            for layer_num in tqdm(range(len(final_projection_matrices))):
                prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
                prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
       
        # Load cached domain knowledge for preserving
        if self.domain_preservation_cache_path:
            domain_preservation_cache_dict = torch.load(self.domain_preservation_cache_path, map_location=device)
        else:
            domain_preservation_cache_dict = {}
            for layer_num in tqdm(range(len(final_projection_matrices))):
                domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
                domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0

        # integrate the preserving knowledge and multi-lora knowledge
        cache_dict = {}
        if len(train_dataset.dict_for_close_form) > max_concept_num:   
            for key in CFR_dict:
                cache_dict[key] = self.train_preserve_scale * (prior_preservation_cache_dict[key] \
                                + self.preserve_weight * domain_preservation_cache_dict[key]) \
                                + CFR_dict[key]

            closed_form_refinement(final_projection_matrices, lamb=self.lamb, preserve_scale=1, cache_dict=cache_dict)
        else:
            for key in prior_preservation_cache_dict:
                cache_dict[key] = prior_preservation_cache_dict[key] \
                                + self.preserve_weight * domain_preservation_cache_dict[key]
                                
            closed_form_refinement(final_projection_matrices, all_contexts, all_valuess, lamb=self.lamb,
                preserve_scale=self.fuse_preserve_scale, cache_dict=cache_dict)

        # save the final model
        final_pipe.save_pretrained(self.final_save_path)

    def train(self):
        logger.info("train function entered")

        # training data preparation
        if self.generate_training_data:
            logger.info("Generating training images for MACE")
            model_id = self.pretrained_model_name_or_path
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
            pipe.safety_checker=None
            pipe.requires_safety_checker=False
            torch.Generator(device=self.device).manual_seed(42)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler)
            num_images = 8
            count = 0
            for single_concept in self.multi_concept:
                for c, t in single_concept:
                    count = count + 1
                    print(f"Generating training data for concept {count} = {c}...")
                    c = c.replace('-', ' ')
                    output_folder = f"{self.output_dir}/{c}"
                    os.makedirs(output_folder, exist_ok=True)
                    if t == "object":
                        prompt = f"a photo of the {c}"
                        print(f'Inferencing = {prompt}')
                        images = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, num_images_per_prompt=num_images).images
                        for i, im in enumerate(images):
                            im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                    elif t == "style" :
                        prompt = f"a photo in the style of {c}"
                        print(f'Inferencing = {prompt}')
                        images = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, num_images_per_prompt=num_images).images
                        for i, im in enumerate(images):
                            im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                    else:
                        raise ValueError("unknown concept type.")
                    del images
                    torch.cuda.empty_cache()
                    gc.collect()

                    del pipe
                    torch.cuda.empty_cache()
                    gc.collect()

            if self.use_gsam_mask:
                grounded_model = load_model(self.grounded_config, self.grounded_checkpoint, device=self.device)

                if self.use_sam_hq:
                    predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.sam_hq_checkpoint).to(self.device))
                else:
                    predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.sam_checkpoint).to(self.device))

                transform = transforms.ToTensor()
                for root, _, files in os.walk(self.input_data_dir) :
                    mask_save_path = root.replace(f'{os.path.basename(root)}',f'{os.path.basename(root)} mask')
                    os.makedirs(mask_save_path, exist_ok=True)
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(file_path)
                        # read images and get masks
                        image = Image.open(file_path)
                        if not image.mode == "RGB":
                            image = image.convert("RGB")
                        tensor_image = transform(image).to(self.device)
                        GSAM_mask = get_mask(tensor_image, os.path.basename(root), grounded_model, predictor, self.device)
                        # save masks
                        GSAM_mask = (GSAM_mask.to(torch.uint8) * 255).squeeze()
                        save_mask = to_pil_image(GSAM_mask)
                        save_mask.save(f"{os.path.join(mask_save_path, file).replace('.jpg','_mask.jpg')}")

        # stage 1 & 2 (CFR and LORA training)
        self.cfr_lora_training()

        # stage 3 (Multi-LoRA fusion)
        self.multi_lora_fusion()

        self.save_entire_model = True

        '''
        if self.test_erased_model :
            self.inference(self.final_save_path,False,self.multi_concept,device,50,self.final_save_path)
            # inference({
            #     "pretrained_model_name_or_path" = self.final_save_path,
            #     "multi_concept" = self.multi_concept,
            #     "generate_training_data" = False,
            #     "device" = device,
            #     "steps" = 50,
            #     "output_dir" = selffinal_save_path,
            # })
        '''

    def evaluate(self) -> Tuple[List[EvalResult], Dict[str, Image.Image]]:
        logger.info("Evaluate function entered")
        assert type(self.forget_prompt) == list  # noqa
        assert type(self.preserve_prompt) == list  # noqa
        # TO determine whether to use AutoPipelineForText2Image or DiffusionPipeline or something else
        if self.save_entire_model:
            pipeline_original = AutoPipelineForText2Image.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker=None).to(self.device)
            pipeline_unlearned = AutoPipelineForText2Image.from_pretrained(self.final_save_path, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        else:
            # Maybe have a static method that loads the base model and applies the uce weights?
            # Should be easy for the user to do the same
            raise NotImplementedError("MACE evaluation currently only supports saving the entire model. Set save_entire_model to True.")

        evaluator = EvaluatorTextToImage(
            pipeline_original=pipeline_original,
            pipeline_unlearned=pipeline_unlearned,
            pipeline_learned=None,
            prompts_forget=self.forget_prompt,
            prompts_retain=self.preserve_prompt,
            metric_clip=MetricImageTextSimilarity(metrics=['clip']),
            compute_runtimes=self.compute_runtimes
        )

        eval_result, eval_images = evaluator.evaluate()

        return eval_result, eval_images


def main():
    logger.info("Main function entered.")
    mace = MACEUnlearner(multi_concept = [[("melania-trump","object")]],
        user_pooler=True,
        train_batch_size=1,
        learning_rate=1.0e-04,
        max_train_steps=50,
        train_preserve_scale=1.0e-4,
        fuse_preserve_scale=10.e-4,
        mapping_concept=["a woman"],
        augment=True,
        lamb=0.0,
        rank=1,
        lora=True,
        train_seperate=True,
        importance_sampling=True,
        max_memory=1000,
        aug_length=30,
        prompt_len=30,
        all_words=False,
        generate_data=True,
        use_gpt=False,
        test_erased_model=False,
        prior_preservation_cache_path="./cache/cache_coco.pt",
        domain_preservation_cache_path="./cache/cache_cele.pt",
        preserve_weight=8.0e+3,
        input_data_dir="./data/1cele",
        output_dir="./saved_model/CFR_with_multi_LoRAs",
        final_save_path="./saved_model/LoRA_fusion_model",
        use_gsam_mask=True,
        use_sam_hq=True,
        grounded_config="./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint="./Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
        sam_hq_checkpoint="./Grounded-Segment-Anything/sam_hq_vit_h.pth",
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        with_prior_preservation=False,
        preserve_prompt=["a person"],
        forget_prompt=["a animal"],
        dataset_retain_name="data/a person",
        dataset_forget_name="data/a animal",
        prior_loss_weight=1.0,
        with_uncond_loss=False,
        negative_guidance=1.0,
        uncond_loss_weight=1.0,
        num_class_images=200,
        seed=2024,
        resolution=512,
        revision=None,
        tokenizer_name=None,
        instance_prompt=None,
        concept_keyword=None,
        no_real_image=False,
        center_crop=False,
        train_text_encoder=False,
        sample_batch_size=4,
        num_train_epochs=1,
        checkpointing_steps=500,
        resume_from_checkpoint=None,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        scale_lr=False,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        lr_num_cycles=1,
        lr_power=1.0,
        use_8bit_adam=False,
        dataloader_num_workers=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.01,
        adam_epsilon=1.0e-08,
        max_grad_norm=1,
        push_to_hub=False,
        hub_token=None,
        hub_model_id=None,
        logging_dir="logs",
        allow_tf32=False,
        report_to="tensorboard",
        mixed_precision=None,
        prior_generation_precision=None,
        local_rank=-1,
        enable_xformers_memory_efficient_attention=False,
        set_grads_to_none=False,
        save_entire_model=False,
        generate_training_data=True,
        compute_runtimes=True)

    mace.data_preparation()
    mace.train()
    mace.evaluate()


    

