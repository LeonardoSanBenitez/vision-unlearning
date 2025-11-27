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


# import torch  # noqa: F401
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
# from vision_unlearning.config.mace_config import MACEConfig 
# from vision_unlearning.utils.model_management import save_model_card
# from vision_unlearning.utils.logger import get_logger

import os
import torch
import gc
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import cv2

from vision_unlearning.unlearner.base import Unlearner
from vision_unlearning.configs.mace_config import MACEConfig
from vision_unlearning.utils.logger import get_logger
from vision_unlearning.unlearner.mace_utils.mace_supplementary import grounded_segmentation
from vision_unlearning.unlearner.mace_utils.cfr_lora_training import main as cfr_lora_training
from vision_unlearning.unlearner.mace_utils.fused_lora_closed_form import main as multi_lora_fusion

from vision_unlearning.unlearner.mace_utils.grounded_sam_util import *
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = get_logger("MACE")

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
    config: MACEConfig

    def data_preparation(self):
        device = "cuda" if torch.is_cuda_available() else "cpu"
        self.config.device = device

        # generate 8 images per concept using the original model for performing erasure
        if self.config.generate_data:
            logger.info("Generating training images for MACE...")

            self.inference("CompVis/stable-diffusion-v1-4",True,self.config.multi_concept,self.config.device,30,self.config.input_data_dir)
            # inference({
            #     "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
            #     "multi_concept": self.config.multi_concept,
            #     "generate_training_data": True,
            #     "device": self.config.device,
            #     "steps": 30,
            #     "output_dir": self.config.input_data_dir,
            # })
        
        # get and save masks for each image
        if self.config.use_gsam_mask:
            grounded_model = load_model(self.config.grounded_config, self.config.grounded_checkpoint, devide = self.config.device)

            if self.config.use_sam_hq:
                predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.config.sam_hq_checkpoint).to(self.config.device))
            else:
                predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=self.config.sam_checkpoint).to(self.config.device))

            transform = transforms.ToTensor()
            for root,_,_, files in os.walk(self.config.input_data_dir):
                mask_save_path = root.replace(f'{os.path.basename(root)}',f'{os.path.basename(root)} mask')
                os.makedirs(mask_save_path,exist_ok=True)
                for file in files:
                    file_path = os.path.join(root,file)
                    print(file_path)
                    # read images and get masks
                    image = Image.open(file_path)
                    if not image.mode == "RGB":
                        image = image.convert("RGB")
                    tensor_image = transform(image).to(self.config.device)
                    GSAM_mask = get_mask(tensor_image,os.path.basename(root), grounded_model, predictor, self.config.device)
                    #save masks
                    GSAM_mask = (GSAM_mask.to(torch.uint8) * 255).squeeze()
                    save_mask = to_pil_image(GSAM_mask)
                    save_mask.save(f"{os.path.join(mask_save_path, file).replace('.jpg','_mask.jpg')}")

    def data_preparation_transformers(self):
        device = "cuda" if torch.is_cuda_available() else "cpu"
        self.config.device = device

        # generate 8 images per concept using the original model for performing erasure
        if self.config.generate_data:
            logger.info("Generating training images for MACE...")

            self.inference("CompVis/stable-diffusion-v1-4",True,self.config.multi_concept,self.config.device,30,self.config.input_data_dir)
            # inference({
            #     "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
            #     "multi_concept": self.config.multi_concept,
            #     "generate_training_data": True,
            #     "device": self.config.device,
            #     "steps": 30,
            #     "output_dir": self.config.input_data_dir,
            # })
        
        #get and save mask for each image
        if self.config.use_gsam_mask:
            detector_id = "IDEA-Research/grounding-dino-base"
            segmenter_id = "facebook/sam-vit-huge"

            for root, _, files in os.walk(self.config.input_data_dir):
                mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
                os.makedirs(mask_save_path, exist_ok=True)
                for file in files:
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

    def trasformer_gsam_util(self):
        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"

        transform = transforms.ToTensor()
        for root,_,_,files in os.walk(self.config.input_data_dir):
            mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
            os.makedirs(mask_save_path, exist_ok=True)
            for file in files:
                print(file, root)
                GSAM_mask = grounded_segmentation(
                    image=os.path.join(root,file),
                    labels='a person',
                    threshold=0.3,
                    polygon_refinement=True,
                    detector_id=detector_id,
                    segmenter_id=segmenter_id
                )

            cv2.imwrite(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}", GSAM_mask)

    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # stage 1 & 2 (CFR and LORA training)
        cfr_lora_training(self.config)
        
        # stage 3 (Multi-LoRA fusion)
        multi_lora_fusion(self.config)

        if self.config.test_erased_model:
            self.inference(self.config.final_save_path,False,self.config.multi_concept,device,50,self.config.final_save_path)
            # inference({
            #     "pretrained_model_name_or_path": self.config.final_save_path,
            #     "multi_concept": self.config.multi_concept,
            #     "generate_training_data": False,
            #     "device": device,
            #     "steps": 50,
            #     "output_dir": self.configfinal_save_path,
            # })

    def inference(self,pretrained_model_name_or_path, generate_training_data,multi_concept, device, steps, output_dir):
        model_id = pretrained_model_name_or_path
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        torch.Generator(device=device).manual_seed(42)

        if generate_training_data:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            num_images = 8
            count = 0
            for single_concept in multi_concept:
                for c, t in single_concept:
                    count += 1
                    print(f"Generating training data for concept {count}: {c}...")
                    c = c.replace('-', ' ')
                    output_folder = f"{output_dir}/{c}"
                    os.makedirs(output_folder, exist_ok=True)
                    if t == "object":
                        prompt = f"a photo of the {c}"
                        print(f'Inferencing: {prompt}')
                        images = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                        for i, im in enumerate(images):
                            im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                    elif t == "style":
                        prompt = f"a photo in the style of {c}"
                        print(f'Inferencing: {prompt}')
                        images = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                        for i, im in enumerate(images):
                            im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                    else:
                        raise ValueError("unknown concept type.")
                    del images
                    torch.cuda.empty_cache()
                    gc.collect()
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            num_images = 8
            output_folder = f"{output_dir}/generated_images"
            os.makedirs(output_folder, exist_ok=True)
            print(f"Inference using {pretrained_model_name_or_path}...")
            prompt = prompt
            images = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
            for i, im in enumerate(images):
                im.save(f"{output_folder}/o_{prompt.replace(' ', '-')}_{i}.jpg")  
            
            torch.cuda.empty_cache()
            gc.collect()

        del pipe
        torch.cuda.empty_cache()
        gc.collect()





    

