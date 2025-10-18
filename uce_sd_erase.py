"""
Taken from the Unified Concept Editing (UCE) repository:
GitHub: https://github.com/rohitgandikota/unified-concept-editing
Arxiv: https://arxiv.org/pdf/2308.14761.pdf
"""

from __future__ import annotations
import os
import time
import copy
import logging
from pathlib import Path
from enum import Enum
from typing import Optional, Any, cast

import torch  # noqa: F401
import torch.nn as nn
from pydantic import Field
from safetensors.torch import save_file, load_file
from diffusers import DiffusionPipeline
from IPython.display import display

import base


class ConceptType(str, Enum):
    """Enum representing the type of concept to unlearn."""
    Object = "object"
    Art = "art"


class UCE(base.Unlearner):
    """Unified Concept Eraser class."""

    pretrained_model_name_or_path: str = Field(
        default="CompVis/stable-diffusion-v1-4",
        description="Path to pretrained model or model identifier from huggingface.co/models."
    )
    device: str = "cuda:0"
    erase_scale: float = 0.5
    preserve_scale: float = 1.0
    lamb: float = 0.5
    output_dir: str = Field(
        default="../uce_models",
        description="Output directory for model predictions and checkpoints."
    )
    edit_concepts: Optional[str] = None
    guide_concepts: Optional[str] = None
    preserve_concepts: Optional[str] = None
    concept_type: ConceptType = Field(
        default=ConceptType.Object,
        description="Type of concept to unlearn."
    )
    expand_prompts: bool = True

    def __init__(self, **data: Any):
        """Custom initializer for UCE with informative logging."""
        super().__init__(**data)

        print("\n[INFO] Initializing Unified Concept Eraser (UCE)...")
        print(f" - Base model:        {self.pretrained_model_name_or_path}")
        print(f" - Device:            {self.device}")
        print(f" - Erase scale:       {self.erase_scale}")
        print(f" - Preserve scale:    {self.preserve_scale}")
        print(f" - Regularization λ:  {self.lamb}")
        print(f" - Edit concepts:     {self.edit_concepts}")
        print(f" - Guide concepts:    {self.guide_concepts}")
        print(f" - Preserve concepts: {self.preserve_concepts}")
        print(f" - Concept type:      {self.concept_type}")
        print(f" - Output directory:  {self.output_dir}")
        print(f" - Expand prompts:    {self.expand_prompts}\n")

    def train(self) -> None:
        """Main UCE training and concept erasure logic."""

        # ==== Sanity checks ====
        assert self.pretrained_model_name_or_path, "Pretrained model path must not be empty."
        assert isinstance(self.erase_scale, (int, float)) and self.erase_scale > 0, "Erase scale must be positive."
        assert isinstance(self.preserve_scale, (int, float)) and self.preserve_scale >= 0, "Preserve scale must be non-negative."
        assert 0.0 <= self.lamb <= 1.0, "Lambda must be between 0 and 1."
        assert self.device in ["cpu", "cuda", "cuda:0", "cuda:1"], f"Invalid device specified: {self.device}"
        assert isinstance(self.concept_type, ConceptType), "concept_type must be of type ConceptType Enum."

        if "cuda" in self.device:
            assert torch.cuda.is_available(), "CUDA device specified but not available!"

        torch_dtype: torch.dtype = torch.float32
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if self.pretrained_model_name_or_path != "CompVis/stable-diffusion-v1-4":
            logging.warning("UCE was not tested with this base model; results may differ.")

        # ==== Concept parsing ====
        assert self.edit_concepts, "At least one edit concept must be provided."
        edit_list: list[str] = [c.strip() for c in self.edit_concepts.split(';') if c.strip()]

        assert len(edit_list) > 0, "Edit concepts list cannot be empty after parsing."

        guide_list: list[str] = []
        if self.guide_concepts:
            guide_list = [c.strip() for c in self.guide_concepts.split(';') if c.strip()]
        elif self.concept_type == ConceptType.Art:
            guide_list = ["art"] * len(edit_list)
        else:
            #default guide for objects: use a neutral object class or same as edit concept
            guide_list = [e for e in edit_list] 

        if len(guide_list) == 1:
            guide_list *= len(edit_list)

        if len(guide_list) != len(edit_list):
            raise ValueError(
                "Mismatch between edit and guide concepts. Ensure they are separated by ';' and have equal counts."
            )

        preserve_list: list[str] = []
        if self.preserve_concepts:
            preserve_list = [c.strip() for c in self.preserve_concepts.split(';') if c.strip()]

        # ==== Prompt expansion ====
        if self.expand_prompts:
            edit_copy = copy.deepcopy(edit_list)
            guide_copy = copy.deepcopy(guide_list)

            for concept, guide_concept in zip(edit_copy, guide_copy):
                if self.concept_type == ConceptType.Art:
                    edit_list.extend([
                        f"painting by {concept}", f"art by {concept}",
                        f"artwork by {concept}", f"picture by {concept}",
                        f"style of {concept}"
                    ])
                    guide_list.extend([
                        f"painting by {guide_concept}", f"art by {guide_concept}",
                        f"artwork by {guide_concept}", f"picture by {guide_concept}",
                        f"style of {guide_concept}"
                    ])
                else:
                    edit_list.extend([
                        f"image of {concept}", f"photo of {concept}",
                        f"portrait of {concept}", f"picture of {concept}",
                        f"painting of {concept}", f"picture of {concept} doing something"
                    ])
                    guide_list.extend([
                        f"image of {guide_concept}", f"photo of {guide_concept}",
                        f"portrait of {guide_concept}", f"picture of {guide_concept}",
                        f"painting of {guide_concept}", f"picture of {concept} doing something"
                    ])

        print(f"\nErasing: {edit_list}\nGuiding: {guide_list}\nPreserving: {preserve_list} with erase_scale: {self.erase_scale}, preserve_scale: {self.preserve_scale} and regularization lambda: {self.lamb}\n")

        # ==== Diffusion pipeline ====
        pipe = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            vae=None
        ).to(self.device)

        uce_run(
            pipe, edit_list, guide_list, preserve_list,
            self.erase_scale, self.preserve_scale, self.lamb,
            self.output_dir, self.device, torch_dtype
        )
    
    def generate_images(self, prompt): 
        device = self.device
        pipe = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None      
        ).to(device)

        print("Base model is loaded.\n")

        uce_weight_path = "../uce_models/uce_sd_weights.safetensors"
        uce_state_dict = load_file(uce_weight_path)

        print(f"Loaded {len(uce_state_dict)} UCE weight tensors")

        # Applying the modified weights
        with torch.no_grad():
            for name, param in pipe.unet.named_parameters():
                if name in uce_state_dict:
                    print(f"Updating: {name}")
                    param.copy_(uce_state_dict[name])
        
        output = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
        image = output.images[0]
        display(image)
        os.makedirs("../generated_images",exist_ok=True)
        image.save("../generated_images/erased_output.png")
        print("Image saved as 'erased_output.png")
        







    


# ===================== Helper Functions ===================== #

def collect_text_embeddings(pipe: Any, concepts: list[str],
                            device: str, torch_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Return dict {concept: last_token_embedding}."""
    uce_embeds: dict[str, torch.Tensor] = {}

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
                return_tensors="pt"
            )["attention_mask"]
        ).sum() - 2

        uce_embeds[e] = t_emb[0][:, last_token_idx, :]
    return uce_embeds


def collect_guide_outputs(concepts: list[str], embeds: dict[str, torch.Tensor],
                          modules: list[torch.nn.Module]) -> dict[str, list[torch.Tensor]]:
    """Collect cross-attention outputs for guide/preserve concepts."""
    outputs: dict[str, list[torch.Tensor]] = {}

    for g in concepts:
        if g in outputs:
            continue
        t_emb = embeds[g]
        for module in modules:
            outputs[g] = outputs.get(g, []) + [module(t_emb)]
    return outputs


def update_weights(original_modules: list[torch.nn.Module],
                   erase_embeds: dict[str, torch.Tensor],
                   guide_outputs: dict[str, list[torch.Tensor]],
                   edit_concepts: list[str], guide_concepts: list[str],
                   preserve_concepts: list[str],
                   erase_scale: float, preserve_scale: float,
                   lamb: float, device: str, torch_dtype: torch.dtype
                   ) -> list[torch.nn.Module]:
    """Apply the UCE weight update to each module and return new modules."""
    uce_modules = copy.deepcopy(original_modules)

    for module_idx, module in enumerate(original_modules):
        if isinstance(module, nn.Module):
            w_old: torch.Tensor = cast(torch.Tensor, module.weight)
        else:
            w_old = cast(torch.Tensor, module)  # fallback if somehow not a module

        # Compute mat1 safely
        mat1: torch.Tensor = lamb * w_old

        # Compute mat2 safely
        mat2: torch.Tensor = lamb * torch.eye(
            w_old.shape[1], device=w_old.device, dtype=w_old.dtype
        )

        # Erase concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = erase_embeds[erase_concept].T
            v_i_star = guide_outputs[guide_concept][module_idx].T
            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)

        # Preserve concepts
        for preserve_concept in preserve_concepts:
            c_i = erase_embeds[preserve_concept].T
            v_i_star = guide_outputs[preserve_concept][module_idx].T
            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)

        # uce_modules[module_idx].weight = torch.nn.Parameter(
        #     mat1 @ torch.inverse(mat2.float()).to(torch_dtype)
        # )

        eps = 1e-6
        mat2_float = mat2.float() + eps * torch.eye(mat2.shape[0], device=mat2.device)
        uce_modules[module_idx].weight = torch.nn.Parameter(
            (mat1 @ torch.inverse(mat2_float)).to(torch_dtype)
        )

    return uce_modules


def save_uce_weights(uce_modules: list[torch.nn.Module],
                     uce_module_names: list[str],
                     save_dir: str) -> None:
    """Save updated module weights to a safetensors file."""
    uce_state_dict: dict[str, torch.Tensor] = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        weight_tensor: torch.Tensor = cast(torch.Tensor, parameter.weight)
        uce_state_dict[name + ".weight"] = weight_tensor

    # You can customize filename here
    save_file(uce_state_dict, os.path.join(save_dir, "uce_sd_weights.safetensors"))


def uce_run(pipe: Any, edit_concepts: list[str], guide_concepts: list[str],
            preserve_concepts: list[str], erase_scale: float,
            preserve_scale: float, lamb: float, save_dir: str,
            device: str = "cuda:0", torch_dtype: torch.dtype = torch.float32) -> None:
    """Main execution routine for Unified Concept Erasure."""
    torch.set_grad_enabled(False)
    start_time = time.time()

    # Find relevant modules
    uce_modules: list[torch.nn.Module] = []
    uce_module_names: list[str] = []

    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_v") or name.endswith("to_k")):
            uce_modules.append(module)
            uce_module_names.append(name)

    assert len(uce_modules) > 0, "No attention modules found for UCE to operate on."
    original_modules = copy.deepcopy(uce_modules)

    # 1. Collect embeddings
    all_concepts = edit_concepts + guide_concepts + preserve_concepts
    erase_embeds = collect_text_embeddings(pipe, all_concepts, device, torch_dtype)
    assert all(c in erase_embeds for c in all_concepts), "Some concepts failed to produce embeddings."

    # 2. Collect guide outputs
    guide_outputs = collect_guide_outputs(guide_concepts + preserve_concepts, erase_embeds, original_modules)

    # 3. Apply weight updates
    updated_modules = update_weights(
        original_modules, erase_embeds, guide_outputs,
        edit_concepts, guide_concepts, preserve_concepts,
        erase_scale, preserve_scale, lamb, device, torch_dtype
    )

    # 4. Save weights
    save_uce_weights(updated_modules, uce_module_names, save_dir)

    duration = time.time() - start_time
    print(f"\nErased concepts using UCE.\nModel edited in {duration:.2f} seconds.\n")


def main() -> None:
    """Run UCE training with example inputs."""
    uce = UCE(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        erase_scale=0.9,
        preserve_scale=1.0,
        lamb=0.5,
        edit_concepts="cat; dog", # Van Gogh; Picasso
        guide_concepts="animals",
        preserve_concepts="lion; tiger; leopard", # Monet; Rembrandt; Warhol
        device="cuda:0",
        concept_type=ConceptType.Object # Can be ConceptType.Art
    )
    uce.train()
    prompt = input("Enter the prompt:-\n")
    uce.generate_images(prompt)


if __name__ == "__main__":
    main()

