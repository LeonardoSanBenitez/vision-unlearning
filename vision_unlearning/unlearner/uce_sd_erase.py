import torch

import argparse
import os
import copy
import time

from safetensors.torch import save_file
from diffusers import DiffusionPipeline


def collect_text_embeddings(pipe, concepts, device, torch_dtype):
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

        uce_embeds[e] = t_emb[0][:, last_token_idx, :]
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
    uce_modules = copy.deepcopy(original_modules)

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

def save_uce_weights(uce_modules, uce_module_names, save_dir, exp_name):
    """Save updated module weights to a safetensors file."""
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name + '.weight'] = parameter.weight
    save_file(uce_state_dict, os.path.join(save_dir, exp_name + '.safetensors'))

def uce_run(pipe, edit_concepts, guide_concepts, preserve_concepts,
        erase_scale, preserve_scale, lamb, save_dir, exp_name,
        device="cuda:0", torch_dtype=torch.float32):
    torch.set_grad_enabled(False)
    start_time = time.time()

    # Find relevant modules
    uce_modules, uce_module_names = [], []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)

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
    save_uce_weights(updated_modules, uce_module_names, save_dir, exp_name)

    end_time = time.time()
    print(f"\n\nErased concepts using UCE\nModel edited in {end_time-start_time:.2f} seconds\n")

def main():
    from .base import UCEUnlearner
    uce = UCEUnlearner(model_id='CompVis/stable-diffusion-v1-4',edit_concepts='Van Gogh; Picasso',guide_concepts='art',preserve_concepts='Monet; Rembrandt; Warhol',device='cuda:0',concept_type='art',exp_name='vangogh_uce_sd') 

    output_path = uce.train()
    print("UCE weights trained at: ",output_path)


if __name__ == "__main__":
    main()

