import os
import json
import sys
import random
import torch
import gc
import argparse

sys.path.append('..')
sys.path.append('../../vision-unlearning')

from unlearner_lora_distillation_UC import UnlearnerLoraDistillation_UC

from vision_unlearning.utils.logger import get_logger, setup_loggers
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

from constants import theme_available


def get_all_prompts_from_valid_json(json_file_path, split_name="train"):
    """
    Reads a single, valid JSON file, expects an array of examples under the
    specified split key (e.g., "train"), and extracts all 'text' fields.

    Args:
        json_file_path (str): The full path to the .json file.
        split_name (str): The key in the JSON file that holds the list of examples (default is "train").

    Returns:
        list: A list of strings containing all the extracted prompts.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return []

    prompts_list = []

    # 1. Read the entire file content
    with open(json_file_path, 'r') as f:
        full_data = json.load(f)  # json.load() reads and parses the entire file

    # 2. Check if the specified split exists in the loaded data
    if split_name not in full_data or not isinstance(full_data[split_name], list):
        print(f"Error: Could not find '{split_name}' list in the JSON data.")
        return []

    # 3. Iterate through the list of dictionaries (the dataset examples)
    for entry in full_data[split_name]:
        if 'text' in entry:
            prompts_list.append(entry['text'])

    return prompts_list


def train_model_style(model_name_or_path,
                      dataset_root,
                      theme_name,
                      output_dir,
                      seed=42,
                      num_train_epochs=1,
                      num_to_select=5,
                      target="Abstractionism",
                      target_overwrite="Photo",
                      # parameters that can be changed at runtime
                      # values used in previous runs
                      mixed_precision:str="fp16",
                      learning_rate:float=1e-4,
                      max_grad_norm:float=1.0,
                      validation_epochs:int=1,
                      lr_scheduler_type:str="cosine",
                      lora_r:int=4,
                      lora_alpha:int=4,
                      lora_dropout:float=0.1
):
    global theme_available

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        forget_theme_index = theme_available.index(target)
    except ValueError:
        print(f"Error: The target theme '{target}' is not in the available themes list.")
        return None
    
    try:
        overwrite_theme_index = theme_available.index(target_overwrite)
    except ValueError:
        print(f"Error: The target theme '{target_overwrite}' is not in the available themes list.")
        return None
    
    if target_overwrite == "Seed_Images":
        target_overwrite = "Photo"

    model_lora_path = os.path.join(output_dir, f"distill_refactoring_{num_train_epochs:03d}")

    dataset_forget_meta = os.path.join(dataset_root, f"metadata-{target}-forget.json")
    dataset_retain_meta = os.path.join(dataset_root, f"metadata-{target}-retain.json")

    prompts_forget = get_all_prompts_from_valid_json(dataset_forget_meta, split_name="train")
    prompts_retain = get_all_prompts_from_valid_json(dataset_retain_meta, split_name="train")

    validation_prompt = f"An image of Dogs in {target_overwrite} style."

    if len(prompts_forget) < num_to_select:
        print(f"Error: The list only contains {len(prompts_forget)} prompts, which is less than the requested {num_to_select}.")
        example_prompts_forget = prompts_forget # Return all prompts if the list is too small
    else:
        example_prompts_forget = random.sample(prompts_forget, num_to_select)

    if len(prompts_retain) < num_to_select:
        print(f"Error: The list only contains {len(prompts_retain)} prompts, which is less than the requested {num_to_select}.")
        example_prompts_retain = prompts_retain # Return all prompts if the list is too small
    else:
        example_prompts_retain = random.sample(prompts_retain, num_to_select)

    logger = get_logger('main')
    setup_loggers(modules_info=['vision_unlearning.'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Overwritting the class {target} by the class {target_overwrite}")
    
    free_memory, total_memory = torch.cuda.mem_get_info()  # in GB
    logger.info(f"Free Memory: {free_memory / 1e9:.2f} GB")
    logger.info(f"Total Memory: {total_memory / 1e9:.2f} GB")

    # Most hyperparameters should be set here, excep the ones that change at runtime
    hyperparameters = {
        "dataloader_num_workers": 2,
        "resolution": 512,
        "num_validation_images": 1,

        "mixed_precision": mixed_precision, # specified by model trained on styles for unlearn canvas
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "lr_warmup_steps": 0,
        "num_train_epochs": num_train_epochs,
        "validation_epochs": validation_epochs,
        "checkpointing_steps": 10000,
        "lr_scheduler_type": lr_scheduler_type,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "random_flip": True,

        "lora_r": lora_r,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,

        "seed": 42,
    }

    # gradient_accumulation_steps can be bigger
    if free_memory > 20e9: # a100
        logger.info('Choosing hyperparams for free_memory>20e9')
        hyperparameters.update({
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
        })
    elif free_memory > 14e9:  # v100
        logger.info('Choosing hyperparams for free_memory>14e9')
        hyperparameters.update({
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
        })
    else:
        logger.error('You should not even be trying...')
        
    logger.info(hyperparameters)

    unlearner = UnlearnerLoraDistillation_UC(
        model_name_or_path=model_name_or_path,
        dataset_forget_name=dataset_root,
        dataset_retain_name=dataset_root,
        output_dir=model_lora_path,
        use_metadata=True,
        json_metafile_forget=dataset_forget_meta,
        json_metafile_retain=dataset_retain_meta,
        image_column="file_name",
        caption_column="text",
        overwrite_column="overwrite",
        validation_prompt=validation_prompt,
        final_eval_prompts_forget = example_prompts_forget,
        final_eval_prompts_retain = example_prompts_retain,
        gradient_weighting_method = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        device=device,
        **hyperparameters,
    )

    eval_results = unlearner.train()
    logger.info(f"Eval results: {eval_results}")
    
    del unlearner

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='train_models_style',
        description='Generate unlearned models with FADE for UnlearnCanvaEval')
    parser.add_argument('--pretrained_model_base_path', type=str, default="assets/data/sd", required=False, help="Name or path to trained model that should be modified.")
    parser.add_argument('--dataset_base_path', type=str, default="assets/data/forget_styles", required=False, help="Root path to the dataset, where the json files with retain and forget info can be found.")
    parser.add_argument('--output_dir', type=str, default="assets/models", required=False, help="Folder to save unlearned models.")
    parser.add_argument('--overwriting_concept', type=str, default=None, required=False, help="Concept to overwrite the forget concept.")

    parser.add_argument('--mix_prec', type=str, default="fp16", required=False, help="Mixed precision type.")
    parser.add_argument('--lr', type=float, default=1e-4, required=False, help='Learning rate.',)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, required=False, help="Max gradient normal.")
    parser.add_argument("--val_epochs", type=int, default=1, required=False, help="Number of validation epochs.")
    parser.add_argument('--scheduler_type', type=str, default="cosine", required=False, help="Learning scheduler type.")
    parser.add_argument('--lora_r', type=int, default=4, required=False, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int, default=4, required=False, help="LoRA scaling factor.")
    parser.add_argument('--lora_dropout', type=float, default=0.1, required=False, help="Dropout rate for LoRA training.")
    parser.add_argument('--themes', type=str, nargs="+", default=theme_available, required=False, help="Forget themes list.")

    args = parser.parse_args()
    
    for target in args.themes:
        if target == "Seed_Images":
            continue

        if target not in theme_available:
            print(f"Error: The target theme '{target}' is not in the available themes list.")
            continue
        
        model_base_path = os.path.join(args.output_dir, f"forget_style_{target}")
        os.makedirs(model_base_path, exist_ok=True)

        # define overwriting concept
        if args.overwriting_concept is not None:
            overwrite_theme = args.overwriting_concept
            if overwrite_theme not in theme_available:
                raise ValueError(f"The overwriting concept '{overwrite_theme}' is not in the available themes list.")
        else:
            # select next theme as overwrite concept
            forget_theme_index = theme_available.index(target)
            selected_index = forget_theme_index + 1

            if selected_index >= len(theme_available):
                selected_index = 0

            overwrite_theme = theme_available[selected_index]
            
            # skip Seed_Images if it was selected as overwriting concept
            if overwrite_theme == "Seed_Images":
                selected_index = selected_index + 1
                if selected_index >= len(theme_available):
                    selected_index = 0
                overwrite_theme = theme_available[selected_index]

        result = train_model_style(
            model_name_or_path=args.pretrained_model_base_path,
            dataset_root=args.dataset_base_path,
            theme_name=target,
            output_dir=model_base_path,
            target=target,
            target_overwrite=overwrite_theme, # used in validation prompt
            mixed_precision=args.mix_prec,
            learning_rate=args.lr,
            max_grad_norm=args.max_grad_norm,
            validation_epochs=args.val_epochs,
            lr_scheduler_type=args.scheduler_type,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

        if result is not None:
            print(f"Finished training for forgetting style: {target}")
        else:
            print(f"Training for forgetting style: {target} was skipped or failed.")

