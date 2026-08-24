
import argparse
import os
import sys
import random
import dotenv
import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForText2Image
from typing import Any, Dict, List, Literal, Optional
import json
import os


sys.path.append('..')
sys.path.append('../TRDP-unlearning')
dotenv.load_dotenv()
os.environ['WANDB_DISABLED'] = "true"
assert os.getenv('HF_TOKEN'), "HF_TOKEN environment variable must be set and non-empty"
#!huggingface-cli login --token ${HF_TOKEN}

from unlearner_lora_distillation import UnlearnerLoraDistillation  # FADE  # noqa: E402
from vision_unlearning.unlearner import UnlearnerLoraDirect  # Munba  # noqa: E402
from vision_unlearning.unlearner import UCE, ConceptType  # noqa: E402


from vision_unlearning.utils.parameter_attribution import ParameterAttributionMethodSaliency  # noqa: E402
from vision_unlearning.utils.logger import get_logger, setup_loggers  # noqa: E402
from vision_unlearning.datasets import UnlearnDatasetImagenette  # noqa: E402
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethod, GradientWeightingMethodSimple, GradientWeightingMethodMunba  # noqa: E402
from vision_unlearning.benchmarks.I_care import check_eval_results  # noqa: E402
from vision_unlearning.datasets.testbed import (  # noqa: E402
    get_target_overwrite,
    get_unlearned_model_folder,
    exists_unlearned_model,
    exists_unlearned_dataset,
    GeneratedDataset,
)
from vision_unlearning.utils.data_generation import generate_dataset  # noqa: E402


# Breeds
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'  # completed
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'  # completed
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 100
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'  # completed
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 50
replace_if_exists: bool = False
'''


# Scenes
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'  # completed
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'  # completed
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 100
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'  # completed
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 100
replace_if_exists: bool = False
'''

# People
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False

index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 200
replace_if_exists: bool = False

index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 400
replace_if_exists: bool = False
'''

# For people and distil, 400 epochs is good, but 100 is enough?
# For people and munba, lr 2e-4, 50 epochs is sometimes enough good, but sometimes not even 200 is enough...400 is definetly too much...
# For people and UCE, ???
# For dog and distil, 100 is good? 50 epochs is enough?
# For dog and munba, ???
# For dog and UCE, ???
# for scene and distil, 400 is good, but 100 is enough

# Running now:
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False


###
perform_dataset_generation: bool = True
generate_dataset_seeds = [42, 43, 44, 45]  # [42]  # [42, 43, 44, 45]
generate_dataset_limit_prompts: Optional[int] = None  # 5  # None
hub_model_id = None  # 'LeonardoBenitez/demo-vision-unlearning-' + 'fade' if method=='distil' else method  #None

# No need to change anything from now on...

# CLI: the assets folder can be overridden so the script runs from any directory.
_parser = argparse.ArgumentParser(description="Unlearn a model for a set of entities.")
_parser.add_argument("--base-folder", default="assets",
                     help="Path to the assets folder (default: 'assets' in the current directory).")
_args, _unknown = _parser.parse_known_args()
base_folder: str = _args.base_folder

# Basic params
with open(os.path.join(base_folder, f"metadata_{task}_2_enriched_filtered.json"), "r", encoding="utf-8") as f:
    metadata_filtered = json.load(f)

model_base_name = "CompVis/stable-diffusion-v1-4"
logger = get_logger('unlearning_main')
setup_loggers(modules_info=['unlearning'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Main loop
for index in range(index_start, index_start + max_identities):
    target = metadata_filtered[index]['name']
    output_dir = get_unlearned_model_folder(task, method, num_train_epochs, target, base_folder=base_folder)
    if task == 'people':
        dataset_base_path = os.path.join(base_folder, 'datasets/lfw_splits_filtered')
    elif task == 'breeds':
        dataset_base_path = os.path.join(base_folder, 'datasets/taras_breeds_splits_filtered')
    elif task == 'scenes':
        dataset_base_path = os.path.join(base_folder, 'datasets/SUN_splits_filtered')
    else:
        raise NotImplementedError()
    assert type(dataset_base_path) == str

    dataset_forget_name = f"{dataset_base_path}/{target}/train_forget"
    dataset_retain_name = f"{dataset_base_path}/{target}/train_retain"

    target_preprocessed, target_overwrite = get_target_overwrite(task, method, target)  # TODO: should this be done right after defining the target?

    # Evaluation set
    # Just for calculating basic metrics and debugging
    if (task == 'people'):
        validation_prompt = f'An image of {target_preprocessed}'
        example_prompts_forget = [
            f'An image of {target_preprocessed}',
            f'Photograph of {target_preprocessed.replace("_", " ")}; high definition',
            f'An picture of {target_preprocessed} in the rain',
            f'An picture of {target_preprocessed} running',
        ]
        example_prompts_retain = [
            f'An image of {target_overwrite}',
            f'Photograph of {target_overwrite}; high definition',
            f'An picture of {target_overwrite} in the rain',
            f'An picture of {target_overwrite} running',
        ]
    elif task == 'scenes':
        validation_prompt = f'An image of {target_preprocessed}'
        example_prompts_forget = [
            f'An image of {target_preprocessed}',
            f'Photograph of {target_preprocessed.replace("_", " ")}; high definition',
            f'An picture of {target_preprocessed} full of people',
            f'An picture of {target_preprocessed} during the night',
        ]
        example_prompts_retain = [
            f'An image of {target_overwrite}',
            f'Photograph of {target_overwrite}; high definition',
            f'An picture of {target_overwrite} full of people',
            f'An picture of {target_overwrite} during the night',
        ]
    elif task == 'breeds':
        validation_prompt = f'An image of {target_preprocessed}'
        target_preprocessed_next, _ = get_target_overwrite(task, method, metadata_filtered[(index + 1) % 100]['name'])
        example_prompts_forget = [
            f'An image of {target_preprocessed}',
            f'Photograph of {target_preprocessed.replace("_", " ")}; high definition',
            f'An picture of {target_preprocessed} in the rain',
            f'An picture of {target_preprocessed} running',
        ]
        example_prompts_retain = [
            f'An image of {target_preprocessed_next}',
            f'Photograph of {target_preprocessed_next}; high definition',
            f'An picture of {target_preprocessed_next} in the rain',
            f'An picture of {target_preprocessed_next} running',
        ]
    else:
        raise NotImplementedError()
    assert type(validation_prompt) == str
    assert type(example_prompts_forget) == list
    assert type(example_prompts_retain) == list

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    free_memory, total_memory = torch.cuda.mem_get_info()  # in GB
    logger.info(f"Free Memory: {free_memory / 1e9:.2f} GB")
    logger.info(f"Total Memory: {total_memory / 1e9:.2f} GB")

    ###########################################
    # Hyperparameters
    ###########################################
    hyperparameters: Dict[str, Any] = {
        "output_dir": output_dir,
        "hub_model_id": hub_model_id,
        "final_eval_prompts_forget": example_prompts_forget,
        "final_eval_prompts_retain": example_prompts_retain,
    }
    if method == 'uce':
        # TODO: these hyperparams are good for scenes
        # At least for UCE and for people, I know they have to be changed (I think i ran with erase 0.7, preserve 0.5, l0.5... we can see in the saved models)
        hyperparameters.update({
            "pretrained_model_name_or_path": model_base_name,
            "erase_scale": 30,
            "preserve_scale": 0.01,
            "lamb": 0.01,
            "edit_concepts": target_preprocessed,  # example: cat
            "guide_concepts": task,  # Optional. Example: animals. Default: same as edit_concepts
            "preserve_concepts": task,  # TODO: this isnt good... Example: lion; tiger; leopard
            "expand_prompts": False,
            "device": device,
            "save_entire_model": False,
        })
    else:
        hyperparameters.update({
            "model_name_or_path": model_base_name,
            "dataset_forget_name": dataset_forget_name,
            "dataset_retain_name": dataset_retain_name,

            "validation_prompt": f"An image of {target_preprocessed}",


            "dataloader_num_workers": 2,
            "resolution": 512,
            "num_validation_images": 1,

            "mixed_precision": "no",
            "learning_rate": 6e-4,
            "max_grad_norm": 5.0,
            
            "num_train_epochs": num_train_epochs,
            "validation_epochs": num_train_epochs + 1,  # No intermediate validation
            "checkpointing_steps": 10000,
            "lr_scheduler_type": "constant",
            "lr_warmup_steps": 0,
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "random_flip": True,
            
            "lora_r": 16,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "lora_alpha": 4,
            "lora_dropout": 0.2,
            
            "seed": 42,
        })
        if free_memory > 20e9:
            hyperparameters.update({
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 1,
            })
        elif free_memory > 14e9:
            hyperparameters.update({
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
            })
        else:
            hyperparameters.update({
                "per_device_train_batch_size": 2,  # 1,
                "gradient_accumulation_steps": 2,  # 4,
            })

    if free_memory > 20e9:
        logger.debug('Choosing hyperparams for free_memory>20e9')
        batch_size_inference = 50
    elif free_memory > 14e9:
        logger.debug('Choosing hyperparams for free_memory>14e9')
        batch_size_inference = 25  # 25
    else:
        logger.error('Too little GPU, diverting power from life support...')
        batch_size_inference = 25

    if method == 'distil':
        hyperparameters['overwritting_concept'] = target_overwrite
        hyperparameters['gradient_weighting_method'] = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0)
        unlearner = UnlearnerLoraDistillation(**hyperparameters)
    elif method == 'munba':
        hyperparameters['gradient_weighting_method'] = GradientWeightingMethodMunba()
        if task=='scenes':
            hyperparameters.update({
                "learning_rate": 1.5e-4,
                "max_grad_norm": 1.0,
                "lora_r": 4,
            })

        unlearner=UnlearnerLoraDirect(**hyperparameters)
    elif method == 'uce':
        if task=='breeds':
            hyperparameters.update({
                "erase_scale": 230,
                "preserve_scale": 1.2,
                "lamb": 0.1,
            })
        unlearner = UCE(**hyperparameters)
    else:
        raise NotImplementedError()


    if replace_if_exists or not exists_unlearned_model(task, method, num_train_epochs, target, base_folder=base_folder):
        logger.info(f"Overwritting the entity '{target}' by '{target_overwrite}'")
        logger.info(hyperparameters)
        eval_results = unlearner.train()
        logger.info('Evaluation metrics:\n' + pd.DataFrame([{'Name': r.metric_name, 'Value': r.metric_value} for r in eval_results]).to_markdown())

        ###########################################
        # Check metrics
        ###########################################
        check_eval_results(eval_results, 'Runtime training seconds', 10 * 3600, 'lt')
        check_eval_results(eval_results, 'Runtime training seconds', 60, 'gt')

        # TODO define ideal values for more combinations
        if task=='breeds':
            if method=='distil':
                check_eval_results(eval_results, 'ForgetSet clip score difference between original and unlearned mean', 3, 'gt')
                check_eval_results(eval_results, 'RetainSet clip score difference between original and unlearned mean', 0.5, 'lt')
        elif task=='people':
            if method=='munba':
                check_eval_results(eval_results, 'ForgetSet clip score difference between original and unlearned mean', 1.5, 'gt')
                check_eval_results(eval_results, 'RetainSet clip score difference between original and unlearned mean', 1, 'lt')
            elif method=='distil':
                check_eval_results(eval_results, 'ForgetSet clip score difference between original and unlearned mean', 5, 'gt')
                check_eval_results(eval_results, 'RetainSet clip score difference between original and unlearned mean', 1, 'lt')

        elif task=='scenes':
            if method=='distil':
                check_eval_results(eval_results, 'ForgetSet clip score difference between original and unlearned mean', 5, 'gt')
                check_eval_results(eval_results, 'RetainSet clip score difference between original and unlearned mean', 1, 'lt')

        #Metrics for task=people, method=uce, epochs=0, index=0
        #ForgetSet clip score difference between original and unlearned mean = 5.882541179656982
        #RetainSet clip score difference between original and unlearned mean = 1.421424388885498                
        
        print('-' * 50)
        print(f"Metrics for task={task}, method={method}, epochs={num_train_epochs}, index={index}")
        print(f"ForgetSet clip score difference between original and unlearned mean = {check_eval_results(eval_results, 'ForgetSet clip score difference between original and unlearned mean', 999, 'lt')}")
        print(f"RetainSet clip score difference between original and unlearned mean = {check_eval_results(eval_results, 'RetainSet clip score difference between original and unlearned mean', 999, 'lt')}")

        del unlearner
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


        free_memory, total_memory = torch.cuda.mem_get_info()  # in GB
        logger.info(f">>> AFTER TRAINING AFTER DEL UNLEARNER <<< Free Memory: {free_memory / 1e9:.2f} GB | Total Memory: {total_memory / 1e9:.2f} GB")
    else:
        logger.info(f'Skipping unlearning for "{target}" since already exists at "{output_dir}"')

    ###########################################
    # Generate dataset
    ###########################################
    if perform_dataset_generation:
        prompts = [f"An image of {get_target_overwrite(task, method, m['name'])[0]}" for m in metadata_filtered]
        if generate_dataset_limit_prompts is not None:
            prompts = prompts[:generate_dataset_limit_prompts]

        ds_entity = GeneratedDataset(
            task=task,
            target=target_preprocessed,
            method=method,
            num_train_epochs=num_train_epochs,
            base_folder=base_folder,
        )
        generated_dataset_output_path = ds_entity.folder_path

        if replace_if_exists or not ds_entity.exists(generate_dataset_seeds, prompts):
            logger.info(f'Generating images for unlearning "{target_preprocessed}" into folder "{generated_dataset_output_path}"')
            # Only generate lora_state='on' (unlearned model) images here.
            # Baseline lora_state='off' images are generated once per task by
            # 0_generate_dataset_original.py and stored in the shared baseline folder.
            model_pipeline: Optional[Any] = None
            if method == 'uce':
                model_generate_name: Optional[str] = None
                lora_generate_name: Optional[str] = None
                model_pipeline = UCE.get_pipeline_from_modified_weights(
                    pretrained_model_name_or_path=model_base_name,
                    device=device,
                    output_dir=output_dir,
                )
            else:
                model_generate_name = model_base_name
                lora_generate_name = output_dir
                model_pipeline = None

            filenames = [f'on_{seed}_{prompt}.png' for seed in generate_dataset_seeds for prompt in prompts]
            generate_dataset(  # type: ignore[arg-type]
                model_base_name=model_generate_name,
                lora_name=lora_generate_name,
                model_pipeline=model_pipeline,  # type: ignore[arg-type]
                prompts=prompts,
                output_path=generated_dataset_output_path,
                seeds=generate_dataset_seeds,
                filenames=filenames,
                batch_size=batch_size_inference,
                lora_requires_inversion=method == 'munba',
            )
            del model_pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f">>> AFTER GEN DATASET AFTER DEL <<< Free Memory: {free_memory / 1e9:.2f} GB | Total Memory: {total_memory / 1e9:.2f} GB")
        else:
            logger.info(f'Skipping dataset generation for "{target_preprocessed}" since already exists at "{generated_dataset_output_path}"')
        assert ds_entity.exists(generate_dataset_seeds, prompts)

    logger.info('-' * 100)
    logger.info(f"Finished unlearning and dataset generation for target '{target_preprocessed}' ({index}/{index_start + max_identities -1})")
    logger.info('-' * 100)

logger.info("ALL DONE YAY =D")
