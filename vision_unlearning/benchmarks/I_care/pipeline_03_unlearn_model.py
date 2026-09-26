
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
from typing import Any, Dict, List, Literal, Optional, get_args
import json
import os


sys.path.append('..')
sys.path.append('../TRDP-unlearning')
dotenv.load_dotenv()
os.environ['WANDB_DISABLED'] = "true"
assert os.getenv('HF_TOKEN'), "HF_TOKEN environment variable must be set and non-empty"
#!huggingface-cli login --token ${HF_TOKEN}

from vision_unlearning.benchmarks.I_care.configuration import type_unlearning_algorithm  # noqa: E402
from vision_unlearning.benchmarks.I_care.session_config import (  # noqa: E402
    assert_split_captions_ready,
    session_hyperparameters,
)
from vision_unlearning.unlearner import Unlearner, UnlearnerSpare  # noqa: E402
from vision_unlearning.unlearner import UnlearnerLoraDirect  # Munba  # noqa: E402
from vision_unlearning.unlearner import UCE, ConceptType  # noqa: E402
from vision_unlearning.unlearner import SalUn  # noqa: E402


from vision_unlearning.utils.parameter_attribution import ParameterAttributionMethodSaliency  # noqa: E402
from vision_unlearning.utils.logger import get_logger, setup_loggers  # noqa: E402
from vision_unlearning.datasets import UnlearnDatasetImagenette  # noqa: E402
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethod, GradientWeightingMethodSimple, GradientWeightingMethodMunba  # noqa: E402
from vision_unlearning.benchmarks.I_care import check_eval_results  # noqa: E402
from vision_unlearning.benchmarks.I_care.configuration import ALGORITHM_REGISTRY  # noqa: E402
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

# Defaults, overridable on the command line -- see the parser below.
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'
method: type_unlearning_algorithm = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False


###
perform_dataset_generation: bool = True
generate_dataset_seeds = [42, 43, 44, 45]  # [42]  # [42, 43, 44, 45]
generate_dataset_limit_prompts: Optional[int] = None  # 5  # None
hub_model_id = None  # 'LeonardoBenitez/demo-vision-unlearning-' + 'fade' if method=='distil' else method  #None

# No need to change anything from now on...

# CLI. Every value above can be set here, so a campaign never edits this file between runs.
#
# It matters that this parser is STRICT. It used to declare only --base-folder and read the
# command line with `parse_known_args()`, discarding everything else -- while
# `pipeline_11_run_all.sh`, the documented end-to-end reference, invoked this script nine times
# with --task/--method/--num-train-epochs. All nine silently ran the defaults above instead, so
# the script that records how the corpus was produced could not produce it, and said nothing.
# A parser that ignores what it does not recognise turns a typo, a renamed flag and a
# never-implemented flag into the same clean exit.
_parser = argparse.ArgumentParser(description="Unlearn a model for a set of entities.")
_parser.add_argument("--base-folder", default="assets",
                     help="Path to the assets folder (default: 'assets' in the current directory).")
_parser.add_argument("--task", default=task, choices=['scenes', 'breeds', 'people'],
                     help=f"Which task's entities to unlearn (default: {task}).")
_parser.add_argument("--method", default=method, choices=list(get_args(type_unlearning_algorithm)),
                     help=f"Unlearning method (default: {method}).")
_parser.add_argument("--num-train-epochs", type=int, default=num_train_epochs,
                     help=f"Epochs, ignored by the closed-form methods (default: {num_train_epochs}).")
_parser.add_argument("--index-start", type=int, default=index_start,
                     help=f"First entity index in the task metadata (default: {index_start}).")
_parser.add_argument("--max-identities", type=int, default=max_identities,
                     help=f"How many entities to process from --index-start (default: {max_identities}).")
_parser.add_argument("--replace-if-exists", action="store_true", default=replace_if_exists,
                     help="Re-run a session whose artifact is already on disk.")
_parser.add_argument("--batch-size-inference", type=int, default=None,
                     help="Images generated per forward pass. Left unset, it is chosen from "
                          "free video memory, which makes the images depend on the card and on "
                          "what else was running -- see the note at the selection site. Set it "
                          "for any run whose images must be comparable with another run's.")
_args = _parser.parse_args()

base_folder: str = _args.base_folder
task = _args.task
method = _args.method
num_train_epochs = _args.num_train_epochs
index_start = _args.index_start
max_identities = _args.max_identities
replace_if_exists = _args.replace_if_exists
batch_size_inference_override: Optional[int] = _args.batch_size_inference

# Basic params
with open(os.path.join(base_folder, f"metadata_{task}_2_enriched_filtered.json"), "r", encoding="utf-8") as f:
    metadata_filtered = json.load(f)

model_base_name = "CompVis/stable-diffusion-v1-4"
logger = get_logger('unlearning_main')
logger.info(
    'session: task=%s method=%s epochs=%s entities=[%s, %s)',
    task, method, num_train_epochs, index_start, index_start + max_identities,
)
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


    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    free_memory, total_memory = torch.cuda.mem_get_info()  # in GB
    logger.info(f"Free Memory: {free_memory / 1e9:.2f} GB")
    logger.info(f"Total Memory: {total_memory / 1e9:.2f} GB")

    ###########################################
    # Hyperparameters
    ###########################################
    assert_split_captions_ready(task, method, dataset_forget_name, dataset_retain_name)
    hyperparameters: Dict[str, Any] = session_hyperparameters(
        task,
        method,
        target,
        output_dir=output_dir,
        dataset_forget_name=dataset_forget_name,
        dataset_retain_name=dataset_retain_name,
        model_base_name=model_base_name,
        device=device,
        num_train_epochs=num_train_epochs,
        hub_model_id=hub_model_id,
    )
    if method in ('distil', 'munba'):
        # How much fits on this card: a property of the machine, not of the session, which is why
        # it is the one piece of the configuration that did not move out of this file.
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

    # The generation batch size is NOT only a performance knob. `generate_dataset` draws one
    # noise tensor per batch, so the batch size decides the initial noise of every image: the
    # same prompt at the same seed in a batch of 8 and a batch of 50 gives different pictures.
    # Choosing it from free memory therefore makes the output depend on which card the job
    # landed on and on what else happened to be running -- which is how a shared baseline
    # generated at one size and per-entity images generated at another came to be compared
    # pair by pair. Any run whose images must be comparable with another run passes
    # --batch-size-inference explicitly; the memory-derived values below remain the default so
    # that nothing existing changes behaviour.
    if batch_size_inference_override is not None:
        logger.info('batch_size_inference pinned to %s by the caller', batch_size_inference_override)
        batch_size_inference = batch_size_inference_override
    elif free_memory > 20e9:
        logger.debug('Choosing hyperparams for free_memory>20e9')
        batch_size_inference = 50
    elif free_memory > 14e9:
        logger.debug('Choosing hyperparams for free_memory>14e9')
        batch_size_inference = 25  # 25
    else:
        logger.error('Too little GPU, diverting power from life support...')
        batch_size_inference = 25

    unlearner: Unlearner
    if method == 'distil':
        hyperparameters['gradient_weighting_method'] = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0)
        unlearner = UnlearnerSpare(**hyperparameters)
    elif method == 'munba':
        hyperparameters['gradient_weighting_method'] = GradientWeightingMethodMunba()
        unlearner=UnlearnerLoraDirect(**hyperparameters)
    elif method == 'uce':
        unlearner = UCE(**hyperparameters)
    elif method == 'salun':
        unlearner = SalUn(**hyperparameters)
    else:
        raise NotImplementedError()


    artifact_filename = ALGORITHM_REGISTRY[method].artifact_filename
    if replace_if_exists or not exists_unlearned_model(
        task, method, num_train_epochs, target, artifact_filename, base_folder=base_folder
    ):
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
            artifact_kind=ALGORITHM_REGISTRY[method].artifact_kind,
            artifact_filename=ALGORITHM_REGISTRY[method].artifact_filename,
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
