
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
    TRAINING_SEED,
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
    ARTIFACT_KIND_LOADERS,
    get_target_overwrite,
    get_unlearned_model_folder,
    exists_unlearned_model,
    exists_unlearned_dataset,
    GeneratedDataset,
)
from vision_unlearning.unlearner.loaders import get_partial_weights_loader  # noqa: E402
from vision_unlearning.utils.data_generation import generate_dataset  # noqa: E402
from vision_unlearning.utils.determinism import seed_training  # noqa: E402
from vision_unlearning.artifact import DEFAULT_REMOTE_REPOSITORY_NAME  # noqa: E402
from vision_unlearning.benchmarks.I_care.configuration import (  # noqa: E402
    GENERATE_DATASET_SEEDS,
    GENERATION_BATCH_SIZE,
    GENERATION_NOISE,
)
from vision_unlearning.benchmarks.I_care.provenance import producer_facts, write_provenance  # noqa: E402
from vision_unlearning.benchmarks.I_care.metadata import receivers_in_order  # noqa: E402
from vision_unlearning.benchmarks.I_care.upload import (  # noqa: E402
    assert_upload_allowed,
    remote_path_of,
    upload_folder,
)
from vision_unlearning.datasets.entity_names import generation_prompt  # noqa: E402
from vision_unlearning.datasets.testbed import get_generated_dataset_file  # noqa: E402


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
generate_dataset_seeds = list(GENERATE_DATASET_SEEDS)
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
                     help="Images generated per forward pass. Left unset, it is the benchmark's "
                          f"configuration.GENERATION_BATCH_SIZE ({GENERATION_BATCH_SIZE}), the size the "
                          "baseline is generated at, because the batch shape changes the pixels.")
_parser.add_argument("--train-batch-size", type=int, default=None,
                     help="Training batch per device. Left unset, a method whose reference configuration "
                          "fixes it (SPARE: 4) uses that, and Munba chooses it from free video memory. A "
                          "batch that does not fit the card fails the session; it is never shrunk.")
_parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                     help="Training steps accumulated per update. Same defaults as --train-batch-size.")
_parser.add_argument("--uce-erase-scale", type=float, default=None,
                     help="UCE only: the strength dial. Left unset, configuration.uce_erase_scale of the task.")
_parser.add_argument("--repository", default=DEFAULT_REMOTE_REPOSITORY_NAME,
                     help=f"HuggingFace dataset repository to upload to (default: {DEFAULT_REMOTE_REPOSITORY_NAME}).")
_parser.add_argument("--upload", action="store_true", default=False,
                     help="Upload each session's model folder and image folder once each is complete, "
                          "checked by content. Only the benchmark's current corpus may be written to; any "
                          "other --repository is refused at start-up.")
_parser.add_argument("--pins", default=None,
                     help="The run's pins.json, whose sha256 every provenance.json records.")
_parser.add_argument("--freeze", default=None,
                     help="The run's environment freeze, whose sha256 every provenance.json records.")
_parser.add_argument("--uce-guide-concepts", default=None,
                     help="UCE only: the concept the edit pushes the entity towards. Left unset it is "
                          "the task's substitute concept. Set to the task name ('people', 'breeds', "
                          "'scenes') to reproduce the configuration used before that change, for a "
                          "control session on the same entity. Refused for any other method.")
_parser.add_argument("--save-at-epochs", type=int, nargs="+", default=None,
                     help="Trained methods only: also save the weights after each of these epochs, into "
                          "epoch-N (SPARE) or epoch_N (SalUn) inside the model folder, each with its own "
                          "provenance.json. Equalization trains once to the top of a ladder this way.")
_parser.add_argument("--no-outcome-checks", action="store_true", default=False,
                     help="Skip the checks of the trained model's runtime and evaluation scores against fixed "
                          "thresholds. Those thresholds were tuned on an earlier corpus; a calibration run at "
                          "a weak or strong setting must be measured and reported, not stopped by them.")
_parser.add_argument("--receivers", nargs="+", default=None,
                     help="Generate the images of these entities' prompts only (metadata names), in the task's "
                          "order: a read-out, as equalization takes one. Left unset, every prompt. Never uploaded.")
_args = _parser.parse_args()
if _args.receivers is not None and _args.upload:
    _parser.error("--receivers makes a read-out, which is never uploaded; drop --upload")

base_folder: str = _args.base_folder
task = _args.task
method = _args.method
num_train_epochs = _args.num_train_epochs
index_start = _args.index_start
max_identities = _args.max_identities
replace_if_exists = _args.replace_if_exists
batch_size_inference_override: Optional[int] = _args.batch_size_inference
uce_guide_concepts: Optional[str] = _args.uce_guide_concepts
if _args.upload:
    # Before any model is loaded: an upload aimed somewhere forbidden stops here.
    assert_upload_allowed(_args.repository)

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
        uce_guide_concepts=uce_guide_concepts,
        retained_entities=[m['name'] for m in metadata_filtered if m['name'] != target] if method == 'uce' else None,
        uce_erase_scale_override=_args.uce_erase_scale,
        save_at_epochs=_args.save_at_epochs,
    )
    if uce_guide_concepts is not None:
        logger.info('UCE guide concept set by the caller to %r (default: the substitute concept)', uce_guide_concepts)
    if _args.train_batch_size is not None or _args.gradient_accumulation_steps is not None:
        if method == 'uce':
            raise ValueError('UCE is a closed-form edit; it has no training batch')
        if _args.train_batch_size is not None:
            hyperparameters['per_device_train_batch_size'] = _args.train_batch_size
        if _args.gradient_accumulation_steps is not None:
            hyperparameters['gradient_accumulation_steps'] = _args.gradient_accumulation_steps
        logger.info('training batch set by the caller: %s x %s', hyperparameters.get('per_device_train_batch_size'),
                    hyperparameters.get('gradient_accumulation_steps', 1))
    elif method == 'munba':
        # How much fits on this card: a property of the machine, not of the session. SPARE's batch
        # is part of its reference configuration and is set in session_config instead.
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

    # The generation batch size is NOT only a performance knob: the batch shape changes the
    # kernels' arithmetic and so the pixels, even with every image's noise fixed. It is the size the
    # baseline was generated at unless the caller says otherwise, never one chosen from free memory.
    batch_size_inference = GENERATION_BATCH_SIZE if batch_size_inference_override is None else batch_size_inference_override
    logger.info('generation batch size %s, noise %s', batch_size_inference, GENERATION_NOISE)

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
        # Deterministic kernels and seeded global generators, so a session retrained on the same card
        # model reproduces its weights byte for byte. The global seeding is what reaches draws a
        # trainer does not seed itself: SalUn's data loaders shuffle, and its random flip draws, from
        # torch's global generator.
        seed_training(TRAINING_SEED)
        eval_results = unlearner.train()
        logger.info('Evaluation metrics:\n' + pd.DataFrame([{'Name': r.metric_name, 'Value': r.metric_value} for r in eval_results]).to_markdown())
        model_facts = producer_facts(
            precision=str(hyperparameters['mixed_precision'] if hyperparameters.get('mixed_precision') not in (None, 'no')
                          else ('tf32' if hyperparameters.get('allow_tf32') else 'float32')),
            pins_path=_args.pins, freeze_path=_args.freeze, producer='pipeline_03',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), training_seed=TRAINING_SEED,
            task=task, method=method, entity=target, num_train_epochs=num_train_epochs,
            configuration={k: v for k, v in hyperparameters.items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        )
        write_provenance(output_dir, model_facts)
        # Each intermediate save is the model of a run stopped at that epoch (a constant learning rate), so
        # it records that epoch count; the folder name differs between the two trainers.
        for saved_epoch in _args.save_at_epochs or []:
            for saved_name in (f'epoch-{saved_epoch}', f'epoch_{saved_epoch}'):
                if os.path.isdir(os.path.join(output_dir, saved_name)):
                    write_provenance(os.path.join(output_dir, saved_name), {**model_facts, 'num_train_epochs': saved_epoch})

        ###########################################
        # Check metrics
        ###########################################
        if not _args.no_outcome_checks:
            check_eval_results(eval_results, 'Runtime training seconds', 10 * 3600, 'lt')
            check_eval_results(eval_results, 'Runtime training seconds', 60, 'gt')

        # TODO define ideal values for more combinations
        if _args.no_outcome_checks:
            pass
        elif task=='breeds':
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
        prompts = [generation_prompt(task, name)
                   for name in receivers_in_order([m['name'] for m in metadata_filtered], _args.receivers)]
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
            # Which route rebuilds the unlearned model is a property of the ARTIFACT the method
            # writes, declared once in ALGORITHM_REGISTRY, and not of the method's name. UCE and
            # SalUn both write a file of modified denoiser tensors; keying the branch on the literal
            # 'uce' sent SalUn down the adapter route, where it looked for a
            # pytorch_lora_weights.safetensors that SalUn never writes and raised after the training
            # had already finished. `pipeline_04_generate_dataset.py` and `datasets/testbed.py`
            # both already choose the route this way; this script was the one site that did not.
            artifact_kind = ALGORITHM_REGISTRY[method].artifact_kind
            if artifact_kind not in ARTIFACT_KIND_LOADERS:
                raise ValueError(
                    f'No load route is declared for artifact kind {artifact_kind!r} (method '
                    f'{method!r}). Known kinds: {", ".join(sorted(ARTIFACT_KIND_LOADERS))}. A new '
                    'method must declare one rather than fall through to the adapter route.'
                )
            model_pipeline: Optional[Any] = None
            model_generate_name: Optional[str] = None
            lora_generate_name: Optional[str] = None
            if ARTIFACT_KIND_LOADERS[artifact_kind] == 'partial_weights_loader':
                load_pipeline = get_partial_weights_loader(ALGORITHM_REGISTRY[method].artifact_filename)
                model_pipeline = load_pipeline(
                    pretrained_model_name_or_path=model_base_name,
                    device=device,
                    output_dir=output_dir,
                )
            else:
                model_generate_name = model_base_name
                lora_generate_name = output_dir

            filenames = [get_generated_dataset_file('on', seed, prompt) for seed in generate_dataset_seeds for prompt in prompts]
            generate_dataset(  # type: ignore[arg-type]
                model_base_name=model_generate_name,
                lora_name=lora_generate_name,
                model_pipeline=model_pipeline,  # type: ignore[arg-type]
                prompts=prompts,
                output_path=generated_dataset_output_path,
                seeds=generate_dataset_seeds,
                filenames=filenames,
                batch_size=batch_size_inference,
                lora_requires_inversion=artifact_kind == 'lora_adapter_inverted',
                noise=GENERATION_NOISE,
                provenance=producer_facts(
                    precision='float16', pins_path=_args.pins, freeze_path=_args.freeze, producer='pipeline_03',
                    task=task, method=method, entity=target, num_train_epochs=num_train_epochs,
                ),
            )
            del model_pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f">>> AFTER GEN DATASET AFTER DEL <<< Free Memory: {free_memory / 1e9:.2f} GB | Total Memory: {total_memory / 1e9:.2f} GB")
        else:
            logger.info(f'Skipping dataset generation for "{target_preprocessed}" since already exists at "{generated_dataset_output_path}"')
        assert ds_entity.exists(generate_dataset_seeds, prompts)
        if _args.upload:
            token = os.getenv('HF_TOKEN') or None
            model_files = sorted(
                os.path.relpath(os.path.join(root, name), output_dir)
                for root, _, files in os.walk(output_dir) for name in files
            )
            upload_folder(output_dir, remote_path_of(output_dir, base_folder), _args.repository, token, model_files)
            image_files = [get_generated_dataset_file('on', seed, prompt) for seed in generate_dataset_seeds for prompt in prompts]
            upload_folder(generated_dataset_output_path, remote_path_of(generated_dataset_output_path, base_folder),
                          _args.repository, token, image_files + ['metadata.jsonl', 'provenance.json'])
            logger.info('Uploaded and checked by content: %s and %s', output_dir, generated_dataset_output_path)

    logger.info('-' * 100)
    logger.info(f"Finished unlearning and dataset generation for target '{target_preprocessed}' ({index}/{index_start + max_identities -1})")
    logger.info('-' * 100)

logger.info("ALL DONE YAY =D")
