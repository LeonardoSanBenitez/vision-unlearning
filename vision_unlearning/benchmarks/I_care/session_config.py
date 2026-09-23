"""What one unlearning session is configured with, as a value a test can hold.

This used to be eighty lines in the middle of ``pipeline_03_unlearn_model.py``, which asserts
``HF_TOKEN`` at import and therefore cannot be imported by a test at all. Nothing in it needs a
card, a token or a checkpoint: given a task, a method, an entity and the paths, it is a dictionary.
So it lives here, the pipeline calls it, and every field it decides is asserted directly.

Two things are deliberately *not* here:

* the objects -- the gradient weighting methods, the unlearner classes -- because constructing them
  imports torch, and this module must stay importable in the lite test tier;
* the settings that depend on the machine rather than on the session, namely the batch size and
  gradient accumulation chosen from the free video memory. Those are a property of where the run
  happens, and the pipeline still decides them.

The precondition on the training data is :func:`assert_split_captions_ready`, kept separate because
it reads the disk and the builder does not.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from vision_unlearning.benchmarks.I_care.configuration import (
    type_task,
    type_unlearning_algorithm,
)
from vision_unlearning.benchmarks.I_care.prompts import evaluation_prompts
from vision_unlearning.benchmarks.I_care.split_captions import assert_captions_prompted
from vision_unlearning.datasets.entity_names import canonical_entity, substitute_concept


#: Methods that train on the benchmark's image splits. UCE is a closed-form edit and reads none.
METHODS_WITH_TRAINING_DATA = ('distil', 'munba', 'salun')

#: The low-rank adapter trainers, which share one block of settings.
_LORA_METHODS = ('distil', 'munba')


def session_hyperparameters(
    task: type_task,
    method: type_unlearning_algorithm,
    entity: str,
    *,
    output_dir: str,
    dataset_forget_name: str,
    dataset_retain_name: str,
    model_base_name: str,
    device: str,
    num_train_epochs: int,
    hub_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the hyperparameters of one unlearning session.

    @param task: which of the three entity families the entity belongs to.
    @param method: the unlearning algorithm.
    @param entity: the entity as the task metadata spells it; it is canonicalised here.
    @param output_dir: where the unlearned model is written.
    @param dataset_forget_name: the forget split folder. Ignored by methods that read no data.
    @param dataset_retain_name: the retain split folder. Ignored by methods that read no data.
    @param model_base_name: the base checkpoint every method starts from.
    @param device: 'cuda' or 'cpu'.
    @param num_train_epochs: epochs, for the methods that train.
    @param hub_model_id: the model hub identifier, or None not to push.
    @raises NotImplementedError: the method has no configuration here.
    """
    forget_prompts, retain_prompts = evaluation_prompts(task, entity)
    hyperparameters: Dict[str, Any] = {
        'output_dir': output_dir,
        'hub_model_id': hub_model_id,
        'final_eval_prompts_forget': forget_prompts,
        'final_eval_prompts_retain': retain_prompts,
    }

    if method == 'uce':
        hyperparameters.update(_uce(task, entity, model_base_name, device))
    elif method == 'salun':
        hyperparameters.update(_salun(
            task, model_base_name, device, num_train_epochs,
            dataset_forget_name, dataset_retain_name,
        ))
    elif method in _LORA_METHODS:
        hyperparameters.update(_lora(
            task, method, entity, model_base_name, num_train_epochs,
            dataset_forget_name, dataset_retain_name, forget_prompts[0],
        ))
    else:
        raise NotImplementedError(f'no session configuration for method {method!r}')

    return hyperparameters


def _uce(task: type_task, entity: str, model_base_name: str, device: str) -> Dict[str, Any]:
    """The closed-form edit.

    ``guide_concepts`` is the task's substitute concept -- the same phrase the other two methods
    push the entity towards -- so the three methods are doing the same thing to the same concept.
    It used to be the literal task name, i.e. the edit was aimed at the embedding of the word
    ``breeds``.

    No preserve concepts are set. The preserve term used to be the literal task name as well, which
    regularised the edit towards a word that means nothing here. Setting it to the retained entities
    instead -- what the paper describes -- changes how strong the edit is and therefore needs
    ``erase_scale`` re-tuned with it, which is the equalization task's job and not this one's.
    """
    settings: Dict[str, Any] = {
        'pretrained_model_name_or_path': model_base_name,
        'erase_scale': 30,
        'preserve_scale': 0.01,
        'lamb': 0.01,
        'edit_concepts': canonical_entity(task, entity),
        'guide_concepts': substitute_concept(task),
        'preserve_concepts': None,
        'expand_prompts': False,
        'device': device,
        'save_entire_model': False,
    }
    if task == 'breeds':
        settings.update({'erase_scale': 230, 'preserve_scale': 1.2, 'lamb': 0.1})
    return settings


def _salun(
    task: type_task,
    model_base_name: str,
    device: str,
    num_train_epochs: int,
    dataset_forget_name: str,
    dataset_retain_name: str,
) -> Dict[str, Any]:
    """SalUn takes the benchmark's existing splits and the concept to push towards, nothing else."""
    return {
        'pretrained_model_name_or_path': model_base_name,
        'dataset_forget_name': dataset_forget_name,
        'dataset_retain_name': dataset_retain_name,
        'overwriting_concept': substitute_concept(task),
        'train_method': 'xattn',
        'num_train_epochs': num_train_epochs,
        'resolution': 512,
        'random_flip': True,
        # workers must pickle the transform; a nested function cannot be
        'dataloader_num_workers': 0,
        'per_device_train_batch_size': 1,
        'device': device,
    }


def _lora(
    task: type_task,
    method: type_unlearning_algorithm,
    entity: str,
    model_base_name: str,
    num_train_epochs: int,
    dataset_forget_name: str,
    dataset_retain_name: str,
    validation_prompt: str,
) -> Dict[str, Any]:
    """The two low-rank adapter trainers, and what distinguishes them."""
    settings: Dict[str, Any] = {
        'model_name_or_path': model_base_name,
        'dataset_forget_name': dataset_forget_name,
        'dataset_retain_name': dataset_retain_name,
        'validation_prompt': validation_prompt,
        'dataloader_num_workers': 2,
        'resolution': 512,
        'num_validation_images': 1,
        'mixed_precision': 'no',
        'learning_rate': 6e-4,
        'max_grad_norm': 5.0,
        'num_train_epochs': num_train_epochs,
        'validation_epochs': num_train_epochs + 1,  # No intermediate validation
        'checkpointing_steps': 10000,
        'lr_scheduler_type': 'constant',
        'lr_warmup_steps': 0,
        'save_strategy': 'epoch',
        'save_total_limit': 2,
        'random_flip': True,
        'lora_r': 16,
        'target_modules': ['to_k', 'to_q', 'to_v', 'to_out.0'],
        'lora_alpha': 4,
        'lora_dropout': 0.2,
        'seed': 42,
    }

    if method == 'distil':
        # Both sides of the distillation pair are one template with a different concept: the forget
        # side is conditioned on the phrase the images are generated and evaluated with rather than
        # on whatever the dataset happens to store, and the target on the substitute concept.
        settings['overwritting_concept'] = substitute_concept(task)
        settings['forget_concept'] = canonical_entity(task, entity)
    elif method == 'munba' and task == 'scenes':
        settings.update({
            'learning_rate': 1.5e-4,
            'max_grad_norm': 1.0,
            'lora_r': 4,
        })

    return settings


def assert_split_captions_ready(
    task: type_task,
    method: type_unlearning_algorithm,
    dataset_forget_name: str,
    dataset_retain_name: str,
) -> None:
    """Refuse a session whose training splits are not captioned with the prompted form.

    Called before the model is built, so a wrong caption is a message naming the file and the row
    rather than a model that trained on the wrong string and looks fine. Methods that read no
    training data are not checked.

    @raises ValueError: a split is missing, empty, or carries a caption that is not the prompt its
        image's entity is generated with.
    """
    if method not in METHODS_WITH_TRAINING_DATA:
        return
    for folder in (dataset_forget_name, dataset_retain_name):
        assert_captions_prompted(folder, task)
