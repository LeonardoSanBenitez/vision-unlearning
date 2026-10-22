"""What one unlearning session is configured with, as a value a test can hold.

This used to be eighty lines in the middle of ``pipeline_03_unlearn_model.py``, which asserts
``HF_TOKEN`` at import and therefore cannot be imported by a test at all. Nothing in it needs a
card, a token or a checkpoint: given a task, a method, an entity and the paths, it is a dictionary.
So it lives here, the pipeline calls it, and every field it decides is asserted directly.

Two things are deliberately *not* here:

* the objects -- the gradient weighting methods, the unlearner classes -- because constructing them
  imports torch, and this module must stay importable in the lite test tier;
* the settings that depend on the machine rather than on the session, namely the batch size and
  gradient accumulation of the methods whose batch is not part of their reference configuration.
  SPARE's batch of 4 is part of its published configuration and is set here; Munba's is still chosen
  by the pipeline from free video memory.

Every setting of a method on the benchmark's method axis -- UCE, SPARE (``distil``) and SalUn -- is its
authors' reference configuration, apart from the one strength dial equalization sets (UCE's
``erase_scale``, the trained methods' epoch count) and a short list of deviations each stated with its
reason where the value is set. The sources: UCE's two code lineages
(``rohitgandikota/unified-concept-editing`` at ``7c724d9d7d19190e9bd33fe62fb656f1df87cd8e``, current,
and at ``c34e5b9d69``, paper era); SPARE's arXiv 2602.07058v2 Appendix 0.A; SalUn's
``OPTML-Group/Unlearn-Saliency`` at ``a6e89fd17ff737c24c6aa16c9574469a234ffd91``.

The precondition on the training data is :func:`assert_split_captions_ready`, kept separate because
it reads the disk and the builder does not.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from vision_unlearning.benchmarks.I_care.configuration import (
    GENERATE_DATASET_SEEDS,
    type_task,
    type_unlearning_algorithm,
    uce_erase_scale,
)
from vision_unlearning.benchmarks.I_care.prompts import evaluation_prompts
from vision_unlearning.benchmarks.I_care.split_captions import assert_captions_prompted
from vision_unlearning.datasets.entity_names import canonical_entity, substitute_concept


#: Methods that train on the benchmark's image splits. UCE is a closed-form edit and reads none.
METHODS_WITH_TRAINING_DATA = ('distil', 'munba', 'salun')

#: The low-rank adapter trainers, which share one block of settings.
_LORA_METHODS = ('distil', 'munba')

#: The seed of every session's own final evaluation: the first generation seed, so the trainer's
#: coarse read-out compares original and unlearned images that start from the same latent.
EVALUATION_NOISE_SEED = GENERATE_DATASET_SEEDS[0]

#: The one training seed of every session (a binding choice: one training run per session). The
#: trainers seed their own generators with it, and `pipeline_03` seeds the global generators with it
#: under `vision_unlearning.utils.determinism.seed_training`.
TRAINING_SEED = 42


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
    uce_guide_concepts: Optional[str] = None,
    retained_entities: Optional[Sequence[str]] = None,
    uce_erase_scale_override: Optional[float] = None,
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
    @param uce_guide_concepts: UCE only. None keeps the default, the task's substitute concept;
        a string replaces it, which is how a control session reproduces an earlier configuration
        on the same entity. Refused for every other method.
    @param retained_entities: UCE only, and required for it: the task's other entities, as the
        metadata spells them, which UCE preserves -- the retention information SPARE and SalUn get
        from their retain splits. The target is refused if present.
    @param uce_erase_scale_override: UCE only: the strength dial, for equalization's sweep. None
        uses the task's value in ``configuration.uce_erase_scale``.
    @raises NotImplementedError: the method has no configuration here.
    """
    forget_prompts, retain_prompts = evaluation_prompts(task, entity)
    hyperparameters: Dict[str, Any] = {
        'output_dir': output_dir,
        'hub_model_id': hub_model_id,
        'final_eval_prompts_forget': forget_prompts,
        'final_eval_prompts_retain': retain_prompts,
        'evaluation_noise_seed': EVALUATION_NOISE_SEED,
    }

    for name, value in (('uce_guide_concepts', uce_guide_concepts), ('retained_entities', retained_entities),
                        ('uce_erase_scale_override', uce_erase_scale_override)):
        if value is not None and method != 'uce':
            raise ValueError(f'{name} is a UCE setting; method {method!r} has no such setting')
    if method == 'uce':
        if retained_entities is None:
            raise ValueError('UCE preserves the task\'s other entities; pass retained_entities')
        hyperparameters.update(_uce(task, entity, model_base_name, device, retained_entities,
                                    uce_guide_concepts, uce_erase_scale_override))
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


def _uce(
    task: type_task,
    entity: str,
    model_base_name: str,
    device: str,
    retained_entities: Sequence[str],
    guide_concepts: Optional[str] = None,
    erase_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """The closed-form edit, at its authors' configuration.

    ``guide_concepts`` is the task's substitute concept -- the same phrase the other two methods push
    the entity towards -- so the three methods are doing the same thing to the same concept.

    ``preserve_concepts`` is every other entity of the task, in canonical form, separated by ``;``:
    the retention information the trained methods receive through their retain splits. UCE sums one
    unnormalised preservation term per concept, so the per-concept weight follows its authors' rule
    when no scale is given, ``max(0.1, 1 / len(retain))`` -- 0.1 for 99 entities -- and ``lamb`` is
    0.5, the value both of its code lineages use. ``erase_scale`` is the strength dial.

    Deviations from the paper-era code, which the current lineage this library ports shares: the
    empty prompt is not added to the retain list (the weight is 0.1 either way), and each concept
    contributes one embedding rather than outer products over padded token positions.
    """
    target = canonical_entity(task, entity)
    preserved = [canonical_entity(task, other) for other in retained_entities]
    if target in preserved:
        raise ValueError(f'the target {target!r} is among the entities UCE is asked to preserve')
    if not preserved:
        raise ValueError('UCE was given no entity to preserve')
    return {
        'pretrained_model_name_or_path': model_base_name,
        'erase_scale': uce_erase_scale[task] if erase_scale is None else erase_scale,
        'preserve_scale': max(0.1, 1.0 / len(preserved)),
        'lamb': 0.5,
        'edit_concepts': target,
        'guide_concepts': substitute_concept(task) if guide_concepts is None else guide_concepts,
        'preserve_concepts': ';'.join(preserved),
        'expand_prompts': False,
        'device': device,
        'save_entire_model': False,
    }


def _salun(
    task: type_task,
    model_base_name: str,
    device: str,
    num_train_epochs: int,
    dataset_forget_name: str,
    dataset_retain_name: str,
) -> Dict[str, Any]:
    """SalUn at its authors' class-forgetting configuration, on the benchmark's splits.

    Learning rate 1e-5, mask threshold 0.5, retain weight 0.5 and mask guidance 7.5 are the
    reference's (its README command; the argparse default of 0.1 for the retain weight is not what the
    authors' command runs). Two deviations, each for a reason that does not depend on any result:
    only the cross-attention tensors are trained, which is ESD's choice -- its training script ships
    in SalUn's diffusion directory -- when erasure must be specific to the prompt, as it is here; and
    the batch is 1 rather than 8, because on forget sets of 5, 20 and 35 images a batch of 8 cuts the
    updates per epoch five- to sevenfold at the same epoch dial. The fine-tune runs in full precision,
    without TensorFloat-32 matrix products: reduced precision moved SPARE's results beyond the spread
    between two ordinary trainings, so neither trained method uses one.
    """
    return {
        'pretrained_model_name_or_path': model_base_name,
        'dataset_forget_name': dataset_forget_name,
        'dataset_retain_name': dataset_retain_name,
        'overwriting_concept': substitute_concept(task),
        'train_method': 'xattn',
        'learning_rate': 1e-5,
        'mask_threshold': 0.5,
        'alpha': 0.5,
        'mask_guidance': 7.5,
        'num_train_epochs': num_train_epochs,
        'resolution': 512,
        'random_flip': True,
        # workers must pickle the transform; a nested function cannot be
        'dataloader_num_workers': 0,
        'per_device_train_batch_size': 1,
        'allow_tf32': False,
        'seed': TRAINING_SEED,
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
    """The two low-rank adapter trainers, and what distinguishes them.

    SPARE (``distil``) runs its only published configuration, arXiv 2602.07058v2 Appendix 0.A:
    learning rate 1e-4, constant, no warm-up; LoRA rank 4, alpha 8, dropout 0.2, Gaussian
    initialisation; batch 4; Adam betas 0.9 and 0.999, epsilon 1e-8, weight decay 1e-2; horizontal
    flip. The paper is silent on gradient clipping and the target modules, so those are the library
    defaults (1.0; the four attention projections). It trains in full precision, not the bfloat16 mixed
    precision it offers, which moved its results beyond the spread between two ordinary trainings. Munba is
    not on the benchmark's method axis and keeps the settings it was run with.
    """
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
        'seed': TRAINING_SEED,
    }

    if method == 'distil':
        # Both sides of the distillation pair are one template with a different concept: the forget
        # side is conditioned on the phrase the images are generated and evaluated with rather than
        # on whatever the dataset happens to store, and the target on the substitute concept.
        settings['overwritting_concept'] = substitute_concept(task)
        settings['forget_concept'] = canonical_entity(task, entity)
        settings.update({
            'learning_rate': 1e-4,
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.2,
            'max_grad_norm': 1.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'adam_weight_decay': 1e-2,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 1,
        })
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
