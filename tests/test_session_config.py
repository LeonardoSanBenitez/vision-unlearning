"""Every field of an unlearning session's configuration, asserted directly.

The three methods on the benchmark's method axis -- UCE, SPARE (``distil``) and SalUn -- run their
authors' reference configurations apart from the strength dial; :class:`TestTheReferenceConfigurations`
holds each value against the published source, and fails on the code before, which ran all three at
settings nobody published (SPARE at rate 6e-4, rank 16, alpha 4, clipping 5.0; UCE at ridge 0.01 with
the task word as its preserve set; SalUn at retain weight 0.1). Munba is not on the axis and keeps the
settings it was run with.

The per-machine settings of Munba -- batch size and gradient accumulation, chosen from free video
memory -- are deliberately not here; they stay in the pipeline. SPARE's batch of 4 is part of its
published configuration, so it is.
"""
import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import session_config
from vision_unlearning.benchmarks.I_care.prompts import evaluation_prompts
from vision_unlearning.datasets.entity_names import canonical_entity, substitute_concept


_MODEL = 'CompVis/stable-diffusion-v1-4'
_TASKS_AND_ENTITIES = [
    ('people', 'Mark_Philippoussis'),
    ('breeds', 'bouvier des flandres dog'),
    ('scenes', 'football_field'),
]
_OTHERS = {
    'people': ['George_W_Bush', 'Colin_Powell'],
    'breeds': ['basenji dog', 'kuvasz dog'],
    'scenes': ['abbey', 'badlands'],
}


def _config(task: str, method: str, entity: str, **overrides: Any) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        'output_dir': 'assets/models/out',
        'dataset_forget_name': 'assets/datasets/x/train_forget',
        'dataset_retain_name': 'assets/datasets/x/train_retain',
        'model_base_name': _MODEL,
        'device': 'cuda',
        'num_train_epochs': 100,
        'hub_model_id': None,
    }
    if method == 'uce':
        settings.setdefault('retained_entities', _OTHERS[task])
    settings.update(overrides)
    return session_config.session_hyperparameters(task, method, entity, **settings)  # type: ignore[arg-type]


class TestUce:
    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_every_field(self, task: str, entity: str) -> None:
        config = _config(task, 'uce', entity, num_train_epochs=0)

        forget, retain = evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert config == {
            'output_dir': 'assets/models/out',
            'hub_model_id': None,
            'final_eval_prompts_forget': forget,
            'final_eval_prompts_retain': retain,
            'evaluation_noise_seed': 42,
            'pretrained_model_name_or_path': _MODEL,
            'erase_scale': {'people': 1500.0, 'scenes': 1500.0, 'breeds': 1150.0}[task],
            'preserve_scale': 0.5,     # max(0.1, 1/2) for the two retained entities of this fixture
            'lamb': 0.5,
            'edit_concepts': canonical_entity(task, entity),  # type: ignore[arg-type]
            'guide_concepts': substitute_concept(task),       # type: ignore[arg-type]
            'preserve_concepts': ';'.join(canonical_entity(task, e) for e in _OTHERS[task]),  # type: ignore[arg-type]
            'expand_prompts': False,
            'device': 'cuda',
            'save_entire_model': False,
        }

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_the_guide_concept_is_never_the_task_name(self, task: str, entity: str) -> None:
        """The defect: the edit was aimed at the embedding of the word `breeds`."""
        config = _config(task, 'uce', entity, num_train_epochs=0)
        assert config['guide_concepts'] != task
        assert config['guide_concepts'] == substitute_concept(task)  # type: ignore[arg-type]

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_the_guide_concept_can_be_set_explicitly(self, task: str, entity: str) -> None:
        """A control session sets the guide concept and changes nothing else.

        The old configuration aimed the edit at the literal task name. Running it again beside the
        current one, on the same entity and baseline, is what measures the guide change; the
        override must reach `guide_concepts` and leave every other field exactly as the default.
        """
        default = _config(task, 'uce', entity, num_train_epochs=0)
        control = _config(task, 'uce', entity, num_train_epochs=0, uce_guide_concepts=task)

        assert control['guide_concepts'] == task
        assert {k: v for k, v in control.items() if k != 'guide_concepts'} == \
            {k: v for k, v in default.items() if k != 'guide_concepts'}

    @pytest.mark.parametrize('method', ['salun', 'distil', 'munba'])
    @pytest.mark.parametrize('setting', ['uce_guide_concepts', 'retained_entities', 'uce_erase_scale_override'])
    def test_a_uce_setting_is_refused_for_other_methods(self, method: str, setting: str) -> None:
        """A UCE-only setting passed to another method is an error, not a silent no-op."""
        value: Any = {'uce_guide_concepts': 'breeds', 'retained_entities': ['x'], 'uce_erase_scale_override': 3.0}[setting]
        with pytest.raises(ValueError, match=setting):
            _config('breeds', method, 'bouvier des flandres dog', **{setting: value})

    def test_the_preserve_set_is_every_other_entity_of_the_task(self) -> None:
        """Fails on the code before, which preserved the one word ``people``. Ninety-nine entities
        give the authors' per-concept weight floor of 0.1."""
        others = [f'Person_{k}' for k in range(99)]
        config = _config('people', 'uce', 'Mark_Philippoussis', num_train_epochs=0, retained_entities=others)
        preserved = config['preserve_concepts'].split(';')
        assert len(preserved) == 99 and 'Mark Philippoussis' not in preserved
        assert preserved[0] == 'Person 0'
        assert config['preserve_scale'] == 0.1

    def test_the_target_is_never_preserved(self) -> None:
        with pytest.raises(ValueError, match='among the entities'):
            _config('people', 'uce', 'Mark_Philippoussis', retained_entities=['Mark_Philippoussis', 'X'])

    def test_uce_needs_its_preserve_set(self) -> None:
        with pytest.raises(ValueError, match='retained_entities'):
            session_config.session_hyperparameters(
                'people', 'uce', 'Mark_Philippoussis', output_dir='o', dataset_forget_name='f',
                dataset_retain_name='r', model_base_name=_MODEL, device='cpu', num_train_epochs=0,
            )

    def test_the_erase_scale_is_the_dial(self) -> None:
        assert _config('scenes', 'uce', 'football_field', uce_erase_scale_override=23.4375)['erase_scale'] == 23.4375

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_the_three_methods_aim_at_the_same_substitute(self, task: str, entity: str) -> None:
        uce = _config(task, 'uce', entity, num_train_epochs=0)
        salun = _config(task, 'salun', entity)
        distil = _config(task, 'distil', entity)
        assert uce['guide_concepts'] == salun['overwriting_concept'] == distil['overwritting_concept']


class TestSalun:
    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_every_field(self, task: str, entity: str) -> None:
        config = _config(task, 'salun', entity)
        forget, retain = evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert config == {
            'output_dir': 'assets/models/out',
            'hub_model_id': None,
            'final_eval_prompts_forget': forget,
            'final_eval_prompts_retain': retain,
            'pretrained_model_name_or_path': _MODEL,
            'dataset_forget_name': 'assets/datasets/x/train_forget',
            'dataset_retain_name': 'assets/datasets/x/train_retain',
            'overwriting_concept': substitute_concept(task),  # type: ignore[arg-type]
            'train_method': 'xattn',
            'learning_rate': 1e-5,
            'mask_threshold': 0.5,
            'alpha': 0.5,
            'mask_guidance': 7.5,
            'num_train_epochs': 100,
            'resolution': 512,
            'random_flip': True,
            'dataloader_num_workers': 0,
            'per_device_train_batch_size': 1,
            'allow_tf32': False,
            'seed': 42,
            'device': 'cuda',
            'evaluation_noise_seed': 42,
        }


class TestTheAdapterTrainers:
    @staticmethod
    def _expected_common(task: str, entity: str) -> Dict[str, Any]:
        forget, retain = evaluation_prompts(task, entity)  # type: ignore[arg-type]
        return {
            'output_dir': 'assets/models/out',
            'hub_model_id': None,
            'final_eval_prompts_forget': forget,
            'final_eval_prompts_retain': retain,
            'model_name_or_path': _MODEL,
            'dataset_forget_name': 'assets/datasets/x/train_forget',
            'dataset_retain_name': 'assets/datasets/x/train_retain',
            'validation_prompt': forget[0],
            'dataloader_num_workers': 2,
            'resolution': 512,
            'num_validation_images': 1,
            'mixed_precision': 'no',
            'learning_rate': 6e-4,
            'max_grad_norm': 5.0,
            'num_train_epochs': 100,
            'validation_epochs': 101,
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
            'evaluation_noise_seed': 42,
        }

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_distil_every_field(self, task: str, entity: str) -> None:
        expected = self._expected_common(task, entity)
        expected['overwritting_concept'] = substitute_concept(task)  # type: ignore[arg-type]
        expected['forget_concept'] = canonical_entity(task, entity)  # type: ignore[arg-type]
        expected.update({
            'learning_rate': 1e-4, 'lora_r': 4, 'lora_alpha': 8, 'lora_dropout': 0.2, 'max_grad_norm': 1.0,
            'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-8, 'adam_weight_decay': 1e-2,
            'per_device_train_batch_size': 4, 'gradient_accumulation_steps': 1,
        })
        assert _config(task, 'distil', entity) == expected

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_munba_every_field(self, task: str, entity: str) -> None:
        expected = self._expected_common(task, entity)
        if task == 'scenes':
            expected.update({'learning_rate': 1.5e-4, 'max_grad_norm': 1.0, 'lora_r': 4})
        assert _config(task, 'munba', entity) == expected

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_the_forget_side_is_conditioned_on_the_phrase_it_is_evaluated_with(
        self, task: str, entity: str,
    ) -> None:
        config = _config(task, 'distil', entity)
        assert f"An image of {config['forget_concept']}" == config['final_eval_prompts_forget'][0]
        assert config['validation_prompt'] == config['final_eval_prompts_forget'][0]

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_no_per_machine_setting_is_decided_here(self, task: str, entity: str) -> None:
        """Batch size and gradient accumulation describe the machine, not the session."""
        config = _config(task, 'munba', entity)
        assert 'per_device_train_batch_size' not in config
        assert 'gradient_accumulation_steps' not in config


class TestTheMethodAxis:
    def test_an_unknown_method_is_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            _config('people', 'esd', 'Mark_Philippoussis')

    @pytest.mark.parametrize('method', ['uce', 'salun', 'distil', 'munba'])
    def test_every_method_carries_both_prompt_lists(self, method: str) -> None:
        config = _config('people', method, 'Mark_Philippoussis')
        assert len(config['final_eval_prompts_forget']) == 4
        assert len(config['final_eval_prompts_retain']) == 4


class TestTheDataPrecondition:
    @staticmethod
    def _split(folder: str, caption: str) -> str:
        from PIL import Image

        os.makedirs(folder, exist_ok=True)
        Image.new('RGB', (4, 4)).save(os.path.join(folder, 'abbey_0001.jpg'))
        with open(os.path.join(folder, 'metadata.jsonl'), 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({'file_name': 'abbey_0001.jpg', 'text': caption}) + '\n')
        return folder

    def test_a_prompted_split_passes(self, tmp_path: Any) -> None:
        forget = self._split(str(tmp_path / 'f'), 'An image of an abbey scene')
        retain = self._split(str(tmp_path / 'r'), 'An image of an abbey scene')

        session_config.assert_split_captions_ready('scenes', 'salun', forget, retain)

    def test_a_bare_caption_stops_the_session_before_anything_is_built(self, tmp_path: Any) -> None:
        forget = self._split(str(tmp_path / 'f'), 'abbey')
        retain = self._split(str(tmp_path / 'r'), 'An image of an abbey scene')

        with pytest.raises(ValueError) as caught:
            session_config.assert_split_captions_ready('scenes', 'distil', forget, retain)

        assert 'abbey_0001.jpg' in str(caught.value)

    def test_a_method_that_reads_no_training_data_is_not_checked(self, tmp_path: Any) -> None:
        """UCE is a closed-form edit: there is no split to be wrong."""
        missing = str(tmp_path / 'does-not-exist')

        session_config.assert_split_captions_ready('scenes', 'uce', missing, missing)

    def test_the_methods_that_read_training_data_are_named(self) -> None:
        trained: List[str] = sorted(session_config.METHODS_WITH_TRAINING_DATA)
        assert trained == ['distil', 'munba', 'salun']



class TestTheReferenceConfigurations:
    """Each method's settings against its published source, value by value (the dial excepted)."""

    UCE = {'lamb': 0.5, 'preserve_scale': 0.1, 'guide_concepts': 'the substitute concept'}
    SPARE = {'learning_rate': 1e-4, 'lr_scheduler_type': 'constant', 'lr_warmup_steps': 0, 'lora_r': 4,
             'lora_alpha': 8, 'lora_dropout': 0.2, 'per_device_train_batch_size': 4, 'adam_beta1': 0.9,
             'adam_beta2': 0.999, 'adam_epsilon': 1e-8, 'adam_weight_decay': 1e-2, 'random_flip': True,
             'max_grad_norm': 1.0, 'target_modules': ['to_k', 'to_q', 'to_v', 'to_out.0'],
             'mixed_precision': 'no'}
    SALUN = {'learning_rate': 1e-5, 'mask_threshold': 0.5, 'alpha': 0.5, 'mask_guidance': 7.5,
             'train_method': 'xattn', 'per_device_train_batch_size': 1, 'allow_tf32': False}

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_uce(self, task: str, entity: str) -> None:
        others = [f'other {k}' for k in range(99)]
        config = _config(task, 'uce', entity, num_train_epochs=0, retained_entities=others)
        assert config['lamb'] == self.UCE['lamb'] and config['preserve_scale'] == self.UCE['preserve_scale']
        assert config['guide_concepts'] == substitute_concept(task)  # type: ignore[arg-type]

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_spare(self, task: str, entity: str) -> None:
        config = _config(task, 'distil', entity)
        assert {key: config[key] for key in self.SPARE} == self.SPARE

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_salun(self, task: str, entity: str) -> None:
        config = _config(task, 'salun', entity)
        assert {key: config[key] for key in self.SALUN} == self.SALUN

    @pytest.mark.parametrize('method', ['uce', 'salun', 'distil'])
    def test_every_evaluation_is_seeded(self, method: str) -> None:
        assert _config('people', method, 'Mark_Philippoussis')['evaluation_noise_seed'] == 42

    def test_both_trained_methods_train_at_full_precision(self) -> None:
        '''Fails on the code before: SPARE trained in bfloat16 mixed precision and SalUn with
        TensorFloat-32 matrix products, and reduced precision moved SPARE's results beyond the spread
        between two ordinary trainings.'''
        spare = _config('people', 'distil', 'Mark_Philippoussis')
        salun = _config('people', 'salun', 'Mark_Philippoussis')
        assert spare['mixed_precision'] == 'no' and 'allow_tf32' not in spare
        assert salun['allow_tf32'] is False and 'mixed_precision' not in salun

    @pytest.mark.parametrize('method', ['salun', 'distil'])
    def test_every_trained_method_uses_the_one_training_seed(self, method: str) -> None:
        assert _config('people', method, 'Mark_Philippoussis')['seed'] == session_config.TRAINING_SEED

    def test_the_internal_record_path_is_gone_from_the_uce_docstring(self) -> None:
        assert 'tickets/' not in (session_config._uce.__doc__ or '')


@pytest.mark.parametrize('method, field', [('distil', 'save_lora_at_epochs'), ('salun', 'save_at_epochs')])
def test_a_trained_method_saves_at_the_requested_epochs(method: str, field: str) -> None:
    """Fails on the code before: no configuration carried the epochs to save at, so equalization's
    one run to the top of a ladder could not yield its lower values."""
    assert _config('people', method, 'George_W_Bush', save_at_epochs=[400, 50, 100, 50])[field] == [50, 100, 400]


def test_uce_has_no_epochs_to_save_at() -> None:
    with pytest.raises(ValueError, match='no epochs'):
        _config('people', 'uce', 'Mark_Philippoussis', save_at_epochs=[1])
