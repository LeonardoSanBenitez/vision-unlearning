"""Every field of an unlearning session's configuration, asserted directly.

The expected dictionaries below were read field by field out of ``pipeline_03_unlearn_model.py`` as
it stood before this configuration was extracted out of it, so this file is also the equivalence
proof for the move: anything that differs from the old behaviour differs on purpose, and there are
exactly three such differences, each marked where it appears.

1. UCE's ``guide_concepts`` is the task's substitute concept instead of the literal task name, so
   the three methods push the entity towards the same thing.
2. UCE sets no preserve concepts. The preserve term used to be the literal task name.
3. Both prompt lists come from the one prompt builder, so the retain side is the substitute concept
   for all three tasks.

The per-machine settings -- batch size and gradient accumulation, chosen from free video memory --
are deliberately not here; they stay in the pipeline, because they describe where a run happens
rather than what it is.
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
    settings.update(overrides)
    return session_config.session_hyperparameters(task, method, entity, **settings)  # type: ignore[arg-type]


class TestUce:
    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_every_field(self, task: str, entity: str) -> None:
        config = _config(task, 'uce', entity, num_train_epochs=0)

        expected_scales = (
            {'erase_scale': 230, 'preserve_scale': 1.2, 'lamb': 0.1} if task == 'breeds'
            else {'erase_scale': 30, 'preserve_scale': 0.01, 'lamb': 0.01}
        )
        forget, retain = evaluation_prompts(task, entity)  # type: ignore[arg-type]
        assert config == {
            'output_dir': 'assets/models/out',
            'hub_model_id': None,
            'final_eval_prompts_forget': forget,   # change 3
            'final_eval_prompts_retain': retain,   # change 3
            'pretrained_model_name_or_path': _MODEL,
            **expected_scales,
            'edit_concepts': canonical_entity(task, entity),  # type: ignore[arg-type]
            'guide_concepts': substitute_concept(task),       # change 1
            'preserve_concepts': None,                        # change 2
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
    def test_no_preserve_concept_is_set(self, task: str, entity: str) -> None:
        config = _config(task, 'uce', entity, num_train_epochs=0)
        assert config['preserve_concepts'] is None

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
            'num_train_epochs': 100,
            'resolution': 512,
            'random_flip': True,
            'dataloader_num_workers': 0,
            'per_device_train_batch_size': 1,
            'device': 'cuda',
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
        }

    @pytest.mark.parametrize('task, entity', _TASKS_AND_ENTITIES)
    def test_distil_every_field(self, task: str, entity: str) -> None:
        expected = self._expected_common(task, entity)
        expected['overwritting_concept'] = substitute_concept(task)  # type: ignore[arg-type]
        expected['forget_concept'] = canonical_entity(task, entity)  # type: ignore[arg-type]
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
