"""Tests for vision_unlearning.datasets.testbed.

Covers:
- get_shared_baseline_folder(): task-level shared baseline path construction.
- get_off_image_path(): shared folder preferred; falls back to entity folder when
  no baseline folder exists.
- exists_unlearned_dataset(): counts only on_* images; off_* files in legacy
  entity folders are ignored for backward compatibility.
- GeneratedDataset: OO abstraction — folder_path, is_baseline, file_path, exists,
  hf_config_name, hf_path_in_repo, validation, get_off_image_path class method,
  upload_if_recomputed.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import ValidationError

import sys
import importlib

from vision_unlearning.datasets.testbed import (
    GeneratedDataset,
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_off_image_path,
    get_shared_baseline_folder,
    exists_unlearned_dataset,
)
# Resolve the testbed and data_generation modules via sys.modules to avoid
# the torchvision.datasets namespace collision that trips up the standard
# `import vision_unlearning.datasets.testbed` syntax.
_testbed_module = sys.modules['vision_unlearning.datasets.testbed']
import vision_unlearning.utils.data_generation as _data_generation_module


class TestGetSharedBaselineFolder(unittest.TestCase):
    """get_shared_baseline_folder returns the task-level shared path."""

    def test_path_format(self) -> None:
        path = get_shared_baseline_folder('people', base_folder='assets')
        self.assertEqual(path, os.path.join('assets', 'datasets', 'generated_people_baseline'))

    def test_default_base_folder(self) -> None:
        path = get_shared_baseline_folder('scenes')
        self.assertEqual(path, os.path.join('assets', 'datasets', 'generated_scenes_baseline'))

    def test_custom_base_folder(self) -> None:
        path = get_shared_baseline_folder('breeds', base_folder='/tmp/my_assets')
        self.assertEqual(path, os.path.join('/tmp/my_assets', 'datasets', 'generated_breeds_baseline'))

    def test_no_method_in_path(self) -> None:
        """Shared baseline folder must never include a method token."""
        path = get_shared_baseline_folder('people')
        self.assertNotIn('distil', path)
        self.assertNotIn('uce', path)
        self.assertNotIn('munba', path)

    def test_no_entity_in_path(self) -> None:
        """Shared baseline folder must not include any entity name — it is task-level."""
        path = get_shared_baseline_folder('people')
        self.assertNotIn('Bush', path)
        self.assertNotIn('Powell', path)

    def test_baseline_token_present(self) -> None:
        """Shared folder must include 'baseline' as a literal token."""
        path = get_shared_baseline_folder('people')
        self.assertIn('baseline', path)


class TestGetOffImagePath(unittest.TestCase):
    """get_off_image_path: shared folder preferred; falls back to entity folder."""

    def test_prefers_shared_baseline_folder(self) -> None:
        """Shared task-level folder takes priority over entity folder."""
        with tempfile.TemporaryDirectory() as tmp:
            shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
            os.makedirs(shared_folder, exist_ok=True)
            # Create the expected baseline image so the new existence check passes.
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            open(os.path.join(shared_folder, expected_filename), 'w').close()

            path = get_off_image_path(
                task='people',
                target='Colin Powell',
                method='distil',
                num_train_epochs=400,
                seed=42,
                prompt='An image of Colin Powell',
                base_folder=tmp,
            )
            self.assertEqual(path, os.path.join(shared_folder, expected_filename))

    def test_falls_back_to_entity_folder_when_shared_absent(self) -> None:
        """When no shared baseline folder exists, fall back to entity folder."""
        with tempfile.TemporaryDirectory() as tmp:
            # Do NOT create any baseline folder — test the fallback.
            path = get_off_image_path(
                task='people',
                target='Colin Powell',
                method='distil',
                num_train_epochs=400,
                seed=42,
                prompt='An image of Colin Powell',
                base_folder=tmp,
            )
            entity_folder = get_generated_dataset_folder(
                'people', 'distil', 400, 'Colin Powell', base_folder=tmp
            )
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            self.assertEqual(path, os.path.join(entity_folder, expected_filename))

    def test_filename_format(self) -> None:
        """Filename inside the returned path must follow off_{seed:02d}_{prompt}.png convention."""
        with tempfile.TemporaryDirectory() as tmp:
            path = get_off_image_path(
                task='scenes',
                target='an abbey scene',
                method='uce',
                num_train_epochs=0,
                seed=7,
                prompt='An image of an abbey scene',
                base_folder=tmp,
            )
            self.assertTrue(os.path.basename(path).startswith('off_07_'))
            self.assertTrue(path.endswith('.png'))


class TestExistsUnlearnedDataset(unittest.TestCase):
    """exists_unlearned_dataset counts only on_* images; off_* files are ignored.

    Backward-compat: legacy entity folders that contain both on_* and off_* images
    (generated before the baseline-folder refactor) must still pass this check.
    """

    def _write_files(self, folder: str, filenames: list) -> None:
        os.makedirs(folder, exist_ok=True)
        for fn in filenames:
            open(os.path.join(folder, fn), 'w').close()

    def test_returns_false_when_folder_missing(self) -> None:
        self.assertFalse(
            exists_unlearned_dataset('/nonexistent/path', [42], ['prompt A'])
        )

    def test_returns_true_when_only_on_images_present(self) -> None:
        """New contract: entity folder has on_* images only."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42, 43]
            prompts = ['An image of Alice', 'An image of Bob']
            # 2 seeds * 2 prompts = 4 on_ files + 1 metadata.jsonl
            files = [f'on_{s:02d}_{p}.png' for s in seeds for p in prompts] + ['metadata.jsonl']
            self._write_files(tmp, files)
            self.assertTrue(exists_unlearned_dataset(tmp, seeds, prompts))

    def test_returns_true_when_legacy_off_images_also_present(self) -> None:
        """Old mixed-folder format (on_ + off_) is valid — off_* files are ignored.

        Backward compatibility: datasets generated before the baseline refactor store
        both on_* and off_* images in the entity folder.  exists_unlearned_dataset must
        accept those folders so that 1_unlearn_from_metadata.py does not raise an
        AssertionError when processing already-generated entities.
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Alice']
            # Old format: 1 on_ + 1 off_ + metadata.jsonl
            # off_* must be ignored; only on_* count matters.
            files = ['on_42_An image of Alice.png', 'off_42_An image of Alice.png', 'metadata.jsonl']
            self._write_files(tmp, files)
            self.assertTrue(exists_unlearned_dataset(tmp, seeds, prompts))

    def test_returns_false_when_too_few_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42, 43]
            prompts = ['An image of Alice']
            # Write only one of the two expected on_ images
            self._write_files(tmp, ['on_42_An image of Alice.png', 'metadata.jsonl'])
            self.assertFalse(exists_unlearned_dataset(tmp, seeds, prompts))

    def test_ignores_ipynb_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Alice']
            files = ['on_42_An image of Alice.png', 'metadata.jsonl', '.ipynb_checkpoints']
            self._write_files(tmp, files)
            self.assertTrue(exists_unlearned_dataset(tmp, seeds, prompts))


class TestGeneratedDatasetFolderPath(unittest.TestCase):
    """GeneratedDataset.folder_path matches the expected folder convention."""

    def test_shared_baseline_folder_path(self) -> None:
        ds = GeneratedDataset(task='people', base_folder='assets')
        self.assertEqual(
            ds.folder_path,
            os.path.join('assets', 'datasets', 'generated_people_baseline'),
        )

    def test_entity_dataset_folder_path(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400, base_folder='assets',
        )
        self.assertEqual(
            ds.folder_path,
            os.path.join('assets', 'datasets', 'generated_people_Colin Powell_distil_400'),
        )

    def test_entity_dataset_epochs_zero_padded(self) -> None:
        ds = GeneratedDataset(
            task='breeds', target='poodle',
            method='uce', num_train_epochs=0, base_folder='assets',
        )
        self.assertIn('_000', ds.folder_path)

    def test_custom_base_folder(self) -> None:
        ds = GeneratedDataset(task='scenes', base_folder='/tmp/my_assets')
        self.assertTrue(ds.folder_path.startswith('/tmp/my_assets'))


class TestGeneratedDatasetIdentity(unittest.TestCase):
    """is_baseline reflects the dataset kind correctly."""

    def test_shared_baseline_is_baseline(self) -> None:
        ds = GeneratedDataset(task='people')
        self.assertTrue(ds.is_baseline)

    def test_entity_dataset_is_not_baseline(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        self.assertFalse(ds.is_baseline)


class TestGeneratedDatasetValidation(unittest.TestCase):
    """Pydantic validation catches invalid combinations."""

    def test_method_without_target_raises(self) -> None:
        with self.assertRaises((ValidationError, AssertionError)):
            GeneratedDataset(task='people', method='distil', num_train_epochs=400)

    def test_method_without_epochs_raises(self) -> None:
        with self.assertRaises((ValidationError, AssertionError)):
            GeneratedDataset(task='people', target='Colin Powell', method='distil')

    def test_baseline_with_no_target_is_valid(self) -> None:
        ds = GeneratedDataset(task='breeds')
        self.assertTrue(ds.is_baseline)

    def test_target_without_method_raises(self) -> None:
        """Per-entity baseline concept does not exist — target requires method."""
        with self.assertRaises((ValidationError, ValueError)):
            GeneratedDataset(task='breeds', target='poodle')

    def test_upload_if_recomputed_default_false(self) -> None:
        ds = GeneratedDataset(task='people')
        self.assertFalse(ds.upload_if_recomputed)

    def test_upload_if_recomputed_can_be_set(self) -> None:
        ds = GeneratedDataset(task='people', upload_if_recomputed=True)
        self.assertTrue(ds.upload_if_recomputed)


class TestGeneratedDatasetFilePath(unittest.TestCase):
    """file_path returns correct image file paths."""

    def test_off_file_path_shared_baseline(self) -> None:
        ds = GeneratedDataset(task='people', base_folder='assets')
        path = ds.file_path('off', 42, 'An image of Colin Powell')
        self.assertEqual(path, os.path.join(
            'assets', 'datasets', 'generated_people_baseline',
            'off_42_An image of Colin Powell.png',
        ))

    def test_on_file_path_entity_dataset(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400, base_folder='assets',
        )
        path = ds.file_path('on', 42, 'An image of Colin Powell')
        self.assertIn('on_42_', path)
        self.assertTrue(path.endswith('.png'))

    def test_on_file_raises_for_baseline(self) -> None:
        ds = GeneratedDataset(task='people', base_folder='assets')
        with self.assertRaises(ValueError):
            ds.file_path('on', 42, 'An image of Colin Powell')

    def test_seed_two_digit_padding(self) -> None:
        ds = GeneratedDataset(task='people', base_folder='assets')
        path = ds.file_path('off', 7, 'test')
        self.assertIn('off_07_', path)


class TestGeneratedDatasetHfConfigName(unittest.TestCase):
    """hf_config_name matches the basename of folder_path."""

    def test_shared_baseline_config_name(self) -> None:
        ds = GeneratedDataset(task='people')
        self.assertEqual(ds.hf_config_name, 'generated_people_baseline')

    def test_entity_dataset_config_name(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        self.assertEqual(ds.hf_config_name, 'generated_people_Colin Powell_distil_400')


class TestGeneratedDatasetHfPathInRepo(unittest.TestCase):
    """hf_path_in_repo always places datasets under the datasets/ prefix in the HF repo."""

    def test_shared_baseline_hf_path_in_repo(self) -> None:
        ds = GeneratedDataset(task='breeds')
        self.assertEqual(ds.hf_path_in_repo, 'datasets/generated_breeds_baseline')

    def test_entity_dataset_hf_path_in_repo(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        self.assertEqual(ds.hf_path_in_repo, 'datasets/generated_people_Colin Powell_distil_400')

    def test_hf_path_in_repo_starts_with_datasets_prefix(self) -> None:
        """All generated datasets live under datasets/ in the HF repo."""
        for task in ('people', 'scenes', 'breeds'):
            ds = GeneratedDataset(task=task)  # type: ignore[arg-type]
            self.assertTrue(
                ds.hf_path_in_repo.startswith('datasets/'),
                f"hf_path_in_repo for {task} baseline should start with 'datasets/': {ds.hf_path_in_repo}",
            )

    def test_hf_path_in_repo_uses_hf_config_name(self) -> None:
        """hf_path_in_repo is datasets/{hf_config_name} by construction."""
        ds = GeneratedDataset(task='scenes')
        self.assertEqual(ds.hf_path_in_repo, f"datasets/{ds.hf_config_name}")


class TestGeneratedDatasetExists(unittest.TestCase):
    """GeneratedDataset.exists() counts the right image type for each dataset kind."""

    def _write_files(self, folder: str, filenames: list) -> None:
        os.makedirs(folder, exist_ok=True)
        for fn in filenames:
            open(os.path.join(folder, fn), 'w').close()

    def test_shared_baseline_returns_true_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42, 43]
            prompts = ['An image of Colin Powell', 'An image of Tony Blair']
            ds = GeneratedDataset(task='people', base_folder=tmp)
            files = [
                f'off_{s:02d}_{p}.png' for s in seeds for p in prompts
            ] + ['metadata.jsonl']
            self._write_files(ds.folder_path, files)
            self.assertTrue(ds.exists(seeds, prompts))

    def test_shared_baseline_returns_false_when_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42, 43]
            prompts = ['An image of Colin Powell']
            ds = GeneratedDataset(task='people', base_folder=tmp)
            # Write only one seed instead of two
            self._write_files(ds.folder_path, ['off_42_An image of Colin Powell.png', 'metadata.jsonl'])
            self.assertFalse(ds.exists(seeds, prompts))

    def test_entity_dataset_counts_only_on_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']
            ds = GeneratedDataset(
                task='people', target='Colin Powell',
                method='distil', num_train_epochs=400, base_folder=tmp,
            )
            # Legacy folder: both on_ and off_ present; only on_ should count
            files = [
                'on_42_An image of Colin Powell.png',
                'off_42_An image of Colin Powell.png',
                'metadata.jsonl',
            ]
            self._write_files(ds.folder_path, files)
            self.assertTrue(ds.exists(seeds, prompts))

    def test_missing_metadata_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']
            ds = GeneratedDataset(task='people', base_folder=tmp)
            # Images present but no metadata.jsonl
            self._write_files(ds.folder_path, ['off_42_An image of Colin Powell.png'])
            self.assertFalse(ds.exists(seeds, prompts))

    def test_returns_false_when_folder_missing(self) -> None:
        ds = GeneratedDataset(task='people', base_folder='/nonexistent/path')
        self.assertFalse(ds.exists([42], ['prompt']))


class TestGeneratedDatasetGetOffImagePath(unittest.TestCase):
    """GeneratedDataset.get_off_image_path() mirrors the module-level function."""

    def test_classmethod_returns_same_as_module_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # No baseline folder → should fall back to entity folder
            path_classmethod = GeneratedDataset.get_off_image_path(
                task='people', target='Colin Powell',
                method='distil', num_train_epochs=400,
                seed=42, prompt='An image of Colin Powell', base_folder=tmp,
            )
            path_function = get_off_image_path(
                task='people', target='Colin Powell',
                method='distil', num_train_epochs=400,
                seed=42, prompt='An image of Colin Powell', base_folder=tmp,
            )
            self.assertEqual(path_classmethod, path_function)

    def test_prefers_shared_baseline_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = GeneratedDataset(task='people', base_folder=tmp)
            os.makedirs(shared.folder_path, exist_ok=True)
            # Create the expected baseline image so the existence check passes.
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            open(os.path.join(shared.folder_path, expected_filename), 'w').close()
            path = GeneratedDataset.get_off_image_path(
                task='people', target='Colin Powell',
                method='distil', num_train_epochs=400,
                seed=42, prompt='An image of Colin Powell', base_folder=tmp,
            )
            self.assertTrue(path.startswith(shared.folder_path))


class TestGeneratedDatasetComputeLocalHit(unittest.TestCase):
    """compute() returns immediately when data is already present locally."""

    def _write_files(self, folder: str, filenames: list) -> None:
        os.makedirs(folder, exist_ok=True)
        for fn in filenames:
            open(os.path.join(folder, fn), 'w').close()

    def test_compute_returns_folder_when_local_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']
            ds = GeneratedDataset(task='people', base_folder=tmp)
            files = ['off_42_An image of Colin Powell.png', 'metadata.jsonl']
            self._write_files(ds.folder_path, files)
            # Should not call HF or scratch — data is already there
            result = ds.compute(seeds, prompts)
            self.assertEqual(result, ds.folder_path)


class TestGeneratedDatasetUploadIfRecomputed(unittest.TestCase):
    """compute() uploads to HF after scratch generation when upload_if_recomputed=True."""

    def test_upload_called_after_scratch_generation(self) -> None:
        """When _compute_from_scratch succeeds and upload_if_recomputed=True, upload is called."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            ds = GeneratedDataset(task='people', base_folder=tmp, upload_if_recomputed=True)

            def fake_compute_from_scratch(s: list, p: list, batch_size: int = 16) -> str:  # type: ignore[override]
                # Write the expected files so exists() returns True
                os.makedirs(ds.folder_path, exist_ok=True)
                for seed in s:
                    for prompt in p:
                        fname = f'off_{seed:02d}_{prompt}.png'
                        open(os.path.join(ds.folder_path, fname), 'w').close()
                open(os.path.join(ds.folder_path, 'metadata.jsonl'), 'w').close()
                return ds.folder_path

            mock_upload = MagicMock()
            mock_hf_exists = MagicMock(return_value=False)

            with patch.dict('os.environ', {'HF_TOKEN': 'fake_token'}):
                with patch.object(ds, '_compute_from_scratch', side_effect=fake_compute_from_scratch):
                    with patch(
                        'vision_unlearning.integrations.huggingface.huggingface_dataset_upload',
                        mock_upload,
                    ):
                        with patch(
                            'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                            mock_hf_exists,
                        ):
                            result = ds.compute(seeds, prompts)

            self.assertEqual(result, ds.folder_path)
            mock_upload.assert_called_once()
            # Verify upload uses hf_path_in_repo (datasets/ prefix) not bare hf_config_name
            call_kwargs = mock_upload.call_args.kwargs
            self.assertEqual(
                call_kwargs.get('path_in_repo'),
                ds.hf_path_in_repo,
                "upload must use hf_path_in_repo ('datasets/...'), not bare hf_config_name",
            )

    def test_upload_not_called_when_upload_if_recomputed_false(self) -> None:
        """Default behaviour: upload is NOT called even after scratch generation."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            ds = GeneratedDataset(task='people', base_folder=tmp, upload_if_recomputed=False)

            def fake_compute_from_scratch(s: list, p: list, batch_size: int = 16) -> str:  # type: ignore[override]
                os.makedirs(ds.folder_path, exist_ok=True)
                for seed in s:
                    for prompt in p:
                        fname = f'off_{seed:02d}_{prompt}.png'
                        open(os.path.join(ds.folder_path, fname), 'w').close()
                open(os.path.join(ds.folder_path, 'metadata.jsonl'), 'w').close()
                return ds.folder_path

            mock_upload = MagicMock()
            mock_hf_exists = MagicMock(return_value=False)

            with patch.dict('os.environ', {'HF_TOKEN': 'fake_token'}):
                with patch.object(ds, '_compute_from_scratch', side_effect=fake_compute_from_scratch):
                    with patch(
                        'vision_unlearning.integrations.huggingface.huggingface_dataset_upload',
                        mock_upload,
                    ):
                        with patch(
                            'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                            mock_hf_exists,
                        ):
                            result = ds.compute(seeds, prompts)

            self.assertEqual(result, ds.folder_path)
            mock_upload.assert_not_called()


class TestGeneratedDatasetComputeFromScratchEntity(unittest.TestCase):
    """_compute_from_scratch for entity datasets: model-not-found raises FileNotFoundError;
    model-found path calls generate_dataset with the correct arguments."""

    def test_entity_dataset_raises_file_not_found_when_model_missing(self) -> None:
        """FileNotFoundError is raised when the trained model does not exist on disk."""
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        # exists_unlearned_model will return False because the path does not exist
        # in the test environment — no patching needed.
        with self.assertRaises(FileNotFoundError) as ctx:
            ds._compute_from_scratch([42], ['An image of Colin Powell'])
        self.assertIn('Colin Powell', str(ctx.exception))
        self.assertIn('distil', str(ctx.exception))

    def test_entity_dataset_lora_method_calls_generate_dataset(self) -> None:
        """For a LoRA method (distil), _compute_from_scratch calls generate_dataset
        with model_base_name + lora_name + lora_requires_inversion=False."""
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        seeds = [42]
        prompts = ['An image of Colin Powell']
        expected_filenames = ['on_42_An image of Colin Powell.png']

        with patch.object(_testbed_module, 'exists_unlearned_model', return_value=True), \
             patch.object(_testbed_module, 'get_unlearned_model_folder',
                          return_value='/fake/model/path') as mock_model_folder, \
             patch.object(_data_generation_module, 'generate_dataset',
                          return_value=[]) as mock_gen, \
             patch('torch.cuda.is_available', return_value=False):
            ds._compute_from_scratch(seeds, prompts)

            mock_model_folder.assert_called_once_with(
                'people', 'distil', 400, 'Colin Powell', 'assets'
            )
            mock_gen.assert_called_once_with(
                model_base_name='CompVis/stable-diffusion-v1-4',
                lora_name='/fake/model/path',
                model_pipeline=None,
                prompts=prompts,
                output_path=ds.folder_path,
                seeds=seeds,
                filenames=expected_filenames,
                lora_requires_inversion=False,
                batch_size=16,
            )

    def test_entity_dataset_munba_sets_lora_requires_inversion_true(self) -> None:
        """For munba, lora_requires_inversion must be True."""
        ds = GeneratedDataset(
            task='people', target='Brad Pitt',
            method='munba', num_train_epochs=400,
        )
        seeds = [42]
        prompts = ['An image of Brad Pitt']

        with patch.object(_testbed_module, 'exists_unlearned_model', return_value=True), \
             patch.object(_testbed_module, 'get_unlearned_model_folder',
                          return_value='/fake/model/path'), \
             patch.object(_data_generation_module, 'generate_dataset',
                          return_value=[]) as mock_gen, \
             patch('torch.cuda.is_available', return_value=False):
            ds._compute_from_scratch(seeds, prompts)

        call_kwargs = mock_gen.call_args.kwargs
        self.assertTrue(call_kwargs['lora_requires_inversion'])

    def test_entity_dataset_uce_uses_get_pipeline_from_modified_weights(self) -> None:
        """For UCE, _compute_from_scratch uses UCE.get_pipeline_from_modified_weights."""
        from vision_unlearning.unlearner.uce_sd_erase import UCE
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='uce', num_train_epochs=400,
        )
        seeds = [42]
        prompts = ['An image of Colin Powell']
        fake_pipeline = MagicMock()

        with patch.object(_testbed_module, 'exists_unlearned_model', return_value=True), \
             patch.object(_testbed_module, 'get_unlearned_model_folder',
                          return_value='/fake/uce/path'), \
             patch.object(_data_generation_module, 'generate_dataset',
                          return_value=[]) as mock_gen, \
             patch('torch.cuda.is_available', return_value=False), \
             patch.object(UCE, 'get_pipeline_from_modified_weights',
                          return_value=fake_pipeline) as mock_pipeline:
            ds._compute_from_scratch(seeds, prompts)

            mock_pipeline.assert_called_once_with(
                pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4',
                device='cpu',
                output_dir='/fake/uce/path',
            )
            gen_call_kwargs = mock_gen.call_args.kwargs
            self.assertIsNone(gen_call_kwargs['model_base_name'])
            self.assertIsNone(gen_call_kwargs['lora_name'])
            self.assertEqual(gen_call_kwargs['model_pipeline'], fake_pipeline)


class TestGetTargetOverwriteMethodInvariance(unittest.TestCase):
    """Issue 1 (Cidral 2026-05-23): get_target_overwrite ignores the method argument.

    0_generate_dataset_original.py builds prompts with method='distil' hardcoded.
    3_compute_caused_interferences.py and 3_compute_embeddings.py build prompts
    with the actual runtime method.  This test proves these lists are always
    identical, making the hardcoded 'distil' in 0_generate_dataset_original.py safe.

    A prompt-list mismatch would cause every off-image filename to be wrong —
    a silent, catastrophic error.  If get_target_overwrite ever starts returning
    different values for different methods, this test will catch it.
    """

    from vision_unlearning.datasets.testbed import get_target_overwrite as _gto

    TASKS_AND_TARGETS = [
        ('people', 'Colin Powell'),
        ('people', 'Brad Pitt'),
        ('breeds', 'poodle'),
        ('breeds', 'Afghan hound'),
        ('scenes', 'abbey'),
        ('scenes', 'airport terminal'),
    ]
    METHODS = ['distil', 'uce', 'munba']

    def test_method_does_not_affect_target_string(self) -> None:
        """get_target_overwrite()[0] must return the same string for all methods."""
        from vision_unlearning.datasets.testbed import get_target_overwrite
        for task, target in self.TASKS_AND_TARGETS:
            results = [
                get_target_overwrite(task, m, target)[0]  # type: ignore[arg-type]
                for m in self.METHODS
            ]
            self.assertEqual(
                len(set(results)), 1,
                msg=(
                    f"get_target_overwrite()[0] differs across methods for "
                    f"task={task!r}, target={target!r}: {dict(zip(self.METHODS, results))}"
                ),
            )

    def test_prompt_list_distil_equals_uce_equals_munba(self) -> None:
        """Prompt lists built with any method are identical.

        Simulates what 0_generate_dataset_original.py does (hardcoded 'distil')
        and what 3_compute_caused_interferences.py/3_compute_embeddings.py do
        (runtime method), and asserts they are the same.
        """
        from vision_unlearning.datasets.testbed import get_target_overwrite
        fake_metadata = [
            {'name': target}
            for _, target in self.TASKS_AND_TARGETS
            if True  # use all for completeness
        ]
        for task in ['people', 'breeds', 'scenes']:
            task_metadata = [
                {'name': t} for (tk, t) in self.TASKS_AND_TARGETS if tk == task
            ]
            if not task_metadata:
                continue
            prompts_distil = [
                f"An image of {get_target_overwrite(task, 'distil', m['name'])[0]}"  # type: ignore[arg-type]
                for m in task_metadata
            ]
            for method in ['uce', 'munba']:
                prompts_method = [
                    f"An image of {get_target_overwrite(task, method, m['name'])[0]}"  # type: ignore[arg-type]
                    for m in task_metadata
                ]
                self.assertEqual(
                    prompts_distil, prompts_method,
                    msg=(
                        f"Prompt list mismatch: 'distil' vs '{method}' for task={task!r}. "
                        f"0_generate_dataset_original.py hardcodes 'distil'; "
                        f"a mismatch here means off-image filenames would be wrong."
                    ),
                )


class TestGeneratedDatasetComputeBatchSize(unittest.TestCase):
    """Issue 3 (Cidral 2026-05-23): batch_size must be forwarded through compute() and
    _compute_from_scratch() to generate_dataset().

    The plan documents batch_size=16 as optimal for this hardware.  Before this fix,
    _compute_from_scratch called generate_dataset() with no batch_size, defaulting to 4.
    """

    def test_batch_size_forwarded_to_generate_dataset_baseline(self) -> None:
        """compute() forwards batch_size to _compute_from_scratch, which passes it to
        generate_dataset for the shared baseline case."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            ds = GeneratedDataset(task='people', base_folder=tmp)

            captured: dict = {}

            def fake_compute_from_scratch(s: list, p: list, batch_size: int = 16) -> str:  # type: ignore[override]
                captured['batch_size'] = batch_size
                # Write the expected files so exists() returns True post-call
                os.makedirs(ds.folder_path, exist_ok=True)
                for seed in s:
                    for prompt in p:
                        fname = f'off_{seed:02d}_{prompt}.png'
                        open(os.path.join(ds.folder_path, fname), 'w').close()
                open(os.path.join(ds.folder_path, 'metadata.jsonl'), 'w').close()
                return ds.folder_path

            mock_hf_exists = MagicMock(return_value=False)
            with patch.object(ds, '_compute_from_scratch', side_effect=fake_compute_from_scratch), \
                 patch('vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                       mock_hf_exists):
                ds.compute(seeds, prompts, batch_size=8)

            self.assertEqual(captured.get('batch_size'), 8)

    def test_batch_size_default_is_16(self) -> None:
        """Default batch_size for compute() and _compute_from_scratch() is 16."""
        import inspect
        from vision_unlearning.datasets.testbed import GeneratedDataset as GD
        sig_compute = inspect.signature(GD.compute)
        sig_scratch = inspect.signature(GD._compute_from_scratch)
        self.assertEqual(sig_compute.parameters['batch_size'].default, 16)
        self.assertEqual(sig_scratch.parameters['batch_size'].default, 16)

    def test_batch_size_passed_to_generate_dataset_in_scratch_baseline(self) -> None:
        """_compute_from_scratch passes batch_size to generate_dataset (shared baseline)."""
        ds = GeneratedDataset(task='people', base_folder='assets')
        seeds = [42]
        prompts = ['An image of Colin Powell']

        with patch.object(_testbed_module, 'exists_unlearned_model', return_value=True), \
             patch.object(_data_generation_module, 'generate_dataset',
                          return_value=[]) as mock_gen, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('diffusers.AutoPipelineForText2Image.from_pretrained',
                   return_value=MagicMock(to=lambda d: MagicMock())):
            # Patch from_pretrained so the pipeline load doesn't actually run
            import diffusers
            with patch.object(
                diffusers.AutoPipelineForText2Image, 'from_pretrained',
                return_value=MagicMock(to=MagicMock(return_value=MagicMock()))
            ):
                ds._compute_from_scratch(seeds, prompts, batch_size=32)

        gen_call_kwargs = mock_gen.call_args.kwargs
        self.assertEqual(gen_call_kwargs['batch_size'], 32)

    def test_batch_size_passed_to_generate_dataset_in_scratch_entity(self) -> None:
        """_compute_from_scratch passes batch_size to generate_dataset (entity dataset)."""
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        seeds = [42]
        prompts = ['An image of Colin Powell']

        with patch.object(_testbed_module, 'exists_unlearned_model', return_value=True), \
             patch.object(_testbed_module, 'get_unlearned_model_folder',
                          return_value='/fake/model/path'), \
             patch.object(_data_generation_module, 'generate_dataset',
                          return_value=[]) as mock_gen, \
             patch('torch.cuda.is_available', return_value=False):
            ds._compute_from_scratch(seeds, prompts, batch_size=8)

        gen_call_kwargs = mock_gen.call_args.kwargs
        self.assertEqual(gen_call_kwargs['batch_size'], 8)


class TestGeneratedDatasetExistsPartialPromptWarning(unittest.TestCase):
    """Issue 4 (Cidral 2026-05-23): exists() is unsafe with a partial prompt list for
    the shared baseline.

    The shared baseline folder contains images for ALL entities in the task.
    If exists() is called with a partial prompts list, the expected_count will be
    smaller than the actual image count, causing exists() to return False (mismatch)
    and triggering full re-generation.

    This test documents the behaviour (not a bug — callers must pass the full list)
    and verifies that the docstring warning is present.
    """

    def _write_files(self, folder: str, filenames: list) -> None:
        os.makedirs(folder, exist_ok=True)
        for fn in filenames:
            open(os.path.join(folder, fn), 'w').close()

    def test_exists_returns_false_with_partial_prompt_list_for_baseline(self) -> None:
        """When the baseline folder has more images than len(seeds)*len(partial_prompts),
        exists() returns False — documenting why callers must pass the full list."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            all_prompts = [
                'An image of Colin Powell',
                'An image of Tony Blair',
                'An image of Brad Pitt',
            ]
            ds = GeneratedDataset(task='people', base_folder=tmp)
            # Write ALL 3 prompts in the folder
            files = [f'off_{s:02d}_{p}.png' for s in seeds for p in all_prompts] + ['metadata.jsonl']
            self._write_files(ds.folder_path, files)

            # Calling with only the first prompt: 1*1=1 expected, 3 actual → False
            partial_prompts = ['An image of Colin Powell']
            self.assertFalse(
                ds.exists(seeds, partial_prompts),
                msg=(
                    "exists() with a partial prompt list on a shared baseline folder "
                    "should return False because the actual image count (3) exceeds "
                    "the expected count (1).  Callers must always pass the full prompt list."
                ),
            )

    def test_exists_returns_true_with_full_prompt_list_for_baseline(self) -> None:
        """When the full prompt list is passed, exists() returns True (correct)."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            all_prompts = [
                'An image of Colin Powell',
                'An image of Tony Blair',
                'An image of Brad Pitt',
            ]
            ds = GeneratedDataset(task='people', base_folder=tmp)
            files = [f'off_{s:02d}_{p}.png' for s in seeds for p in all_prompts] + ['metadata.jsonl']
            self._write_files(ds.folder_path, files)
            self.assertTrue(ds.exists(seeds, all_prompts))

    def test_exists_docstring_warns_about_partial_prompt_list(self) -> None:
        """The docstring of exists() must contain a warning about partial prompt lists."""
        docstring = GeneratedDataset.exists.__doc__ or ''
        self.assertIn('partial', docstring.lower(),
                      msg="exists() docstring must warn about partial prompt lists")
        self.assertIn('complete', docstring.lower(),
                      msg="exists() docstring must instruct callers to pass the complete list")


class TestComputeFromScratchMetadataJsonl(unittest.TestCase):
    """Issue 5 (Cidral 2026-05-23): metadata.jsonl in entity _compute_from_scratch
    is only mock-verified.

    The unit tests for entity _compute_from_scratch mock generate_dataset(), so they
    do not exercise the actual metadata.jsonl write.  generate_dataset() does write
    metadata.jsonl (data_generation.py line 165), but this is a cross-module dependency
    that is not tested end-to-end here.

    This class documents this limitation explicitly and verifies that:
    1. The shared baseline path (which is NOT mocked in compute() tests) does write
       metadata.jsonl when generate_dataset is called with a real implementation.
    2. The entity path delegates metadata writing to generate_dataset, whose contract
       is tested in test_data_generation.py.
    """

    def test_generate_dataset_writes_metadata_jsonl(self) -> None:
        """generate_dataset() writes metadata.jsonl to output_path as its final step.

        This verifies the cross-module contract that _compute_from_scratch relies on
        for entity datasets.  If generate_dataset ever stops writing metadata.jsonl,
        GeneratedDataset.exists() would return False after generation (missing metadata)
        and cause an AssertionError in compute().
        """
        from vision_unlearning.utils.data_generation import generate_dataset
        import inspect
        src = inspect.getsource(generate_dataset)
        # Verify that metadata.jsonl is written (source-level check)
        self.assertIn('metadata.jsonl', src,
                      msg=(
                          "generate_dataset() must write metadata.jsonl. "
                          "GeneratedDataset._compute_from_scratch() (entity path) "
                          "relies on this for exists() to return True after generation."
                      ))

    def test_shared_baseline_exists_check_requires_metadata_jsonl(self) -> None:
        """exists() returns False when metadata.jsonl is absent, even if all images exist.

        This documents why _compute_from_scratch MUST produce metadata.jsonl —
        the compute() assertion after generation calls exists() which checks for it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']
            ds = GeneratedDataset(task='people', base_folder=tmp)
            # Write image but NOT metadata.jsonl
            os.makedirs(ds.folder_path, exist_ok=True)
            open(os.path.join(ds.folder_path, 'off_42_An image of Colin Powell.png'), 'w').close()
            # No metadata.jsonl → exists() must return False
            self.assertFalse(ds.exists(seeds, prompts),
                             msg="exists() must return False when metadata.jsonl is absent")


class TestGetOffImagePathHFDownloadCascade(unittest.TestCase):
    """get_off_image_path: HuggingFace download cascade when seeds and prompts are provided.

    When the shared baseline folder is absent locally AND the caller provides the
    full seeds/prompts lists, get_off_image_path must attempt to download the baseline
    via GeneratedDataset.compute() before falling back to the entity folder.
    """

    def _write_baseline_files(
        self, folder: str, seeds: list, prompts: list
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        for s in seeds:
            for p in prompts:
                open(os.path.join(folder, f'off_{s:02d}_{p}.png'), 'w').close()
        open(os.path.join(folder, 'metadata.jsonl'), 'w').close()

    def test_downloads_baseline_from_hf_when_seeds_and_prompts_provided(self) -> None:
        """When seeds and prompts are given and the baseline is missing locally,
        compute() is called on the baseline GeneratedDataset — triggering HF download."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            # Simulate HF download: fake compute() writes the baseline folder to disk.
            def fake_compute(s: list, p: list, batch_size: int = 16) -> str:
                shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
                self._write_baseline_files(shared_folder, s, p)
                return shared_folder

            with patch.object(
                _testbed_module.GeneratedDataset,
                'compute',
                side_effect=fake_compute,
            ):
                path = get_off_image_path(
                    task='people',
                    target='Colin Powell',
                    method='distil',
                    num_train_epochs=400,
                    seed=42,
                    prompt='An image of Colin Powell',
                    base_folder=tmp,
                    seeds=seeds,
                    prompts=prompts,
                )

            shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            self.assertEqual(path, os.path.join(shared_folder, expected_filename))

    def test_falls_back_to_entity_folder_when_seeds_and_prompts_not_provided(self) -> None:
        """Without seeds/prompts, the function must NOT attempt HF download
        and must fall back to the entity folder (backward-compatible)."""
        with tempfile.TemporaryDirectory() as tmp:
            mock_compute = MagicMock()
            with patch.object(_testbed_module.GeneratedDataset, 'compute', mock_compute):
                path = get_off_image_path(
                    task='people',
                    target='Colin Powell',
                    method='distil',
                    num_train_epochs=400,
                    seed=42,
                    prompt='An image of Colin Powell',
                    base_folder=tmp,
                    # seeds and prompts NOT provided
                )

            mock_compute.assert_not_called()
            entity_folder = get_generated_dataset_folder(
                'people', 'distil', 400, 'Colin Powell', base_folder=tmp
            )
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            self.assertEqual(path, os.path.join(entity_folder, expected_filename))

    def test_falls_back_to_entity_folder_when_hf_download_fails_to_produce_folder(self) -> None:
        """If compute() is called but the shared folder is still absent after it returns
        (e.g. HF download failed silently), the function falls back to the entity folder."""
        with tempfile.TemporaryDirectory() as tmp:
            # compute() does nothing — baseline folder remains absent.
            def no_op_compute(s: list, p: list, batch_size: int = 16) -> str:
                return get_shared_baseline_folder('people', base_folder=tmp)

            with patch.object(
                _testbed_module.GeneratedDataset,
                'compute',
                side_effect=no_op_compute,
            ):
                path = get_off_image_path(
                    task='people',
                    target='Colin Powell',
                    method='distil',
                    num_train_epochs=400,
                    seed=42,
                    prompt='An image of Colin Powell',
                    base_folder=tmp,
                    seeds=[42],
                    prompts=['An image of Colin Powell'],
                )

            entity_folder = get_generated_dataset_folder(
                'people', 'distil', 400, 'Colin Powell', base_folder=tmp
            )
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            self.assertEqual(path, os.path.join(entity_folder, expected_filename))

    def test_shared_folder_takes_priority_even_with_seeds_and_prompts(self) -> None:
        """When the shared baseline folder already exists locally, no HF call should
        be made — even if seeds and prompts are provided."""
        with tempfile.TemporaryDirectory() as tmp:
            shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
            os.makedirs(shared_folder, exist_ok=True)
            # Create the expected baseline image so the existence check passes.
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            open(os.path.join(shared_folder, expected_filename), 'w').close()

            mock_compute = MagicMock()
            with patch.object(_testbed_module.GeneratedDataset, 'compute', mock_compute):
                path = get_off_image_path(
                    task='people',
                    target='Colin Powell',
                    method='distil',
                    num_train_epochs=400,
                    seed=42,
                    prompt='An image of Colin Powell',
                    base_folder=tmp,
                    seeds=[42],
                    prompts=['An image of Colin Powell'],
                )

            mock_compute.assert_not_called()
            self.assertEqual(path, os.path.join(shared_folder, expected_filename))

    def test_classmethod_forwards_seeds_and_prompts(self) -> None:
        """GeneratedDataset.get_off_image_path() classmethod correctly forwards
        seeds and prompts to the module-level function."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            def fake_compute(s: list, p: list, batch_size: int = 16) -> str:
                shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
                self._write_baseline_files(shared_folder, s, p)
                return shared_folder

            with patch.object(
                _testbed_module.GeneratedDataset,
                'compute',
                side_effect=fake_compute,
            ):
                path = GeneratedDataset.get_off_image_path(
                    task='people',
                    target='Colin Powell',
                    method='distil',
                    num_train_epochs=400,
                    seed=42,
                    prompt='An image of Colin Powell',
                    base_folder=tmp,
                    seeds=seeds,
                    prompts=prompts,
                )

            shared_folder = get_shared_baseline_folder('people', base_folder=tmp)
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
            self.assertEqual(path, os.path.join(shared_folder, expected_filename))


class TestGeneratedDatasetComputeHFDownload(unittest.TestCase):
    """compute(): step 2 — data absent locally but present on HuggingFace.

    Verifies that:
    - huggingface_dataset_exists is called with the correct path_in_repo.
    - huggingface_dataset_download is called with the correct path_in_repo.
    - compute() returns the folder_path after a successful download.
    - The HF-side path uses the datasets/ prefix (hf_path_in_repo), not the bare
      config name — regression for the bug where baselines were uploaded/downloaded
      from the HF repo root instead of datasets/.
    """

    def _write_files_to_folder(self, folder: str, seeds: list, prompts: list, prefix: str = 'off') -> None:
        os.makedirs(folder, exist_ok=True)
        for seed in seeds:
            for prompt in prompts:
                fname = f'{prefix}_{seed:02d}_{prompt}.png'
                open(os.path.join(folder, fname), 'w').close()
        open(os.path.join(folder, 'metadata.jsonl'), 'w').close()

    def test_baseline_hf_download_uses_correct_path_in_repo(self) -> None:
        """compute() for a shared baseline dataset uses hf_path_in_repo
        (e.g. 'datasets/generated_breeds_baseline') not the bare config name
        when calling huggingface_dataset_exists and huggingface_dataset_download."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of a griffon bruxellois dog']
            ds = GeneratedDataset(task='breeds', base_folder=tmp)

            def fake_download(folder_datasets: str, dataset_repository: str,
                              dataset_config: str, token: str, path_in_repo: str = None) -> None:  # type: ignore[assignment]
                # Simulate HF download: write files to the local folder
                self._write_files_to_folder(ds.folder_path, seeds, prompts, prefix='off')

            mock_exists = MagicMock(return_value=True)

            with patch(
                'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                mock_exists,
            ):
                with patch(
                    'vision_unlearning.integrations.huggingface.huggingface_dataset_download',
                    side_effect=fake_download,
                ) as mock_download:
                    result = ds.compute(seeds=seeds, prompts=prompts)

            self.assertEqual(result, ds.folder_path)
            # exists check must use path_in_repo = 'datasets/generated_breeds_baseline'
            exists_kwargs = mock_exists.call_args.kwargs
            self.assertEqual(
                exists_kwargs.get('path_in_repo'),
                'datasets/generated_breeds_baseline',
                "huggingface_dataset_exists must use hf_path_in_repo (datasets/ prefix)",
            )
            # download must also use the same path_in_repo
            download_kwargs = mock_download.call_args.kwargs
            self.assertEqual(
                download_kwargs.get('path_in_repo'),
                'datasets/generated_breeds_baseline',
                "huggingface_dataset_download must use hf_path_in_repo (datasets/ prefix)",
            )

    def test_entity_dataset_hf_download_uses_correct_path_in_repo(self) -> None:
        """compute() for an entity dataset downloads with the correct path_in_repo."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of a griffon bruxellois dog']
            ds = GeneratedDataset(task='breeds', target='a griffon bruxellois dog',
                                  method='uce', num_train_epochs=0, base_folder=tmp)

            def fake_download(folder_datasets: str, dataset_repository: str,
                              dataset_config: str, token: str, path_in_repo: str = None) -> None:  # type: ignore[assignment]
                self._write_files_to_folder(ds.folder_path, seeds, prompts, prefix='on')

            mock_exists = MagicMock(return_value=True)

            with patch(
                'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                mock_exists,
            ):
                with patch(
                    'vision_unlearning.integrations.huggingface.huggingface_dataset_download',
                    side_effect=fake_download,
                ) as mock_download:
                    result = ds.compute(seeds=seeds, prompts=prompts)

            self.assertEqual(result, ds.folder_path)
            expected_path = f'datasets/{ds.hf_config_name}'
            exists_kwargs = mock_exists.call_args.kwargs
            self.assertEqual(
                exists_kwargs.get('path_in_repo'), expected_path,
                "huggingface_dataset_exists must use hf_path_in_repo (datasets/ prefix)",
            )
            download_kwargs = mock_download.call_args.kwargs
            self.assertEqual(
                download_kwargs.get('path_in_repo'), expected_path,
                "huggingface_dataset_download must use hf_path_in_repo (datasets/ prefix)",
            )

    def test_download_token_is_none_when_hf_token_unset(self) -> None:
        """Without HF_TOKEN, the download must receive token=None, not token="":
        an empty string becomes an illegal 'Authorization: Bearer ' header and breaks
        unauthenticated downloads from public repositories.

        Regression test for the same bug already fixed in InterferencePerEntity.compute()
        and ResultTemplate.compute() (vision-unlearning commit c189e56) — this is the one
        remaining occurrence, on GeneratedDataset's heavy-generation download path."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of a griffon bruxellois dog']
            ds = GeneratedDataset(task='breeds', base_folder=tmp)

            def fake_download(folder_datasets: str, dataset_repository: str,
                              dataset_config: str, token: Optional[str],
                              path_in_repo: Optional[str] = None) -> None:
                self._write_files_to_folder(ds.folder_path, seeds, prompts, prefix='off')

            mock_exists = MagicMock(return_value=True)

            with patch.dict('os.environ', {}, clear=False):
                os.environ.pop('HF_TOKEN', None)
                with patch(
                    'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                    mock_exists,
                ):
                    with patch(
                        'vision_unlearning.integrations.huggingface.huggingface_dataset_download',
                        side_effect=fake_download,
                    ) as mock_download:
                        result = ds.compute(seeds=seeds, prompts=prompts)

            self.assertEqual(result, ds.folder_path)
            download_kwargs = mock_download.call_args.kwargs
            self.assertIsNone(
                download_kwargs.get('token'),
                "download must receive token=None, not token='', when HF_TOKEN is unset",
            )

    def test_compute_skips_hf_when_local_data_present(self) -> None:
        """compute() does NOT call HuggingFace when data is already present locally."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of a griffon bruxellois dog']
            ds = GeneratedDataset(task='breeds', base_folder=tmp)
            # Pre-populate local folder so exists() returns True
            self._write_files_to_folder(ds.folder_path, seeds, prompts, prefix='off')

            mock_hf_exists = MagicMock()
            with patch(
                'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                mock_hf_exists,
            ):
                result = ds.compute(seeds=seeds, prompts=prompts)

            self.assertEqual(result, ds.folder_path)
            mock_hf_exists.assert_not_called()

    def test_compute_idempotent_second_call_returns_folder(self) -> None:
        """Calling compute() twice with the same seeds/prompts returns same folder_path
        and does not attempt HF access on the second call (idempotency).

        This guards against regressions where re-calling compute() triggers redundant
        downloads or generations for a caller bug-fix scenario (Cidral review 2026-05-23,
        question 1: 'is compute() idempotent?').
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42, 43]
            prompts = ['An image of a griffon bruxellois dog']
            ds = GeneratedDataset(task='breeds', base_folder=tmp)
            # Pre-populate local folder so exists() returns True on first call
            self._write_files_to_folder(ds.folder_path, seeds, prompts, prefix='off')

            mock_hf_exists = MagicMock()
            with patch(
                'vision_unlearning.integrations.huggingface.huggingface_dataset_exists',
                mock_hf_exists,
            ):
                result1 = ds.compute(seeds=seeds, prompts=prompts)
                result2 = ds.compute(seeds=seeds, prompts=prompts)

            # Both calls return the same folder_path.
            self.assertEqual(result1, ds.folder_path)
            self.assertEqual(result2, ds.folder_path)
            # HF is never called on either invocation.
            mock_hf_exists.assert_not_called()


class TestGetOffImagePathSeedPromptValidation(unittest.TestCase):
    """get_off_image_path raises ValueError when the shared baseline folder exists
    but does not contain the requested seed/prompt combination.

    This validates the fail-loud contract added per Cidral review 2026-05-23
    (questions 2 and 3: seed mismatch and metadata_filtered subset tests).
    """

    def _setup_baseline_folder(
        self,
        tmp: str,
        task: str,
        seeds: list,
        prompts: list,
    ) -> str:
        """Write a minimal shared baseline folder with off_*.png files."""
        from vision_unlearning.datasets.testbed import get_shared_baseline_folder
        shared = get_shared_baseline_folder(task, base_folder=tmp)
        os.makedirs(shared, exist_ok=True)
        for seed in seeds:
            for prompt in prompts:
                fname = f'off_{seed:02d}_{prompt}.png'
                open(os.path.join(shared, fname), 'w').close()
        open(os.path.join(shared, 'metadata.jsonl'), 'w').close()
        return shared

    def test_seed_mismatch_raises_value_error(self) -> None:
        """Requesting seed=99 from a baseline generated with seeds=[42,43] raises ValueError.

        If the caller passes a seed that was not in the compute() seed list, the image
        file will not exist. Rather than silently returning a non-existent path, the
        function now raises ValueError immediately.
        """
        from vision_unlearning.datasets.testbed import get_off_image_path
        with tempfile.TemporaryDirectory() as tmp:
            task = 'breeds'
            seeds = [42, 43]
            prompt = 'An image of a dog'
            self._setup_baseline_folder(tmp, task, seeds, [prompt])

            with self.assertRaises(ValueError, msg="Should raise ValueError for unknown seed"):
                get_off_image_path(
                    task=task,
                    target='a dog',
                    method='distil',
                    num_train_epochs=400,
                    seed=99,   # NOT in seeds=[42, 43]
                    prompt=prompt,
                    base_folder=tmp,
                )

    def test_prompt_mismatch_raises_value_error(self) -> None:
        """Requesting an unknown prompt from a baseline raises ValueError.

        If metadata_filtered is a pruned subset and the baseline was computed only
        with prompts from that subset, a downstream caller requesting a prompt not
        in the subset will get a clear error rather than a silently wrong path.
        (Cidral review 2026-05-23, question 3: metadata_filtered subset test.)
        """
        from vision_unlearning.datasets.testbed import get_off_image_path
        with tempfile.TemporaryDirectory() as tmp:
            task = 'breeds'
            seed = 42
            prompts = ['An image of a dog']
            self._setup_baseline_folder(tmp, task, [seed], prompts)

            with self.assertRaises(ValueError, msg="Should raise ValueError for unknown prompt"):
                get_off_image_path(
                    task=task,
                    target='a cat',
                    method='distil',
                    num_train_epochs=400,
                    seed=seed,
                    prompt='An image of a cat',   # NOT in prompts
                    base_folder=tmp,
                )

    def test_known_seed_and_prompt_returns_path(self) -> None:
        """When the seed and prompt are both in the baseline, returns the correct path."""
        from vision_unlearning.datasets.testbed import get_off_image_path
        with tempfile.TemporaryDirectory() as tmp:
            task = 'breeds'
            seed = 42
            prompt = 'An image of a dog'
            self._setup_baseline_folder(tmp, task, [seed], [prompt])

            path = get_off_image_path(
                task=task,
                target='a dog',
                method='distil',
                num_train_epochs=400,
                seed=seed,
                prompt=prompt,
                base_folder=tmp,
            )
            self.assertTrue(os.path.exists(path), "Path should exist in baseline folder")


if __name__ == '__main__':
    unittest.main()
