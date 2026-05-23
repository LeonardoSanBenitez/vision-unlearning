"""Tests for vision_unlearning.datasets.testbed.

Covers:
- get_shared_baseline_folder(): task-level shared baseline path construction.
- get_off_image_path(): shared folder preferred; falls back to entity folder when
  no baseline folder exists.
- exists_unlearned_dataset(): counts only on_* images; off_* files in legacy
  entity folders are ignored for backward compatibility.
- GeneratedDataset: OO abstraction — folder_path, is_baseline, file_path, exists,
  hf_config_name, validation, get_off_image_path class method, upload_if_recomputed.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import ValidationError

from vision_unlearning.datasets.testbed import (
    GeneratedDataset,
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_off_image_path,
    get_shared_baseline_folder,
    exists_unlearned_dataset,
)


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

            path = get_off_image_path(
                task='people',
                target='Colin Powell',
                method='distil',
                num_train_epochs=400,
                seed=42,
                prompt='An image of Colin Powell',
                base_folder=tmp,
            )
            expected_filename = get_generated_dataset_file('off', 42, 'An image of Colin Powell')
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

            def fake_compute_from_scratch(s: list, p: list) -> str:  # type: ignore[override]
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

    def test_upload_not_called_when_upload_if_recomputed_false(self) -> None:
        """Default behaviour: upload is NOT called even after scratch generation."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Colin Powell']

            ds = GeneratedDataset(task='people', base_folder=tmp, upload_if_recomputed=False)

            def fake_compute_from_scratch(s: list, p: list) -> str:  # type: ignore[override]
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


class TestGeneratedDatasetComputeFromScratchEntityRaises(unittest.TestCase):
    """_compute_from_scratch raises NotImplementedError for entity datasets."""

    def test_entity_dataset_raises_not_implemented(self) -> None:
        ds = GeneratedDataset(
            task='people', target='Colin Powell',
            method='distil', num_train_epochs=400,
        )
        with self.assertRaises(NotImplementedError):
            ds._compute_from_scratch([42], ['An image of Colin Powell'])


if __name__ == '__main__':
    unittest.main()
