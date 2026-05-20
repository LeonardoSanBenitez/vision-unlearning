"""Tests for vision_unlearning.datasets.testbed.

Covers:
- get_baseline_dataset_folder(): path construction.
- get_off_image_path(): baseline folder preferred when it exists on disk;
  falls back to entity folder when baseline folder is absent.
- exists_unlearned_dataset(): counts only on_* images now (not on_* + off_*).
"""
from __future__ import annotations

import os
import tempfile
import unittest

from vision_unlearning.datasets.testbed import (
    get_baseline_dataset_folder,
    get_generated_dataset_file,
    get_generated_dataset_folder,
    get_off_image_path,
    exists_unlearned_dataset,
)


class TestGetBaselineDatasetFolder(unittest.TestCase):
    """get_baseline_dataset_folder returns the correct path."""

    def test_path_format(self) -> None:
        path = get_baseline_dataset_folder('people', 'George W Bush', base_folder='assets')
        self.assertEqual(path, os.path.join('assets', 'datasets', 'generated_people_baseline_George W Bush'))

    def test_default_base_folder(self) -> None:
        path = get_baseline_dataset_folder('scenes', 'abbey')
        self.assertEqual(path, os.path.join('assets', 'datasets', 'generated_scenes_baseline_abbey'))

    def test_custom_base_folder(self) -> None:
        path = get_baseline_dataset_folder('breeds', 'poodle', base_folder='/tmp/my_assets')
        self.assertEqual(path, os.path.join('/tmp/my_assets', 'datasets', 'generated_breeds_baseline_poodle'))

    def test_no_method_in_path(self) -> None:
        """Baseline folder must never include a method token."""
        path = get_baseline_dataset_folder('people', 'Colin Powell')
        self.assertNotIn('distil', path)
        self.assertNotIn('uce', path)
        self.assertNotIn('munba', path)

    def test_baseline_token_present(self) -> None:
        """Baseline folder must include 'baseline' as a literal token."""
        path = get_baseline_dataset_folder('people', 'Colin Powell')
        self.assertIn('baseline', path)


class TestGetOffImagePath(unittest.TestCase):
    """get_off_image_path returns baseline folder path when it exists."""

    def test_prefers_baseline_folder_when_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_folder = get_baseline_dataset_folder('people', 'Colin Powell', base_folder=tmp)
            os.makedirs(baseline_folder, exist_ok=True)

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
            self.assertEqual(path, os.path.join(baseline_folder, expected_filename))

    def test_falls_back_to_entity_folder_when_baseline_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Do NOT create the baseline folder — test the fallback path.
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
    """exists_unlearned_dataset counts only on_* images (no off_* any more)."""

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

    def test_returns_false_when_extra_off_images_present(self) -> None:
        """Old mixed-folder format (on_ + off_) has too many files — should return False."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [42]
            prompts = ['An image of Alice']
            # Old format: 1 on_ + 1 off_ + metadata.jsonl = 3 files
            # New expected count: 1 * 1 + 1 = 2 files → mismatch → False
            files = ['on_42_An image of Alice.png', 'off_42_An image of Alice.png', 'metadata.jsonl']
            self._write_files(tmp, files)
            self.assertFalse(exists_unlearned_dataset(tmp, seeds, prompts))

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


if __name__ == '__main__':
    unittest.main()
