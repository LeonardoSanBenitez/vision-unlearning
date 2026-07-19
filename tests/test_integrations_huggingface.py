"""Tests for vision_unlearning.integrations.huggingface.

Scope: get_hf_token_from_env(), the shared HF_TOKEN normalization helper used
by InterferencePerEntity.compute(), ResultTemplate.compute(), and
GeneratedDataset.compute() to avoid passing an empty-string token (which
builds an illegal 'Authorization: Bearer ' header) instead of None.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import List
from unittest.mock import patch

from vision_unlearning.integrations.huggingface import (
    get_hf_token_from_env,
    huggingface_dataset_download,
)

_MOD = 'vision_unlearning.integrations.huggingface'


class TestGetHfTokenFromEnv(unittest.TestCase):

    def test_returns_none_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_TOKEN', None)
            self.assertIsNone(get_hf_token_from_env())

    def test_returns_none_when_set_to_empty_string(self) -> None:
        with patch.dict(os.environ, {'HF_TOKEN': ''}, clear=False):
            self.assertIsNone(get_hf_token_from_env())

    def test_returns_token_when_set_to_real_value(self) -> None:
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_real_token_value'}, clear=False):
            self.assertEqual(get_hf_token_from_env(), 'hf_real_token_value')


class TestHuggingfaceDatasetDownload(unittest.TestCase):
    """huggingface_dataset_download: skip/clean semantics and partial-folder cleanup.

    snapshot_download is patched throughout, so no real network call is made.
    """

    def _fake_cache(self, cache_root: str, hf_path: str, filenames: List[str]) -> str:
        # Emulate snapshot_download's return: a repo_path whose files live at
        # repo_path/<hf_path>/*, matching the copy loop in the function under test.
        repo_path = os.path.join(cache_root, 'repo')
        src = os.path.join(repo_path, hf_path)
        os.makedirs(src, exist_ok=True)
        for name in filenames:
            open(os.path.join(src, name), 'w').close()
        return repo_path

    def test_overwrite_true_fills_partial_folder_in_place(self) -> None:
        """overwrite=True downloads into an existing folder (does not skip, does not delete it).

        The missing files are added and the folder itself is preserved -- the bind-mount-safe
        way to complete an interrupted download without an rmdir the container may be denied.
        """
        with tempfile.TemporaryDirectory() as tmp:
            folder_datasets = os.path.join(tmp, 'datasets')
            config = 'generated_x'
            dest = os.path.join(folder_datasets, config)
            os.makedirs(dest)
            open(os.path.join(dest, 'a.png'), 'w').close()  # one of two already present

            repo_path = self._fake_cache(tmp, config, ['a.png', 'b.png'])
            with patch(f'{_MOD}.snapshot_download', return_value=repo_path) as mock_snap:
                huggingface_dataset_download(
                    folder_datasets=folder_datasets,
                    dataset_repository='repo/id',
                    dataset_config=config,
                    token=None,
                    overwrite=True,
                    folder_cache=os.path.join(tmp, 'cache'),
                )
            mock_snap.assert_called_once()  # did NOT skip
            self.assertTrue(os.path.exists(os.path.join(dest, 'a.png')))
            self.assertTrue(os.path.exists(os.path.join(dest, 'b.png')))  # missing file added

    def test_clean_true_removes_then_redownloads(self) -> None:
        """clean=True wipes a pre-existing folder's stale content and refetches."""
        with tempfile.TemporaryDirectory() as tmp:
            folder_datasets = os.path.join(tmp, 'datasets')
            config = 'generated_x'
            dest = os.path.join(folder_datasets, config)
            os.makedirs(dest)
            open(os.path.join(dest, 'stale.txt'), 'w').close()

            repo_path = self._fake_cache(tmp, config, ['a.png', 'b.png'])
            with patch(f'{_MOD}.snapshot_download', return_value=repo_path) as mock_snap:
                huggingface_dataset_download(
                    folder_datasets=folder_datasets,
                    dataset_repository='repo/id',
                    dataset_config=config,
                    token=None,
                    clean=True,
                    folder_cache=os.path.join(tmp, 'cache'),
                )
            mock_snap.assert_called_once()
            self.assertFalse(os.path.exists(os.path.join(dest, 'stale.txt')))  # wiped
            self.assertTrue(os.path.exists(os.path.join(dest, 'a.png')))

    def test_skips_when_folder_exists_and_not_overwrite(self) -> None:
        """Default (no overwrite/clean) keeps the historical skip-if-folder-present behavior."""
        with tempfile.TemporaryDirectory() as tmp:
            folder_datasets = os.path.join(tmp, 'datasets')
            config = 'generated_x'
            os.makedirs(os.path.join(folder_datasets, config))
            with patch(f'{_MOD}.snapshot_download') as mock_snap:
                huggingface_dataset_download(
                    folder_datasets=folder_datasets,
                    dataset_repository='repo/id',
                    dataset_config=config,
                    token=None,
                )
            mock_snap.assert_not_called()  # skipped

    def test_created_folder_removed_on_download_failure(self) -> None:
        """A folder this call CREATED must not survive a failed download (no stranded poison)."""
        with tempfile.TemporaryDirectory() as tmp:
            folder_datasets = os.path.join(tmp, 'datasets')
            config = 'generated_x'
            dest = os.path.join(folder_datasets, config)
            with patch(f'{_MOD}.snapshot_download', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    huggingface_dataset_download(
                        folder_datasets=folder_datasets,
                        dataset_repository='repo/id',
                        dataset_config=config,
                        token=None,
                        folder_cache=os.path.join(tmp, 'cache'),
                    )
            self.assertFalse(os.path.exists(dest))  # cleaned up, not left behind

    def test_preexisting_folder_kept_on_overwrite_failure(self) -> None:
        """A folder that already existed is NOT deleted when an overwrite download fails.

        Its files may let the next overwrite pass resume, and on a bind mount the
        directory may not be removable by this process anyway.
        """
        with tempfile.TemporaryDirectory() as tmp:
            folder_datasets = os.path.join(tmp, 'datasets')
            config = 'generated_x'
            dest = os.path.join(folder_datasets, config)
            os.makedirs(dest)
            open(os.path.join(dest, 'partial.png'), 'w').close()
            with patch(f'{_MOD}.snapshot_download', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    huggingface_dataset_download(
                        folder_datasets=folder_datasets,
                        dataset_repository='repo/id',
                        dataset_config=config,
                        token=None,
                        overwrite=True,
                        folder_cache=os.path.join(tmp, 'cache'),
                    )
            self.assertTrue(os.path.exists(os.path.join(dest, 'partial.png')))  # preserved


if __name__ == '__main__':
    unittest.main()
