"""Under HF_HUB_OFFLINE the storage cascade computes what it can, instead of failing.

Fails on the code before: the remote existence check ignored offline mode, found the file, and the
download that followed was refused by the Hub client with LocalEntryNotFoundError -- so a reader
reproducing from a downloaded revision with the Hub switched off could not compute a single
artifact that was missing locally.
"""
from __future__ import annotations

from typing import Any

import pytest

import vision_unlearning.artifact as artifact_mod
from vision_unlearning.artifact import SingleFileArtifact
from vision_unlearning.integrations.huggingface import hub_offline


class Computable(SingleFileArtifact):
    def _get_data_path_remote(self) -> str:
        return 'datasets/computable.json'

    def _compute_from_scratch(self) -> Any:
        return {'computed': True}


def _remote_has_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub holds the file, and fetching it is what offline mode refuses."""
    monkeypatch.setattr(artifact_mod, 'huggingface_dataset_file_exists', lambda *a, **k: True)

    def refused(*a: Any, **k: Any) -> None:
        raise RuntimeError('LocalEntryNotFoundError: offline mode is enabled')
    monkeypatch.setattr(artifact_mod, 'huggingface_dataset_file_download', refused)


def test_offline_a_computable_absent_artifact_is_computed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    _remote_has_everything(monkeypatch)
    assert Computable(base_folder=str(tmp_path))._resolve() == {'computed': True}
    assert (tmp_path / 'datasets' / 'computable.json').exists()


def test_online_the_remote_copy_is_still_preferred(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    _remote_has_everything(monkeypatch)
    with pytest.raises(RuntimeError, match='offline mode'):
        Computable(base_folder=str(tmp_path))._resolve()


@pytest.mark.parametrize('value,expected', [('1', True), ('on', True), ('TRUE', True), ('yes', True),
                                            ('0', False), ('', False), ('false', False)])
def test_offline_mode_is_read_with_the_hub_truth_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv('HF_HUB_OFFLINE', value)
    assert hub_offline() is expected
