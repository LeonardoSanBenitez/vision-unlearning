"""Pinned inputs: what a campaign loads is fixed by commit or hash, and recorded once per campaign.

Fails on the code before: DINOv2 was loaded from the default branch of its repository, the breeds
source was cloned at whatever its default branch held, and nothing recorded which Hub revisions,
which DINOv2 code or which environment a run had actually used.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, List, Sequence

import pytest

from vision_unlearning.benchmarks.I_care import capture_pins

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(relative: str) -> str:
    with open(os.path.join(REPO, relative), encoding='utf-8') as handle:
        return handle.read()


def test_dinov2_is_loaded_at_a_pinned_commit() -> None:
    source = _source('vision_unlearning/benchmarks/I_care/embeddings.py')
    assert 'DINOV2_HUB_REFERENCE = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"' in source
    assert re.search(r'torch\.hub\.load\(DINOV2_HUB_REFERENCE,', source)
    assert 'torch.hub.load("facebookresearch/dinov2"' not in source


def test_the_breeds_source_is_checked_out_at_its_pinned_commit() -> None:
    source = _source('vision_unlearning/datasets/others.py')
    assert 'TARAS_BREEDS_COMMIT = "7164ba68ce2902e680f92432094a3e028a878dbe"' in source
    assert '"checkout", "--quiet", TARAS_BREEDS_COMMIT' in source


def _fixture_caches(tmp_path: Any) -> Any:
    hf = tmp_path / 'hf'
    for role, folder in capture_pins.HUB_REPOSITORIES.items():
        refs = hf / folder / 'refs'
        refs.mkdir(parents=True)
        (refs / 'main').write_text(f'commit-of-{role}\n')
    hub = tmp_path / 'torchhub'
    code = hub / 'facebookresearch_dinov2_7764ea0f'
    code.mkdir(parents=True)
    (code / 'hubconf.py').write_text('dependencies = []\n')
    (hub / 'checkpoints').mkdir()
    (hub / 'checkpoints' / capture_pins.DINOV2_CHECKPOINT).write_bytes(b'weights')
    archive = tmp_path / 'SUNAttributeDB.tar.gz'
    archive.write_bytes(b'archive')
    return hf, hub, archive


def _fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    def run_to_file(command: Sequence[str], path: str) -> bool:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(' '.join(command) + '\n')
        return True
    monkeypatch.setattr(capture_pins, '_run_to_file', run_to_file)


def test_the_capture_writes_every_field(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tools(monkeypatch)
    hf, hub, archive = _fixture_caches(tmp_path)
    pins = capture_pins.capture(str(tmp_path / 'out'), str(hf), str(hub), scenes_archives=[str(archive)])
    with open(tmp_path / 'out' / 'pins.json', encoding='utf-8') as handle:
        assert json.load(handle) == pins
    assert pins['huggingface'] == {role: f'commit-of-{role}' for role in capture_pins.HUB_REPOSITORIES}
    assert pins['torch_hub']['dinov2_checkout'] == 'facebookresearch_dinov2_7764ea0f'
    assert all(len(pins['torch_hub'][k]) == 64 for k in ('dinov2_code_sha256', 'dinov2_checkpoint_sha256'))
    assert list(pins['sources']['scenes_archives_sha256']) == ['SUNAttributeDB.tar.gz']
    assert all(pins[k] for k in ('freeze_sha256', 'conda_sha256', 'sbom_sha256'))
    assert pins['missing'] == []
    for name in ('freeze.txt', 'conda.txt', 'sbom.json'):
        assert (tmp_path / 'out' / name).exists()


def test_the_code_hash_follows_the_code(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tools(monkeypatch)
    hf, hub, _ = _fixture_caches(tmp_path)
    before = capture_pins.capture(str(tmp_path / 'a'), str(hf), str(hub))['torch_hub']['dinov2_code_sha256']
    (hub / 'facebookresearch_dinov2_7764ea0f' / 'hubconf.py').write_text('dependencies = ["torch"]\n')
    after = capture_pins.capture(str(tmp_path / 'b'), str(hf), str(hub))['torch_hub']['dinov2_code_sha256']
    assert before != after


def test_a_missing_pin_fails_unless_allowed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tools(monkeypatch)
    hf, hub, _ = _fixture_caches(tmp_path)
    os.remove(hf / capture_pins.HUB_REPOSITORIES['clip_scorer'] / 'refs' / 'main')
    with pytest.raises(capture_pins.MissingPin, match='huggingface.clip_scorer'):
        capture_pins.capture(str(tmp_path / 'out'), str(hf), str(hub))
    pins = capture_pins.capture(str(tmp_path / 'out'), str(hf), str(hub), allow_missing=True)
    assert pins['huggingface']['clip_scorer'] is None and pins['missing'] == ['huggingface.clip_scorer']


def test_the_freeze_is_the_real_pip_freeze(tmp_path: Any) -> None:
    assert capture_pins._run_to_file([sys.executable, '-m', 'pip', 'freeze', '--all'], str(tmp_path / 'f.txt'))
    lines: List[str] = (tmp_path / 'f.txt').read_text().splitlines()
    assert any(line.lower().startswith('pip==') for line in lines)


def test_a_repository_loaded_by_commit_is_pinned_by_its_only_snapshot(tmp_path: Any) -> None:
    folder = tmp_path / 'datasets--bitmind--lfw' / 'snapshots' / '274dfbca'
    folder.mkdir(parents=True)
    assert capture_pins.hub_commit(str(tmp_path), 'datasets--bitmind--lfw') == '274dfbca'
    (tmp_path / 'datasets--bitmind--lfw' / 'snapshots' / 'another').mkdir()
    assert capture_pins.hub_commit(str(tmp_path), 'datasets--bitmind--lfw') is None
