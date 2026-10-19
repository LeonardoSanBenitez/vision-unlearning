"""The I-CARE upload helper: which repository it writes to, what it uploads, and how it checks it.

The Hub is replaced by a fake client that records every call and answers listings from what it was
"uploaded", so these run without network. Every test fails on the code before the change, because
no such helper existed: each script uploaded through the bare client calls to whatever repository it
held, a folder was uploaded whatever it contained, and nothing compared the result with the local
files.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from vision_unlearning.benchmarks.I_care import upload
from vision_unlearning.benchmarks.I_care.upload import (
    CAMPAIGN_REPOSITORY,
    IncompleteArtifact,
    UploadMismatch,
    UploadRefused,
    git_blob_id,
    sha256_file,
    upload_file,
    upload_folder,
)

OLD_CORPUS = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'


class FakeHub:
    """Records uploads; answers listings from the bytes it received, unless told to lie.

    ``lfs_suffixes`` decides which files it reports as Large File Storage objects (with a sha256)
    and which as plain git files (with a blob id), as the Hub does by extension.
    """

    def __init__(self, corrupt: Optional[str] = None, lfs_suffixes: tuple = ('.png', '.json')) -> None:
        self.calls: List[str] = []
        self.stored: Dict[str, str] = {}
        self.corrupt = corrupt
        self.lfs_suffixes = lfs_suffixes

    def _store(self, local: str, remote: str) -> None:
        # A copy of the bytes as uploaded, so a later change to the local file does not change them.
        snapshot = os.path.join(tempfile.mkdtemp(), os.path.basename(local))
        shutil.copyfile(local, snapshot)
        self.stored[remote] = snapshot

    def upload_file(self, path_or_fileobj: str, path_in_repo: str, **kwargs: Any) -> None:
        self.calls.append(f"upload_file {kwargs['repo_id']} {path_in_repo}")
        self._store(path_or_fileobj, path_in_repo)

    def upload_folder(self, folder_path: str, path_in_repo: str, allow_patterns: List[str], **kwargs: Any) -> None:
        self.calls.append(f"upload_folder {kwargs['repo_id']} {path_in_repo}")
        for root, _, files in os.walk(folder_path):
            for name in files:
                relative = os.path.relpath(os.path.join(root, name), folder_path).replace(os.sep, '/')
                if relative in allow_patterns:
                    self._store(os.path.join(root, name), f'{path_in_repo}/{relative}')

    def _entry(self, remote: str) -> Any:
        local = self.stored[remote]
        if remote.endswith(self.lfs_suffixes):
            sha = sha256_file(local) if remote != self.corrupt else '0' * 64
            return SimpleNamespace(path=remote, lfs=SimpleNamespace(sha256=sha), blob_id='ignored')
        blob = git_blob_id(local) if remote != self.corrupt else '0' * 40
        return SimpleNamespace(path=remote, lfs=None, blob_id=blob)

    def get_paths_info(self, repository: str, paths: List[str], **kwargs: Any) -> List[Any]:
        self.calls.append('get_paths_info')
        return [self._entry(path) for path in paths if path in self.stored]

    def list_repo_tree(self, repository: str, path_in_repo: str, **kwargs: Any) -> List[Any]:
        self.calls.append('list_repo_tree')
        return [self._entry(path) for path in self.stored if path.startswith(path_in_repo + '/')]


def _folder(tmp_path: Any, names: List[str]) -> str:
    folder = tmp_path / 'generated_people_x'
    folder.mkdir()
    for index, name in enumerate(names):
        (folder / name).write_bytes(f'image {index}'.encode())
    return str(folder)


# -- the guard -------------------------------------------------------------------------------


@pytest.mark.parametrize('target', [OLD_CORPUS, 'someone/else', 'vision-unlearning/i-care-scratch'])
def test_an_upload_to_any_other_repository_is_refused_before_any_network_call(tmp_path: Any, target: str) -> None:
    hub = FakeHub()
    path = tmp_path / 'a.json'
    path.write_text('{}')
    with pytest.raises(UploadRefused, match='refusing to upload'):
        upload_file(str(path), 'datasets/a.json', target, 'token', api=hub)
    with pytest.raises(UploadRefused):
        upload_folder(_folder(tmp_path, ['0.png']), 'datasets/x', target, 'token', ['0.png'], api=hub)
    assert hub.calls == []


def test_the_guard_needs_no_client_at_all() -> None:
    """Refusal happens before a client would be constructed, so no token or network is touched."""
    with pytest.raises(UploadRefused):
        upload.assert_upload_allowed(OLD_CORPUS)
    upload.assert_upload_allowed(CAMPAIGN_REPOSITORY)


# -- completeness ----------------------------------------------------------------------------


def test_an_incomplete_folder_is_never_uploaded(tmp_path: Any) -> None:
    hub = FakeHub()
    folder = _folder(tmp_path, ['0.png', '1.png'])
    with pytest.raises(IncompleteArtifact, match='lacks 1 of its 3 files'):
        upload_folder(folder, 'datasets/x', CAMPAIGN_REPOSITORY, 'token', ['0.png', '1.png', '2.png'], api=hub)
    assert hub.calls == []


def test_a_complete_folder_is_uploaded_once_without_strays(tmp_path: Any) -> None:
    hub = FakeHub()
    folder = _folder(tmp_path, ['0.png', '1.png', 'metadata.jsonl', 'stray.tmp'])
    digests = upload_folder(folder, 'datasets/x', CAMPAIGN_REPOSITORY, 'token',
                            ['0.png', '1.png', 'metadata.jsonl'], api=hub)
    assert [call for call in hub.calls if call.startswith('upload')] == [f'upload_folder {CAMPAIGN_REPOSITORY} datasets/x']
    assert sorted(hub.stored) == ['datasets/x/0.png', 'datasets/x/1.png', 'datasets/x/metadata.jsonl']
    assert digests['datasets/x/0.png'] == sha256_file(os.path.join(folder, '0.png'))


# -- the content check -----------------------------------------------------------------------


def test_a_remote_file_whose_bytes_differ_is_reported(tmp_path: Any) -> None:
    folder = _folder(tmp_path, ['0.png', '1.png'])
    with pytest.raises(UploadMismatch, match='datasets/x/1.png'):
        upload_folder(folder, 'datasets/x', CAMPAIGN_REPOSITORY, 'token', ['0.png', '1.png'],
                      api=FakeHub(corrupt='datasets/x/1.png'))


def test_a_local_file_changed_after_its_upload_is_reported(tmp_path: Any) -> None:
    """Fails on the code before: nothing could re-check a published file without uploading it again."""
    hub = FakeHub()
    image = tmp_path / 'probe.png'
    image.write_bytes(bytes(range(256)))
    upload_file(str(image), 'scratch/probe.png', CAMPAIGN_REPOSITORY, 'token', api=hub)
    assert upload.remote_mismatches([(str(image), 'scratch/probe.png')], CAMPAIGN_REPOSITORY, 'token', api=hub) == []
    data = bytearray(image.read_bytes())
    data[100] ^= 0x01
    image.write_bytes(bytes(data))
    problems = upload.remote_mismatches([(str(image), 'scratch/probe.png')], CAMPAIGN_REPOSITORY, 'token', api=hub)
    assert len(problems) == 1 and problems[0].startswith('scratch/probe.png: remote sha256')
    assert upload.remote_mismatches([(str(image), 'scratch/absent.png')], CAMPAIGN_REPOSITORY, 'token', api=hub) == [
        'scratch/absent.png: not on the remote']


def test_a_file_stored_in_git_is_checked_by_its_blob_id(tmp_path: Any) -> None:
    readme = tmp_path / 'README.md'
    readme.write_text('# corpus\n')
    assert upload_file(str(readme), 'README.md', CAMPAIGN_REPOSITORY, 'token', api=FakeHub())
    with pytest.raises(UploadMismatch, match='git blob id'):
        upload_file(str(readme), 'README.md', CAMPAIGN_REPOSITORY, 'token', api=FakeHub(corrupt='README.md'))


def test_the_local_digest_is_taken_before_the_upload(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote copy is never hashed against itself: the local record exists before any call."""
    order: List[str] = []
    hub = FakeHub()
    real_upload = hub.upload_file
    monkeypatch.setattr(upload, 'sha256_file', lambda path: order.append('hash') or sha256_file(path))
    monkeypatch.setattr(hub, 'upload_file', lambda **kwargs: order.append('upload') or real_upload(**kwargs))
    path = tmp_path / 'a.json'
    path.write_text('{}')
    upload_file(str(path), 'datasets/a.json', CAMPAIGN_REPOSITORY, 'token', api=hub)
    assert order[0] == 'hash' and order.index('upload') > 0


def test_git_blob_id_matches_git() -> None:
    """``git hash-object`` of the bytes 'hello\\n' is ce013625030ba8dba906f756967f9e9ca394464a."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(b'hello\n')
    try:
        assert git_blob_id(handle.name) == 'ce013625030ba8dba906f756967f9e9ca394464a'
    finally:
        os.unlink(handle.name)


# -- one repository argument, and every uploading script guarded -----------------------------

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vision_unlearning')
EXCLUDED = (os.sep + 'u_care' + os.sep, os.sep + 'hub_care' + os.sep, os.sep + 'ablations' + os.sep, os.sep + 'reports' + os.sep)


def _string_constants(path: str) -> List[str]:
    import ast
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_each_repository_name_is_written_in_exactly_one_place() -> None:
    """Fails on the code before: five scripts each typed the first corpus's name themselves."""
    owners = {OLD_CORPUS: 'artifact.py', CAMPAIGN_REPOSITORY: 'upload.py'}
    offenders: List[str] = []
    for root, _, files in os.walk(PACKAGE):
        for name in files:
            path = os.path.join(root, name)
            if not name.endswith('.py') or any(part in path for part in EXCLUDED):
                continue
            for literal in set(_string_constants(path)) & set(owners):
                if name != owners[literal]:
                    offenders.append(f'{os.path.relpath(path, PACKAGE)}: {literal}')
    assert not offenders, offenders


@pytest.mark.parametrize('script', [
    'pipeline_05_compute_embeddings.py',
    'pipeline_06_compute_interference_per_pair.py',
    'pipeline_08_run_all_rts.py',
    'pipeline_09_synchronize_huggingface.py',
])
def test_every_uploading_script_checks_its_target_at_start_up(script: str) -> None:
    with open(os.path.join(PACKAGE, 'benchmarks', 'I_care', script), encoding='utf-8') as handle:
        source = handle.read()
    assert 'assert_upload_allowed(' in source
    assert 'huggingface_dataset_file_upload(' not in source and 'huggingface_dataset_upload(' not in source
