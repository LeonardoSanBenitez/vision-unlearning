"""Every upload the I-CARE pipeline scripts make, guarded by repository and checked by content.

Three rules, each enforced here so that no script can forget one:

1. **Only one repository can be written to.** :data:`ALLOWED_UPLOAD_REPOSITORIES` holds the
   benchmark's current corpus, ``vision-unlearning/i-care``. Any other target -- above all the
   first corpus, ``LeonardoBenitez/VisionUnlearningEvaluationTestbeds``, which is frozen and must
   never be modified -- raises :class:`UploadRefused` before any network call. The guard exists
   because the token a pipeline runs with may be able to write elsewhere; the code, not the token,
   decides where uploads go. Scripts call :func:`assert_upload_allowed` when they parse their
   arguments, so a wrong target fails before any compute, not after it.
2. **Only a complete artifact is uploaded.** :func:`upload_folder` takes the exact list of files
   the folder must hold and refuses, with :class:`IncompleteArtifact`, when any is missing.
3. **An upload is checked by content.** The sha256 of every file is computed from the local file
   *before* it is uploaded; afterwards the remote repository's own record of each file -- the
   Large File Storage object id for files stored that way, the git blob id otherwise -- is read from
   one listing and compared. No remote bytes are downloaded, so the remote copy is never compared
   with itself. A mismatch raises :class:`UploadMismatch`. The digests are returned, so a caller can
   record exactly what it published.

Reading from any repository is unaffected: these rules apply to writes only.
"""
from __future__ import annotations

import glob
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

CAMPAIGN_REPOSITORY = 'vision-unlearning/i-care'
ALLOWED_UPLOAD_REPOSITORIES = frozenset({CAMPAIGN_REPOSITORY})

_CHUNK = 1 << 20


class UploadRefused(RuntimeError):
    """An upload was aimed at a repository the benchmark may not write to."""


class IncompleteArtifact(RuntimeError):
    """A folder was about to be uploaded without every file it must hold."""


class UploadMismatch(RuntimeError):
    """After an upload, the remote record of a file does not match the local file."""


def remote_path_of(local_path: str, base_folder: str) -> str:
    """The repository path of a local artifact: its path below `base_folder`, with forward slashes.

    Every I-CARE artifact is laid out the same way locally and remotely (``datasets/...``,
    ``models/...``, files at the root), so this is the whole mapping.
    """
    relative = os.path.relpath(local_path, base_folder)
    if relative.startswith('..'):
        raise ValueError(f'{local_path} is not inside {base_folder}')
    return relative.replace(os.sep, '/')


def assert_upload_allowed(repository: str) -> None:
    """Raise :class:`UploadRefused` unless `repository` may be written to. Makes no network call."""
    if repository not in ALLOWED_UPLOAD_REPOSITORIES:
        raise UploadRefused(
            f"refusing to upload to {repository!r}: the I-CARE pipeline writes only to "
            f"{sorted(ALLOWED_UPLOAD_REPOSITORIES)}. Pass --repository {CAMPAIGN_REPOSITORY}, or run "
            f"without uploading."
        )


def sha256_file(path: str) -> str:
    """The sha256 of a local file, which is also its Large File Storage object id."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: str) -> str:
    """The git blob id of a local file: sha1 over ``blob <size>\\0`` and the content."""
    digest = hashlib.sha1()
    digest.update(f'blob {os.path.getsize(path)}\0'.encode('ascii'))
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _local_records(pairs: Iterable[Sequence[str]]) -> Dict[str, Dict[str, str]]:
    """``{remote path: {"sha256", "git_blob_id"}}`` for ``(local path, remote path)`` pairs."""
    return {
        remote: {'sha256': sha256_file(local), 'git_blob_id': git_blob_id(local)}
        for local, remote in pairs
    }


def _mismatches(local: Dict[str, Dict[str, str]], remote_entries: Iterable[Any]) -> List[str]:
    """Every remote path whose record disagrees with the local file, or that is missing remotely."""
    remote: Dict[str, Any] = {str(getattr(entry, 'path', '')): entry for entry in remote_entries}
    problems: List[str] = []
    for path, record in sorted(local.items()):
        entry = remote.get(path)
        if entry is None:
            problems.append(f'{path}: not on the remote')
            continue
        lfs = getattr(entry, 'lfs', None)
        if lfs is not None:
            if lfs.sha256 != record['sha256']:
                problems.append(f'{path}: remote sha256 {lfs.sha256}, local {record["sha256"]}')
        elif getattr(entry, 'blob_id', None) != record['git_blob_id']:
            problems.append(
                f'{path}: remote git blob id {getattr(entry, "blob_id", None)}, local {record["git_blob_id"]}'
            )
    return problems


def _api(token: Optional[str], api: Any) -> Any:
    if api is not None:
        return api
    from huggingface_hub import HfApi
    return HfApi(token=token)


def upload_file(
    local_path: str,
    remote_path: str,
    repository: str,
    token: Optional[str],
    api: Any = None,
) -> Dict[str, str]:
    """Upload one file, then check it by content. Returns ``{remote path: sha256}``."""
    assert_upload_allowed(repository)
    remote_path = remote_path.replace(os.sep, '/')
    local = _local_records([(local_path, remote_path)])
    client = _api(token, api)
    client.upload_file(
        path_or_fileobj=local_path, path_in_repo=remote_path, repo_id=repository,
        repo_type='dataset', commit_message=f'Upload {remote_path}',
    )
    entries = client.get_paths_info(repository, [remote_path], repo_type='dataset', expand=False)
    problems = _mismatches(local, entries)
    if problems:
        raise UploadMismatch('; '.join(problems))
    return {path: record['sha256'] for path, record in local.items()}


def remote_mismatches(
    pairs: Sequence[Sequence[str]],
    repository: str,
    token: Optional[str],
    api: Any = None,
) -> List[str]:
    """Check local files against the repository's record of them, uploading nothing.

    `pairs` are ``(local path, remote path)``. Returns one line per remote path whose Large File
    Storage sha256 or git blob id differs from the local file's, or that is not on the remote; an
    empty list when every file agrees. Reading needs no permission, so no repository is refused.
    """
    local = _local_records([(str(a), str(b).replace(os.sep, '/')) for a, b in pairs])
    entries = _api(token, api).get_paths_info(repository, sorted(local), repo_type='dataset', expand=False)
    return _mismatches(local, entries)


def upload_folder(
    local_folder: str,
    remote_folder: str,
    repository: str,
    token: Optional[str],
    expected_files: Sequence[str],
    api: Any = None,
) -> Dict[str, str]:
    """Upload a folder that holds every one of `expected_files`, then check every file by content.

    `expected_files` are paths relative to the folder. Refuses with :class:`IncompleteArtifact`
    when any is missing locally, before anything is uploaded. Uploads exactly the expected files,
    so a stray file in the folder is never published. Returns ``{remote path: sha256}``.
    """
    assert_upload_allowed(repository)
    remote_folder = remote_folder.replace(os.sep, '/').rstrip('/')
    missing = [name for name in expected_files if not os.path.isfile(os.path.join(local_folder, name))]
    if missing:
        raise IncompleteArtifact(
            f'{local_folder} lacks {len(missing)} of its {len(expected_files)} files '
            f'(first: {missing[:3]}); it is not uploaded.'
        )
    pairs = [
        (os.path.join(local_folder, name), f'{remote_folder}/{name.replace(os.sep, "/")}')
        for name in expected_files
    ]
    local = _local_records(pairs)
    client = _api(token, api)
    client.upload_folder(
        folder_path=local_folder, path_in_repo=remote_folder, repo_id=repository, repo_type='dataset',
        # Escaped, because the patterns are matched as globs and a file name may hold [ or *.
        allow_patterns=[glob.escape(name.replace(os.sep, '/')) for name in expected_files],
        commit_message=f'Upload {remote_folder}',
    )
    entries = client.list_repo_tree(repository, path_in_repo=remote_folder, recursive=True, repo_type='dataset')
    problems = _mismatches(local, entries)
    if problems:
        raise UploadMismatch(f'{len(problems)} files differ after uploading {remote_folder}: ' + '; '.join(problems[:5]))
    return {path: record['sha256'] for path, record in local.items()}
