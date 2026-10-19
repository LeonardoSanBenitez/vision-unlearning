"""``provenance.json``: what produced a generated folder or a trained model, recorded beside it.

Every generated-image folder and every model folder of the benchmark carries one. A generated
folder's record is written by `vision_unlearning.utils.data_generation.generate_dataset` itself (its
``provenance`` argument), because only that call knows what it ran: the seeds, the ordered prompt
list and its hash, the batch size, the noise convention, and the ``pairing`` object of every image's
initial-latent hash, the scheduler and the effective arguments. The producer adds what it knows,
built here by :func:`producer_facts`: the library commit, the card model and the precision, and the
sha256 of the run's ``pins.json`` and environment freeze.

Readers go through :func:`read_provenance`, which raises :class:`ProvenanceMismatch` when the folder
was produced from a different prompt list -- including a reordering of the same prompts -- or a
different seed list than the one the reader is about to use.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional, Sequence

PROVENANCE_FILENAME = 'provenance.json'


class ProvenanceMismatch(ValueError):
    """A folder's recorded inputs differ from the ones its reader is about to assume."""


def prompt_list_sha256(prompts: Sequence[str]) -> str:
    """The hash of an ordered prompt list, as `generate_dataset` records it."""
    return hashlib.sha256(json.dumps(list(prompts)).encode('utf-8')).hexdigest()


def file_sha256(path: Optional[str]) -> Optional[str]:
    """The sha256 of a file, or None when no path is given."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def library_commit() -> Optional[str]:
    """The git commit of the checkout this library is imported from, or None outside a checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def library_tree_is_clean() -> Optional[bool]:
    """Whether the library checkout has no uncommitted change to tracked files, or None outside one."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() == '' if result.returncode == 0 else None


def card_model() -> str:
    """The name of the graphics card in use, or ``'cpu'``."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def producer_facts(
    precision: str,
    pins_path: Optional[str] = None,
    freeze_path: Optional[str] = None,
    card: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """What a producer knows about its own run, for the ``provenance`` of what it writes.

    ``card`` defaults to the card in use (:func:`card_model`). ``extra`` adds fields of the
    producer's own -- the session's task, method, entity and training configuration, for a model.
    """
    return {
        'commit': library_commit(),
        'tree_clean': library_tree_is_clean(),
        'card_model': card if card is not None else card_model(),
        'precision': precision,
        'pins_sha256': file_sha256(pins_path),
        'freeze_sha256': file_sha256(freeze_path),
        **extra,
    }


def write_provenance(folder: str, record: Dict[str, Any]) -> str:
    """Write a record as the folder's ``provenance.json`` (for folders not written by generate_dataset)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, PROVENANCE_FILENAME)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=1)
    return path


def read_provenance(folder: str, prompts: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Read a folder's ``provenance.json``, refusing it when its prompts or seeds are not these.

    The prompt list is compared as an ordered sequence, by its hash, so a reordering of the same
    prompts is refused too: under the batching every producer uses, the order decides which
    prompts shared a batch. A folder with no record raises FileNotFoundError.
    """
    path = os.path.join(folder, PROVENANCE_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} does not exist: this folder records nothing about what produced it')
    with open(path, encoding='utf-8') as handle:
        record: Dict[str, Any] = json.load(handle)
    if prompts is not None and record.get('prompt_list_sha256') != prompt_list_sha256(prompts):
        recorded = record.get('prompts_ordered') or []
        same_set = sorted(recorded) == sorted(prompts)
        raise ProvenanceMismatch(
            f'{path} was produced from {"a reordering of" if same_set else "another"} prompt list '
            f'({len(recorded)} prompts) than the {len(prompts)} this reader uses'
        )
    if seeds is not None and list(record.get('seeds') or []) != list(seeds):
        raise ProvenanceMismatch(f'{path} records seeds {record.get("seeds")}, this reader uses {list(seeds)}')
    return record
