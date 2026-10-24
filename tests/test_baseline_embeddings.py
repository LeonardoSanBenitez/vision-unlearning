"""``pipeline_05 --baseline``: the task's baseline is embedded on its own, before any session exists.

Fails on the code before this change: ``run_baseline_embeddings`` does not exist. ``pipeline_05``
embedded the baseline only as a side effect of a session's first entity, and skipped that entity, the
baseline included, when the entity's own image folder was absent -- which it always is before the
first session has run.
"""
import json
import os
from typing import Any, Dict, List, Literal

import pytest

from vision_unlearning.benchmarks.I_care import pipeline_05_compute_embeddings as p05
from vision_unlearning.benchmarks.I_care.metadata import BaselineEmbeddings
from vision_unlearning.datasets.entity_names import generation_prompt
from vision_unlearning.datasets.testbed import get_generated_dataset_file, get_shared_baseline_folder

TASK: Literal['scenes'] = 'scenes'
NAMES = ['abbey', 'badlands']
SEEDS = [42, 43]


def _prompts() -> List[str]:
    return [generation_prompt(TASK, name) for name in NAMES]  # type: ignore[arg-type]


def _baseline(base: str, drop: int = 0) -> None:
    folder = get_shared_baseline_folder(TASK, base)
    os.makedirs(folder)
    files = [get_generated_dataset_file('off', s, p) for s in SEEDS for p in _prompts()]
    for name in files[drop:]:
        with open(os.path.join(folder, name), 'wb') as handle:
            handle.write(name.encode())


def _embed(path: str) -> List[float]:
    return [float(os.path.getsize(path)), 1.0]


def _run(base: str, uploads: List[Any]) -> str:
    progress: Dict[str, str] = {}
    return p05.run_baseline_embeddings(
        task=TASK, generate_dataset_seeds=SEEDS, prompts=_prompts(),
        metadata_filtered=[{'name': n} for n in NAMES], base_folder=base, replace_if_exists=False,
        upload_to_hf=True, hf_token='t', progress=progress,
        progress_path=os.path.join(base, 'progress.json'), embed_image_fn=_embed,
        hf_file_exists_fn=lambda *a: False, hf_upload_fn=lambda *a: uploads.append(a),
    )


def test_the_baseline_is_embedded_without_any_session(tmp_path: Any) -> None:
    base = str(tmp_path)
    _baseline(base)
    uploads: List[Any] = []
    path = _run(base, uploads)
    assert path == BaselineEmbeddings(task=TASK, base_folder=base)._get_data_path_local()  # type: ignore[arg-type]
    with open(path, encoding='utf-8') as handle:
        written = json.load(handle)
    assert len(written['embeddings']) == len(SEEDS) * len(NAMES)
    assert {(r['prompt'], r['seed']) for r in written['embeddings']} == {(p, s) for p in _prompts() for s in SEEDS}
    meta = written['metadata']
    assert (meta['forgotten_entity'], meta['lora_state'], meta['method'], meta['num_train_epochs']) == ('original', 'off', None, None)
    assert [a[1] for a in uploads] == [BaselineEmbeddings(task=TASK)._get_data_path_remote()]  # type: ignore[arg-type]


def test_an_incomplete_baseline_raises_and_writes_nothing(tmp_path: Any) -> None:
    base = str(tmp_path)
    _baseline(base, drop=1)
    with pytest.raises(FileNotFoundError, match='does not hold all 4 baseline images'):
        _run(base, [])
    assert not os.path.exists(BaselineEmbeddings(task=TASK, base_folder=base)._get_data_path_local())  # type: ignore[arg-type]
