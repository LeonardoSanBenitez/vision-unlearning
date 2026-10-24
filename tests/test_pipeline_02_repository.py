"""``pipeline_02``'s similarity matrices read from, and upload to, the repository it is given.

Fails on the code before this change: ``pipeline_02`` had no ``--repository``, so ``--upload`` wrote
through the shared artifact cascade to the library's default repository -- the first corpus, which is
frozen -- and a matrix absent locally was read from that corpus too. Every other uploading I-CARE
script already checked its target at start-up.
"""
import sys
from typing import Any, Dict, List

import pytest

import vision_unlearning.benchmarks.I_care as vb
from vision_unlearning.benchmarks.I_care import pipeline_02_compute_similarities as p02
from vision_unlearning.benchmarks.I_care.upload import CAMPAIGN_REPOSITORY, UploadRefused


def test_an_upload_to_the_default_repository_is_refused_before_any_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    computed: List[str] = []
    monkeypatch.setattr(p02, 'run_clip', lambda *a, **k: computed.append('clip'))
    monkeypatch.setattr(sys, 'argv', ['pipeline_02', '--task', 'people', '--similarity', 'clip', '--upload'])
    with pytest.raises(UploadRefused):
        p02.main()
    assert computed == []


def test_the_matrix_is_resolved_against_the_given_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[Dict[str, Any]] = []

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

        def compute(self) -> None:
            return None

    monkeypatch.setattr(vb, 'Similarity', Recorder)
    monkeypatch.setattr(vb, 'ResultTemplateSimilarityMatrix', Recorder)
    monkeypatch.setattr(sys, 'argv', ['pipeline_02', '--task', 'people', '--similarity', 'jacc', '--upload',
                                      '--repository', CAMPAIGN_REPOSITORY, '--base-folder', 'b'])
    p02.main()
    assert [k['remote_repository_name'] for k in seen] == [CAMPAIGN_REPOSITORY, CAMPAIGN_REPOSITORY]
    assert all(k['similarity_metric'] == 'jacc' and k['base_folder'] == 'b' for k in seen)
