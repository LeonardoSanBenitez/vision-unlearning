"""Rehearsal: the licences, the upload guard, the content check and a tag, on the campaign repository.

Steps 12 and 13 of the rehearsal, the only part of it that writes to the Hub:

1. **guard** -- an upload aimed at the first corpus is refused before any network call (a client
   that records calls receives none);
2. **licences** -- the corpus card (``corpus_card/README.md``: the repository's licence paragraph)
   becomes the repository's ``README.md``, and ``models/LICENSE`` receives the CreativeML OpenRAIL-M
   text, downloaded from its source (the ``CompVis/stable-diffusion-license`` Space at
   ``--licence-revision``); both are uploaded through the guarded, content-checked upload and stay;
3. **content check** -- a probe file is uploaded to ``--scratch-path``, re-checked against the
   repository (no mismatch), one byte of the local copy is flipped and the re-check must report it;
   the probe is then deleted from the repository and its absence confirmed;
4. **tag** -- a tag of ``--scratch-tag`` is created on the repository, seen in its refs, deleted, and
   its absence confirmed.

Needs ``HF_TOKEN`` in the environment with write access to the campaign repository. Usage::

    python upload_rehearsal.py --probe IMAGE.png --work DIR --output RESULT.json \\
        --scratch-path rehearsal_scratch/s2/probe.png --scratch-tag rehearsal-s2-scratch

Prints ``UPLOADS <verdicts>`` and ``UPLOADS_DONE``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List

from vision_unlearning.benchmarks.I_care.upload import (
    CAMPAIGN_REPOSITORY,
    UploadRefused,
    remote_mismatches,
    upload_file,
)

OLD_CORPUS = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'
LICENCE_SPACE = 'CompVis/stable-diffusion-license'
CARD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'corpus_card', 'README.md')


class _RecordingClient:
    """Stands in for the Hub client; records every call it receives."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append(name)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe', required=True, help='any small local file to upload as the scratch probe')
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scratch-path', required=True, help='where in the repository the probe goes; deleted after')
    parser.add_argument('--scratch-tag', required=True, help='the tag created and deleted')
    parser.add_argument('--licence-revision', required=True, help=f'the commit of the {LICENCE_SPACE} Space to read the licence from')
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ['HF_TOKEN']
    api = HfApi(token=token)
    os.makedirs(args.work, exist_ok=True)
    result: Dict[str, Any] = {'repository': CAMPAIGN_REPOSITORY}

    # 1. the guard
    recorder = _RecordingClient()
    try:
        upload_file(args.probe, args.scratch_path, OLD_CORPUS, token, api=recorder)
        result['guard'] = 'NOT REFUSED'
    except UploadRefused as exc:
        result['guard'] = f'refused with {len(recorder.calls)} network calls: {exc}'

    # 2. the licences
    licence = hf_hub_download(LICENCE_SPACE, 'license.txt', repo_type='space', revision=args.licence_revision,
                              token=token, local_dir=os.path.join(args.work, 'licence'))
    result['readme'] = upload_file(CARD, 'README.md', CAMPAIGN_REPOSITORY, token)
    result['models_licence'] = upload_file(licence, 'models/LICENSE', CAMPAIGN_REPOSITORY, token)
    result['licence_source'] = f'{LICENCE_SPACE}@{args.licence_revision}/license.txt'

    # 3. the content check, on a scratch probe
    probe = os.path.join(args.work, os.path.basename(args.scratch_path))
    shutil.copyfile(args.probe, probe)
    result['probe_upload'] = upload_file(probe, args.scratch_path, CAMPAIGN_REPOSITORY, token)
    result['probe_recheck_intact'] = remote_mismatches([(probe, args.scratch_path)], CAMPAIGN_REPOSITORY, token)
    data = bytearray(open(probe, 'rb').read())
    data[len(data) // 2] ^= 0x01
    with open(probe, 'wb') as handle:
        handle.write(bytes(data))
    result['probe_recheck_flipped'] = remote_mismatches([(probe, args.scratch_path)], CAMPAIGN_REPOSITORY, token)
    api.delete_file(args.scratch_path, CAMPAIGN_REPOSITORY, repo_type='dataset', commit_message=f'Delete {args.scratch_path}')
    result['probe_deleted'] = not api.file_exists(CAMPAIGN_REPOSITORY, args.scratch_path, repo_type='dataset')

    # 4. a tag, created and deleted
    api.create_tag(CAMPAIGN_REPOSITORY, tag=args.scratch_tag, repo_type='dataset')
    tags = [tag.name for tag in api.list_repo_refs(CAMPAIGN_REPOSITORY, repo_type='dataset').tags]
    result['tag_created'] = args.scratch_tag in tags
    api.delete_tag(CAMPAIGN_REPOSITORY, tag=args.scratch_tag, repo_type='dataset')
    tags = [tag.name for tag in api.list_repo_refs(CAMPAIGN_REPOSITORY, repo_type='dataset').tags]
    result['tag_deleted'] = args.scratch_tag not in tags

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print('UPLOADS ' + ' '.join([
        f"guard={result['guard'][:20]}", f"intact={len(result['probe_recheck_intact'])}",
        f"flipped={len(result['probe_recheck_flipped'])}", f"deleted={result['probe_deleted']}",
        f"tag={result['tag_created']}/{result['tag_deleted']}",
    ]))
    print('UPLOADS_DONE')


if __name__ == '__main__':
    main()
