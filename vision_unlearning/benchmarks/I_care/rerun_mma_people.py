"""Targeted MMA re-run for people/munba and people/uce.

Context: During the Phase 2 pipeline_08 run, HuggingFace was unreachable (DNS failure)
and the HF token had a CRLF corruption (\r in token), causing most people/munba and
people/uce MMA pairs to fail silently.  people/distil is also partially missing (1039/1275).

Strategy: compute all missing local files WITHOUT individual HF uploads (fast), then
batch-upload the whole MMA directory in one shot via upload_folder (much faster than
1254 × individual file uploads each taking ~16s = 5+ hours).

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
Requirements: HF_TOKEN env var set (no \\r character).
"""
import logging
import os
import sys

# Ensure the repo root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('rerun_mma')

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402, F401

_MMA_LOCAL_DIR = os.path.join(
    os.path.dirname(__file__),
    'assets', 'results', 'MetricMetricAlignment',
)
_HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
_HF_MMA_PATH = "results/MetricMetricAlignment"


def _batch_upload(hf_token: str) -> None:
    """Upload people MMA files to a task subdirectory on HF.

    The flat results/MetricMetricAlignment/ directory already holds ~8,731
    breeds+scenes files from Phase 2a.  Adding all 3,846 people files would
    push the total to ~12,577, exceeding HuggingFace's 10,000-files-per-
    directory limit.  Fix: upload people files to the task subdirectory
    results/MetricMetricAlignment/people/ (3,846 files, safely under 10k).
    Breeds and scenes remain in the flat directory; the subdirectory is
    self-consistent.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    people_hf_path = _HF_MMA_PATH + "/people"
    logger.info(
        "Batch-uploading people MMA files → %s/%s (allow_patterns=*_people_*) ...",
        _HF_REPO, people_hf_path,
    )
    api.upload_folder(
        repo_id=_HF_REPO,
        repo_type="dataset",
        folder_path=_MMA_LOCAL_DIR,
        path_in_repo=people_hf_path,
        allow_patterns=["*_people_*"],
        commit_message="MMA people rerun: upload people results to task subdirectory",
    )
    logger.info("Batch upload complete.")


def main() -> None:
    hf_token = os.getenv('HF_TOKEN', '').strip()
    if not hf_token:
        logger.error("HF_TOKEN not set. Set it before running.")
        sys.exit(1)
    if '\r' in hf_token:
        logger.error("HF_TOKEN contains \\r — strip CRLF from .env first.")
        sys.exit(1)

    # Compute phase is DONE (people/munba=1282, people/uce=1282, people/distil=1282
    # all computed locally in previous run).  Skip straight to upload.
    for method in ['munba', 'uce', 'distil']:
        count = len([
            f for f in os.listdir(_MMA_LOCAL_DIR)
            if f'_people_{method}_' in f
        ])
        logger.info("Local people/%s count: %d", method, count)

    # Batch-upload people files to task subdirectory to stay under HF 10k limit.
    _batch_upload(hf_token)


if __name__ == '__main__':
    main()
