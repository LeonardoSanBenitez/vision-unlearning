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
from typing import FrozenSet

# Ensure the repo root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('rerun_mma')

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402
from pipeline_08_run_all_rts import (  # noqa: E402
    run_metric_metric_alignment,
)

_MMA_LOCAL_DIR = os.path.join(
    os.path.dirname(__file__),
    'assets', 'results', 'MetricMetricAlignment',
)
_HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
_HF_MMA_PATH = "results/MetricMetricAlignment"


def _batch_upload(hf_token: str) -> None:
    """Upload the full local MMA directory to HF in one git commit."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    logger.info("Batch-uploading %s → %s/%s ...", _MMA_LOCAL_DIR, _HF_REPO, _HF_MMA_PATH)
    api.upload_folder(
        repo_id=_HF_REPO,
        repo_type="dataset",
        folder_path=_MMA_LOCAL_DIR,
        path_in_repo=_HF_MMA_PATH,
        commit_message="MMA people rerun: upload computed results",
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

    # Step 1: Compute all missing local MMA files WITHOUT individual HF uploads.
    # Passing hf_files=frozenset() and upload_if_recomputed=False means:
    #   - _should_skip returns True for local-existing files (no upload).
    #   - _should_skip returns False for missing files (compute, no upload).
    # This makes computation fast (no per-file HTTP round-trips).
    hf_files: FrozenSet[str] = frozenset()
    targets = [
        ('people', 'munba'),   # 21/1275 local — critical
        ('people', 'uce'),     # 21/1275 local — critical
        ('people', 'distil'),  # ~1039/1275 local — partial
    ]
    for task, method in targets:
        logger.info("Computing MMA for %s/%s (no upload) ...", task, method)
        run_metric_metric_alignment(
            tasks=[task],
            methods=[method],
            hf_files=hf_files,
            upload_if_recomputed=False,
        )
        count = len([
            f for f in os.listdir(_MMA_LOCAL_DIR)
            if f'_{task}_{method}_' in f
        ])
        logger.info("Done computing %s/%s — local files now: %d", task, method, count)

    # Step 2: Batch-upload the full MMA directory.
    _batch_upload(hf_token)


if __name__ == '__main__':
    main()
