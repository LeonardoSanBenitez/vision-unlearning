#!/usr/bin/env bash
# Phase 2b: Batch upload all computed RT results to HuggingFace.
#
# Run AFTER launch_phase2.sh (Phase 2a) has completed.
# Uses HfApi.upload_folder() to make a single git commit to HF per RT directory,
# instead of one commit per file (which would take 50+ hours).
#
# Run from: zoo/unlearning/vision-unlearning/
# Logs to:  logs/phase2b_upload.log

set -euo pipefail

LOG_FILE="$(pwd)/logs/phase2b_upload.log"
echo "[$(date)] Phase 2b batch upload start" | tee -a "$LOG_FILE"

# Load credentials
source /home/leonardo/Desktop/zoo/.claude/memory/priya/.env
HF_TOKEN_CLEAN=$(echo -n "$HF_TOKEN" | tr -d '\r')
export HF_TOKEN="$HF_TOKEN_CLEAN"

echo "[$(date)] HF_TOKEN length: ${#HF_TOKEN_CLEAN}" | tee -a "$LOG_FILE"

RESULTS_DIR="vision_unlearning/benchmarks/I_care/assets/results"
ASSETS_DIR="vision_unlearning/benchmarks/I_care/assets"
HF_REPO="LeonardoBenitez/VisionUnlearningEvaluationTestbeds"

echo "[$(date)] Counting local result files..." | tee -a "$LOG_FILE"
TOTAL_FILES=$(find "$RESULTS_DIR" -name "*.json" 2>/dev/null | wc -l)
echo "[$(date)] Total result files to upload: $TOTAL_FILES" | tee -a "$LOG_FILE"

# Run inside Docker container for upload_folder
docker compose exec \
  -e HF_TOKEN="$HF_TOKEN_CLEAN" \
  notebooks \
  sh -c "
    set -e
    cd /src/vision_unlearning/benchmarks/I_care
    echo '[$(date)] Starting batch HF upload from assets/results/'
    poetry run python -c \"
import os
from huggingface_hub import HfApi

api = HfApi()
token = os.getenv('HF_TOKEN')
repo_id = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'

# Upload the interference_per_entity files (root level)
for task in ['breeds', 'scenes', 'people']:
    local_path = f'assets/interference_per_entity_{task}.json'
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f'interference_per_entity_{task}.json',
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
        )
        print(f'Uploaded: interference_per_entity_{task}.json', flush=True)

# Upload all RT results in batch (one commit per RT subfolder)
results_dir = 'assets/results'
for rt_dir in sorted(os.listdir(results_dir)):
    rt_path = os.path.join(results_dir, rt_dir)
    if not os.path.isdir(rt_path):
        continue
    json_files = [f for f in os.listdir(rt_path) if f.endswith('.json')]
    if not json_files:
        print(f'Skipping {rt_dir}: no json files', flush=True)
        continue
    print(f'Uploading {rt_dir}: {len(json_files)} files...', flush=True)
    api.upload_folder(
        folder_path=rt_path,
        path_in_repo=f'results/{rt_dir}',
        repo_id=repo_id,
        repo_type='dataset',
        token=token,
        commit_message=f'Phase 2: upload {rt_dir} ({len(json_files)} RT results)',
        ignore_patterns=['*.png', '*.pdf', '*.csv'],
    )
    print(f'Done: {rt_dir}', flush=True)

print('All uploads complete.', flush=True)
\"
    echo '[$(date)] Batch upload complete'
  " 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Phase 2b complete" | tee -a "$LOG_FILE"
