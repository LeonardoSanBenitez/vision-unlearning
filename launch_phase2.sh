#!/usr/bin/env bash
# Phase 2 launcher: compute all RT combinations locally (no per-file upload).
#
# Two-phase approach to avoid the 15-second-per-file HF upload bottleneck:
#   Phase 2a: compute all RT JSONs locally (fast, CPU-only)
#   Phase 2b: batch-upload the results/ folder in one go (run upload_phase2b.sh after)
#
# Run from: zoo/unlearning/vision-unlearning/
# Logs to:  logs/phase2_pipeline08.log
#
# HF_TOKEN must be set in .claude/memory/priya/.env — loaded below.
# The token is stripped of Windows carriage-return characters before use.

set -euo pipefail

LOG_FILE="$(pwd)/logs/phase2_pipeline08.log"
echo "[$(date)] Phase 2a start (compute-only, no per-file uploads)" | tee -a "$LOG_FILE"

# Load credentials
source /home/leonardo/Desktop/zoo/.claude/memory/priya/.env
HF_TOKEN_CLEAN=$(echo -n "$HF_TOKEN" | tr -d '\r')
export HF_TOKEN="$HF_TOKEN_CLEAN"

echo "[$(date)] HF_TOKEN length: ${#HF_TOKEN_CLEAN}" | tee -a "$LOG_FILE"

# Run inside Docker container — compute only, no uploads
# (--upload-if-recomputed is intentionally OMITTED here;
#  use upload_phase2b.sh for batch upload after this completes)
docker compose exec \
  -e HF_TOKEN="$HF_TOKEN_CLEAN" \
  notebooks \
  sh -c "
    set -e
    cd /src/vision_unlearning/benchmarks/I_care
    echo '[$(date)] Starting pipeline_08 inside container (compute-only)'
    poetry run python pipeline_08_run_all_rts.py \
      --action run \
      --rts all \
      --tasks breeds scenes people \
      --methods distil munba uce \
      2>&1
    echo '[$(date)] pipeline_08 compute-only complete'
  " 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Phase 2a complete. Run upload_phase2b.sh to batch-upload results to HF." | tee -a "$LOG_FILE"
