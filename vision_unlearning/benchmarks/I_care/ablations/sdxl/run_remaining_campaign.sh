#!/usr/bin/env bash
# Orchestration only -- calls the already-tested run_campaign.py repeatedly, each invocation its own
# process (S2's one-pipeline-per-process constraint; run_campaign.py itself refuses a comma-separated
# --epochs list for the same reason). No new logic lives here.
#
# Finishes S5 (remaining seed-42 checkpoints) then runs S6 in full (seed 43 train + all generation).
#
# Usage (from this directory):  bash run_remaining_campaign.sh
set -u

PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"

echo "=== S5: seed 42 remaining checkpoints ==="
for epoch in 2 10 15 20 30 50 75 100 150 200; do
    echo "--- seed 42, epoch ${epoch} : $(date +%H:%M:%S) ---"
    "$PY" run_campaign.py --stage generate --seed 42 --epochs "${epoch}"
done

echo "=== S6: seed 43 training ==="
echo "--- seed 43 train : $(date +%H:%M:%S) ---"
"$PY" run_campaign.py --stage train --seed 43

echo "=== S6: seed 43 generation (off + all 13 checkpoints) ==="
echo "--- seed 43, off : $(date +%H:%M:%S) ---"
"$PY" run_campaign.py --stage generate --seed 43 --epochs off
for epoch in 1 2 3 5 10 15 20 30 50 75 100 150 200; do
    echo "--- seed 43, epoch ${epoch} : $(date +%H:%M:%S) ---"
    "$PY" run_campaign.py --stage generate --seed 43 --epochs "${epoch}"
done

echo "=== RUN_REMAINING_CAMPAIGN_DONE ==="
