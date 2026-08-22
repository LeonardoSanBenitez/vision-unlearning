#!/usr/bin/env bash
# S6 second half: generate every checkpoint the manifest does not yet hold, one process per epoch.
#
# It calls run_campaign.py's own `remaining` selector rather than a list written here, so the manifest
# decides what is next and a re-launch after an abort picks up exactly where it stopped. The loop bound
# is the number of checkpoints, which is a safety stop, not the termination condition: the stage prints
# `remaining_after_this=0` on the last epoch and this exits on seeing it.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

for _ in $(seq 1 13); do
    echo "=== s6 remaining: starting a stage at $(date +%H:%M:%S)"
    bash "$HERE/run_campaign_stage.sh" generate 42 remaining
    STATUS=$?
    echo "=== s6 remaining: stage exited with status ${STATUS} at $(date +%H:%M:%S)"
    if [ "$STATUS" -ne 0 ]; then
        echo "S6_REMAINING_FAILED status=${STATUS}"
        exit "$STATUS"
    fi
    if grep -q "remaining_after_this=0" "$HERE/assets/campaign_generate_seed42_remaining_attempt1.log" 2>/dev/null; then
        echo "S6_REMAINING_ALL_EPOCHS_FINISHED"
        exit 0
    fi
done

echo "S6_REMAINING_LOOP_BOUND_REACHED"
