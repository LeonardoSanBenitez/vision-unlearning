#!/usr/bin/env bash
# S6 early half: generate the seed-42 campaign images at checkpoints 1, 3 and 5, one process per
# epoch, each through run_campaign_stage.sh so that the headroom wait, the retry and the per-attempt
# logs apply to every one of them. Stops after epoch 5 -- the plan's sign-off gate sits there.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

for EPOCH in 1 3 5; do
    echo "=== s6 early: starting epoch ${EPOCH} at $(date +%H:%M:%S)"
    bash "$HERE/run_campaign_stage.sh" generate 42 "$EPOCH"
    STATUS=$?
    echo "=== s6 early: epoch ${EPOCH} stage exited with status ${STATUS} at $(date +%H:%M:%S)"
    if [ "$STATUS" -ne 0 ]; then
        echo "S6_EARLY_FAILED epoch=${EPOCH} status=${STATUS}"
        exit "$STATUS"
    fi
done

echo "S6_EARLY_ALL_EPOCHS_FINISHED"
