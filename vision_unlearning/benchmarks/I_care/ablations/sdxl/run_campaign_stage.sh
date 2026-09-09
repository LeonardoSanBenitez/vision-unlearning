#!/usr/bin/env bash
# Drives ONE run_campaign.py stage to completion. The launch discipline itself -- wait for memory
# headroom, run, retry until the completion marker appears, keep every attempt's log -- lives in
# run_stage.sh, which this only supplies the stage's arguments to.
#
#   bash run_campaign_stage.sh generate 42 off
#   bash run_campaign_stage.sh generate 42 3
#   bash run_campaign_stage.sh generate 42 remaining
#   bash run_campaign_stage.sh train 42 -
#
# The third argument is passed straight to --epochs; for --stage train it is ignored and any
# placeholder (`-`) will do. run_stage.sh's tunables (REQUIRED_GB, MAX_ATTEMPTS, WAIT_TICKS, TICK_S)
# are environment variables and pass through unchanged.
set -u

STAGE="${1:?stage: train | generate}"
SEED="${2:?seed: 42 | 43}"
EPOCHS="${3:?epochs selector: off | all | remaining | <int> | - for --stage train}"

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$STAGE" = "train" ]; then
    exec bash "$HERE/run_stage.sh" "campaign_train_seed${SEED}" CAMPAIGN_TRAIN_DONE run_campaign.py \
        --stage train --seed "$SEED"
fi

exec bash "$HERE/run_stage.sh" "campaign_generate_seed${SEED}_${EPOCHS}" CAMPAIGN_GENERATE_DONE run_campaign.py \
    --stage generate --seed "$SEED" --epochs "$EPOCHS"
