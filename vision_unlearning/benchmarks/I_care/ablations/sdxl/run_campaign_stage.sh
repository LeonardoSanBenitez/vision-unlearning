#!/usr/bin/env bash
# Drives ONE run_campaign.py stage to completion on a machine whose free memory is not ours.
#
# Same two measured facts as run_validate_generation_768.sh, which this is modelled on:
#   * building the Stable Diffusion XL pipeline transiently costs about 4.5 GB of system memory and
#     the job's watchdog aborts below 1.5 GB free, so launching under roughly 6 GB free is launching
#     into a wall;
#   * the WSL2 virtual machine re-inflates without warning and has killed a run MID-GENERATION. No
#     launch condition prevents that one.
#
# So: wait for headroom, run, and if the completion marker is not in the log, wait and run again.
# run_campaign.py skips images already on disk and writes its manifest rows only once all ten exist,
# so each attempt costs at most the image that was in flight.
#
#   bash run_campaign_stage.sh generate 42 off [required_free_gb] [max_attempts] [wait_ticks] [tick_seconds]
#   bash run_campaign_stage.sh generate 42 3
#   bash run_campaign_stage.sh generate 42 remaining
#   bash run_campaign_stage.sh train 42 -
#
# The third argument is passed straight to --epochs; for --stage train it is ignored and any
# placeholder (`-`) will do.
#
# Logs: assets/campaign_<stage>_seed<seed>[_<epochs>]_attemptN.log, with the last attempt copied over
# assets/campaign_<stage>_seed<seed>[_<epochs>].log.
set -u

STAGE="${1:?stage: train | generate}"
SEED="${2:?seed: 42 | 43}"
EPOCHS="${3:?epochs selector: off | all | remaining | <int> | - for --stage train}"
REQUIRED_GB="${4:-6.0}"
MAX_ATTEMPTS="${5:-8}"
WAIT_TICKS="${6:-60}"
TICK_S="${7:-30}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"

cd "$HERE" || exit 3

if [ "$STAGE" = "train" ]; then
    ARGS=(--stage train --seed "$SEED")
    NAME="campaign_train_seed${SEED}"
    MARKER="CAMPAIGN_TRAIN_DONE"
else
    ARGS=(--stage generate --seed "$SEED" --epochs "$EPOCHS")
    NAME="campaign_generate_seed${SEED}_${EPOCHS}"
    MARKER="CAMPAIGN_GENERATE_DONE"
fi
LOG="assets/${NAME}.log"

free_gb () {
    "$PY" -c "import psutil; print(round(psutil.virtual_memory().available / 1024**3, 2))"
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    READY=0
    for tick in $(seq 1 "$WAIT_TICKS"); do
        FREE="$(free_gb)"
        READY="$("$PY" -c "print(1 if float('$FREE') >= float('$REQUIRED_GB') else 0)")"
        echo "attempt $attempt, tick $tick: free system memory ${FREE} GB, required ${REQUIRED_GB} GB, ready=${READY}"
        [ "$READY" = "1" ] && break
        sleep "$TICK_S"
    done
    if [ "$READY" != "1" ]; then
        echo "GAVE UP: free system memory never reached ${REQUIRED_GB} GB."
        exit 4
    fi

    echo "attempt $attempt: starting run_campaign.py ${ARGS[*]}"
    PYTHONPATH="$REPO_ROOT" HF_HUB_DISABLE_XET=1 "$PY" run_campaign.py "${ARGS[@]}" \
        > "assets/${NAME}_attempt${attempt}.log" 2>&1
    STATUS=$?
    cp "assets/${NAME}_attempt${attempt}.log" "$LOG"
    echo "attempt $attempt: exited with status $STATUS"

    if grep -q "$MARKER" "$LOG"; then
        echo "MARKER FOUND after attempt $attempt"
        exit 0
    fi
    echo "no marker yet; waiting before the next attempt"
    sleep "$TICK_S"
done

echo "GAVE UP: $MAX_ATTEMPTS attempts without the marker."
exit 5
