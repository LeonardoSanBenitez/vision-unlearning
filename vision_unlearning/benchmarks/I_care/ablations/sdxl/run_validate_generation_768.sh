#!/usr/bin/env bash
# Drives validate_generation_768.py to completion on a machine whose free memory is not ours.
#
# Two facts this wrapper exists for, both measured on 2026-08-18:
#   * building the pipeline transiently costs about 4.5 GB of system memory, and the job's watchdog
#     aborts below 1.5 GB free, so launching under roughly 6 GB free is launching into a wall;
#   * the WSL2 virtual machine re-inflates without warning and has killed a run MID-GENERATION, not
#     only during the load. No launch condition can prevent that one.
#
# So: wait for headroom, run, and if the marker is not in the log, wait and run again. The script
# skips any image already on disk, so each attempt costs only the image that was in flight.
#
#   bash run_validate_generation_768.sh [required_free_gb] [max_attempts] [wait_ticks] [tick_seconds]
set -u

REQUIRED_GB="${1:-6.0}"
MAX_ATTEMPTS="${2:-8}"
WAIT_TICKS="${3:-60}"
TICK_S="${4:-30}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"
LOG="assets/validate_generation_768.log"
MARKER="VALIDATE_GENERATION_768_DONE"

cd "$HERE" || exit 3

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

    echo "attempt $attempt: starting validate_generation_768.py"
    PYTHONPATH="$REPO_ROOT" HF_HUB_DISABLE_XET=1 "$PY" validate_generation_768.py \
        > "assets/validate_generation_768_attempt${attempt}.log" 2>&1
    STATUS=$?
    cp "assets/validate_generation_768_attempt${attempt}.log" "$LOG"
    echo "attempt $attempt: exited with status $STATUS"

    IMAGES=$(ls assets/validate_generation_768/*.png 2>/dev/null | wc -l)
    echo "attempt $attempt: $IMAGES images on disk"

    if grep -q "$MARKER" "$LOG"; then
        echo "MARKER FOUND after attempt $attempt"
        exit 0
    fi
    echo "no marker yet; waiting before the next attempt"
    sleep "$TICK_S"
done

echo "GAVE UP: $MAX_ATTEMPTS attempts without the marker."
exit 5
