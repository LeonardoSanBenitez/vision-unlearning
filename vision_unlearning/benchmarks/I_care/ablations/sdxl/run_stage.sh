#!/usr/bin/env bash
# Drives ONE stage of ONE ablation script to completion on a machine whose free memory is not ours.
#
# This is the single implementation of the launch discipline every long stage of this ablation needs;
# run_campaign_stage.sh and run_s4_schedule_probe.sh both go through it. Two measured facts shape it:
#
#   * a generation stage costs about 1.3 GB of system memory from launch to its trough, and the job's
#     watchdog aborts below 1.5 GB free, so a launch gate of 4.5 GB leaves roughly 2.7 GB of margin.
#     MEASURED 2026-08-22 on the ten control off-baselines: entered at 5.16 GB free, monitor trough
#     `min_ram_free_gb` 4.675, pipeline built in about 23 s and 41 s per image after it. The 6.0 GB
#     gate this file used to carry came from a 3.13 GB drawdown recorded on 2026-08-19
#     (campaign_generate_seed42_epoch3_monitor.log, 8.01 GB free entering the load, 4.88 GB at the
#     trough) -- but that day's guard log records four Docker Desktop relaunches, so most of that
#     drawdown was the WSL virtual machine re-inflating underneath the run, not the pipeline. The
#     over-tight gate then blocked the campaign for 45 minutes on a machine that could have run it;
#   * the WSL2 virtual machine re-inflates without warning and has killed a run MID-GENERATION. No
#     launch condition prevents that one -- keep_docker_stopped.sh, following the driver's PID, does.
#
# So: wait for headroom, run, and if the completion marker is not in the log, wait and run again.
# Every script driven this way skips work already on disk, so an attempt costs at most the image or
# the step that was in flight.
#
#   bash run_stage.sh <log_name> <completion_marker> <script.py> [script arguments...]
#
# For example:
#   bash run_stage.sh campaign_generate_seed42_off CAMPAIGN_GENERATE_DONE run_campaign.py \
#        --stage generate --seed 42 --epochs off
#
# Tunables, as environment variables: REQUIRED_GB (default 4.5), MAX_ATTEMPTS (8), WAIT_TICKS (60),
# TICK_S (30).
#
# Logs: assets/<log_name>_attemptN.log, with the last attempt copied over assets/<log_name>.log.
# Exit codes: 0 marker found, 3 bad directory, 4 never got the memory headroom, 5 attempts exhausted.
set -u

NAME="${1:?log name, e.g. campaign_generate_seed42_off}"
MARKER="${2:?completion marker printed by the script on success}"
SCRIPT="${3:?python script to run, e.g. run_campaign.py}"
shift 3
ARGS=("$@")

REQUIRED_GB="${REQUIRED_GB:-4.5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
WAIT_TICKS="${WAIT_TICKS:-60}"
TICK_S="${TICK_S:-30}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"

cd "$HERE" || exit 3
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

    echo "attempt $attempt: starting $SCRIPT ${ARGS[*]}"
    PYTHONPATH="$REPO_ROOT" HF_HUB_DISABLE_XET=1 "$PY" "$SCRIPT" "${ARGS[@]}" \
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
