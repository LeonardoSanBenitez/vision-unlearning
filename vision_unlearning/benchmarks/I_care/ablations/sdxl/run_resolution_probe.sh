#!/usr/bin/env bash
# Waits for enough free system memory, then runs resolution_probe.py.
#
# Why the wait exists: building the Stable Diffusion XL pipeline transiently costs about 4.5 GB of
# system memory even with device_map="balanced", and this machine has been hovering between 5.9 and
# 6.3 GB free while the user's browser and editor are open. Three launches today were hard-aborted
# by the watchdog DURING THE LOAD, at 1.45, 1.38 and 1.34 GB free against a 1.5 GB floor. The floor
# is not the problem and is not lowered; the launch condition is.
#
# So this script polls until free memory clears the threshold below, then launches. It gives up
# after the bounded number of ticks rather than waiting forever, because "the user needs to close
# something" is a real answer and it should arrive as a message rather than as silence.
#
#   bash run_resolution_probe.sh [required_free_gb] [max_ticks] [tick_seconds]
set -u

REQUIRED_GB="${1:-7.0}"
MAX_TICKS="${2:-60}"
TICK_S="${3:-30}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"

free_gb () {
    "$PY" -c "import psutil; print(round(psutil.virtual_memory().available / 1024**3, 2))"
}

for tick in $(seq 1 "$MAX_TICKS"); do
    FREE="$(free_gb)"
    READY="$("$PY" -c "print(1 if float('$FREE') >= float('$REQUIRED_GB') else 0)")"
    echo "tick $tick: free system memory ${FREE} GB, required ${REQUIRED_GB} GB, ready=${READY}"
    if [ "$READY" = "1" ]; then
        cd "$HERE" || exit 3
        PYTHONPATH="$REPO_ROOT" HF_HUB_DISABLE_XET=1 "$PY" resolution_probe.py --conditions "512_control_2,768_default,768_original1024,1024_native" \
            > assets/resolution_probe.log 2>&1
        STATUS=$?
        echo "resolution_probe.py exited with status $STATUS"
        exit "$STATUS"
    fi
    sleep "$TICK_S"
done

echo "GAVE UP: free system memory never reached ${REQUIRED_GB} GB in $((MAX_TICKS * TICK_S)) seconds."
echo "The pipeline load needs about 4.5 GB transiently and the watchdog floor is 1.5 GB free."
exit 4
