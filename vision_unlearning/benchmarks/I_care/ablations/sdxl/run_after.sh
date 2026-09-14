#!/usr/bin/env bash
# Waits for a running process to finish, checks that it finished SUCCESSFULLY, then runs the next job.
#
#   bash run_after.sh <pid-to-wait-for> <success-marker> <log-to-check> <script.sh> [arguments...]
#
# Why the success marker and not just the exit of the process: this machine's jobs are driven by shell
# scripts whose failure path also exits cleanly, and starting three hours of graphics-card work on top
# of a half-finished campaign wastes the card and produces artifacts nobody can trust. So the next job
# starts only when the previous one's own log says it completed.
#
# The point of this script is that the card does not sit idle between two long jobs when nobody is
# watching. It is not a scheduler: it holds exactly one dependency, and it never retries.
set -u

WAIT_PID="${1:?pid to wait for}"
MARKER="${2:?success marker expected in the log}"
LOG="${3:?log file to check for the marker}"
shift 3

echo "$(date +%H:%M:%S) waiting for PID ${WAIT_PID} before running: $*"
while ps -p "$WAIT_PID" > /dev/null 2>&1; do
    sleep 30
done
echo "$(date +%H:%M:%S) PID ${WAIT_PID} has exited"

if ! grep -q "$MARKER" "$LOG" 2>/dev/null; then
    echo "RUN_AFTER_ABORTED: '${MARKER}' is not in ${LOG}; the previous job did not finish cleanly."
    echo "--- last lines of that log ---"
    tail -n 15 "$LOG" 2>/dev/null
    exit 1
fi

echo "$(date +%H:%M:%S) marker found; starting: $*"
exec "$@"
