#!/usr/bin/env bash
# S6.5, the random-ten control: both generation stages and the scoring, one process per stage.
#
# The draw (`--stage draw`) is deliberately NOT here. It costs nothing, decides which ten entities
# every later number is about, and is looked at by a human before any graphics-card time is spent --
# so it is run on its own and this driver starts from images.
#
# Each stage goes through run_stage.sh, which is where the launch discipline lives: wait for free
# system memory, run, retry until the completion marker appears, keep every attempt's log. The two
# generation stages need the full 6.0 GB headroom (building the Stable Diffusion XL pipeline costs
# about 3.1 GB transiently, measured from campaign_generate_seed42_epoch3_monitor.log: 8.01 GB free
# at the start of the load, 4.88 GB at its trough, and the watchdog aborts below 1.5 GB). The scoring
# stage loads CLIP alone, so it asks for 2.0 GB and is not made to queue behind a gate it does not
# need.
#
# WAIT_TICKS is raised well above run_stage.sh's default because this machine's free memory belongs
# to whatever else is open on it: 240 ticks of 30 s is two hours of patience per attempt, so the run
# starts by itself whenever the headroom appears instead of failing while nobody is watching.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== s6.5 control: off-baselines at $(date +%H:%M:%S)"
WAIT_TICKS=240 bash "$HERE/run_stage.sh" random_ten_control_off RANDOM_TEN_GENERATE_DONE \
    random_ten_control.py --stage generate --epoch off
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S6_5_CONTROL_FAILED stage=off status=${STATUS}"
    exit "$STATUS"
fi

echo "=== s6.5 control: epoch 200 at $(date +%H:%M:%S)"
WAIT_TICKS=240 bash "$HERE/run_stage.sh" random_ten_control_epoch200 RANDOM_TEN_GENERATE_DONE \
    random_ten_control.py --stage generate --epoch 200
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S6_5_CONTROL_FAILED stage=epoch200 status=${STATUS}"
    exit "$STATUS"
fi

echo "=== s6.5 control: scoring at $(date +%H:%M:%S)"
REQUIRED_GB=2.0 WAIT_TICKS=240 bash "$HERE/run_stage.sh" random_ten_control_score RANDOM_TEN_SCORE_DONE \
    random_ten_control.py --stage score
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S6_5_CONTROL_FAILED stage=score status=${STATUS}"
    exit "$STATUS"
fi

echo "S6_5_CONTROL_ALL_STAGES_FINISHED at $(date +%H:%M:%S)"
