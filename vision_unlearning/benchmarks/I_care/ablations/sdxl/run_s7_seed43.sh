#!/usr/bin/env bash
# S7: the whole second seed. Trains seed 43 for 200 epochs, generates its 140 images one epoch per
# process, then re-scores BOTH seeds with CLIP so the two trajectories land in one artifact.
#
# Three things this fixes relative to how the seed-42 half was driven on 2026-08-19:
#
#   * every generation stage is asked for by its OWN epoch, taken from `run_campaign.py --stage
#     labels` (which reads the checkpoint list from the every-epoch campaign JSON, so no epoch list
#     is written down here). That gives one log per epoch instead of fourteen stages all called
#     `remaining` and each overwriting the previous one's log;
#   * an epoch the manifest already holds now prints the completion marker and exits, so re-running
#     this driver after an abort walks the whole list cheaply instead of failing on the first epoch
#     that is already done;
#   * the CLIP scoring runs at the end of the same driver, because a campaign whose images exist but
#     whose scores do not is a stage that looks finished and is not.
#
# Every stage goes through run_stage.sh: wait for free system memory, run, retry until the
# completion marker appears, keep each attempt's log. WAIT_TICKS is raised to 240 (two hours per
# attempt at 30 s a tick) because the free memory on this machine belongs to whatever else is open.
# The scoring stage loads CLIP alone and asks for 2.0 GB rather than the 6.0 GB a Stable Diffusion XL
# pipeline needs.
#
# Expected cost, from the seed-42 measurements: training 400 optimizer steps at 10.15 s plus about
# 8.5 minutes of load and checkpoint writes = ~1.3 h; generation 140 images at ~48 s including the
# per-epoch pipeline load = ~1.9 h.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
SEED=43
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"

echo "=== s7: training seed ${SEED} at $(date +%H:%M:%S)"
WAIT_TICKS=240 bash "$HERE/run_campaign_stage.sh" train "$SEED" -
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S7_FAILED stage=train status=${STATUS}"
    exit "$STATUS"
fi

cd "$HERE" || exit 3
# tr -d strips any carriage return the interpreter's stdout translation may add. A label
# carrying an invisible carriage return produces a stage that fails identically eight times
# over and reports nothing useful, which is what happened to the seed-43 generation half on
# 2026-08-22. The labels stage itself now writes plain newlines; this is the second belt.
LABELS="$(PYTHONPATH="$REPO_ROOT" "$PY" run_campaign.py --stage labels --seed "$SEED" | tr -d "\r")"
if [ -z "$LABELS" ]; then
    echo "S7_FAILED stage=labels status=empty"
    exit 6
fi
echo "=== s7: generation labels for seed ${SEED}: $(echo $LABELS | tr '\n' ' ')"

for LABEL in $LABELS; do
    echo "=== s7: generating ${LABEL} at $(date +%H:%M:%S)"
    WAIT_TICKS=240 bash "$HERE/run_campaign_stage.sh" generate "$SEED" "$LABEL"
    STATUS=$?
    if [ "$STATUS" -ne 0 ]; then
        echo "S7_FAILED stage=generate label=${LABEL} status=${STATUS}"
        exit "$STATUS"
    fi
done

echo "=== s7: scoring both seeds at $(date +%H:%M:%S)"
REQUIRED_GB=2.0 WAIT_TICKS=240 bash "$HERE/run_stage.sh" clip_diff_campaign_both_seeds \
    "images scored equals expected: True" clip_diff_campaign.py --seeds 42,43
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S7_FAILED stage=score status=${STATUS}"
    exit "$STATUS"
fi

echo "S7_ALL_STAGES_FINISHED at $(date +%H:%M:%S)"
