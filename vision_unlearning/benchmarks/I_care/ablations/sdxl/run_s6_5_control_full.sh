#!/usr/bin/env bash
# S6.5 in full: the random ten measured exactly like the selected ten -- every checkpoint, both seeds.
#
# The control began as an endpoint probe (off-baseline and epoch 200, seed 42) and answered its
# question: collateral damage exists on entities nobody selected. It could not answer the question
# that turned out to matter more, because this campaign's largest effects are transient and sit in
# the middle of training. So the control now generates what the campaign generates, and the two sets
# are drawn in the same grids and read against the same floor.
#
# Whatever already exists is skipped: the ten seed-42 off-baselines and the ten seed-42 epoch-200
# images are on disk and in the manifest, so those two stages print their marker and cost nothing.
#
# Cost, at the measured 41 s an image plus about 23 s of pipeline build per stage: 130 images for
# seed 42 and 140 for seed 43, so roughly 3.2 hours. Run it after the seed-43 campaign, never beside
# it -- two Stable Diffusion XL pipelines on this machine is the condition that takes the host down.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"

cd "$HERE" || exit 3
# tr -d strips any carriage return the interpreter's stdout translation may add; a label carrying one
# produces a stage that fails identically eight times over and reports nothing useful.
LABELS="$(PYTHONPATH="$REPO_ROOT" "$PY" random_ten_control.py --stage labels | tr -d "\r")"
if [ -z "$LABELS" ]; then
    echo "S6_5_FULL_FAILED stage=labels status=empty"
    exit 6
fi

for SEED in 42 43; do
    echo "=== s6.5 full: seed ${SEED}, labels: $(echo $LABELS | tr '\n' ' ')"
    for LABEL in $LABELS; do
        echo "=== s6.5 full: seed ${SEED} generating ${LABEL} at $(date +%H:%M:%S)"
        WAIT_TICKS=240 bash "$HERE/run_stage.sh" "random_ten_control_seed${SEED}_${LABEL}" \
            RANDOM_TEN_GENERATE_DONE random_ten_control.py \
            --stage generate --seed "$SEED" --epoch "$LABEL"
        STATUS=$?
        if [ "$STATUS" -ne 0 ]; then
            echo "S6_5_FULL_FAILED seed=${SEED} label=${LABEL} status=${STATUS}"
            exit "$STATUS"
        fi
    done
done

echo "=== s6.5 full: scoring both seeds at $(date +%H:%M:%S)"
REQUIRED_GB=2.0 WAIT_TICKS=240 bash "$HERE/run_stage.sh" random_ten_control_score \
    RANDOM_TEN_SCORE_DONE random_ten_control.py --stage score --seeds 42,43
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "S6_5_FULL_FAILED stage=score status=${STATUS}"
    exit "$STATUS"
fi

echo "S6_5_FULL_ALL_STAGES_FINISHED at $(date +%H:%M:%S)"
