#!/usr/bin/env bash
# Drives one whole run of plan stage S4 -- train, generate, report -- one process per stage, in risk
# order, and RESUMABLE: re-running it after the machine was switched off picks up where it stopped.
#
#   bash run_s4_schedule_probe.sh                                              # inherited defaults
#   bash run_s4_schedule_probe.sh --learning-rate 1e-4 --lora-r 4 --forget-weight 0.5
#
# Every argument is passed straight through to run_schedule_probe.py, which defaults to the inherited
# Stable Diffusion 1.4 hyperparameters and tags all of its artifacts with the four values, so two
# settings never collide and no argument has to be repeated per stage by hand.
#
# What "resumable" rests on, per stage:
#   * train    -- skipped when all four adapter files are already on disk. It cannot resume mid-way:
#                 the training stage deletes the model directory and starts clean, so an interrupted
#                 training is redone from epoch 1. At ~28 minutes that is an acceptable loss; the
#                 200-epoch campaign at S5 is the run that needs real checkpoint resume.
#   * generate -- images already on disk are skipped inside the script, so an interrupted stage costs
#                 the one image in flight.
#   * report   -- reads only what is on disk and is safe to repeat.
#
# Every stage goes through run_stage.sh, which waits for memory headroom and retries. Logs:
# assets/schedule_probe_<stage>_<tag>[_<label>].log, with <tag> the same hyperparameter tag the
# artifacts carry.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 3

REPO_ROOT="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"
HP=("$@")
CHECKPOINTS=(1 3 5 10)

# The tag is computed by the script itself (`--stage tag`), so the shell never carries a second copy
# of the naming rule that would drift from it.
TAG="$(PYTHONPATH="$REPO_ROOT" "$PY" run_schedule_probe.py --stage tag "$@")"
if [ -z "$TAG" ]; then
    echo "could not compute the run tag; refusing to launch"
    exit 3
fi
echo "run tag: ${TAG}"

echo "===== ${TAG}: train ====="
MODEL_DIR="assets/schedule_probe_model_${TAG}"
TRAINED=1
for EPOCH in "${CHECKPOINTS[@]}"; do
    [ -f "${MODEL_DIR}/epoch-${EPOCH}/pytorch_lora_weights.safetensors" ] || TRAINED=0
done
if [ "$TRAINED" = "1" ]; then
    echo "all ${#CHECKPOINTS[@]} adapters already on disk under ${MODEL_DIR}; skipping training"
else
    bash run_stage.sh "schedule_probe_train_${TAG}" SCHEDULE_PROBE_TRAIN_DONE run_schedule_probe.py \
        --stage train "${HP[@]}" || exit $?
fi

echo "===== ${TAG}: generate off-baseline ====="
bash run_stage.sh "schedule_probe_generate_off_${TAG}" SCHEDULE_PROBE_GENERATE_DONE run_schedule_probe.py \
    --stage generate --off "${HP[@]}" || exit $?

for EPOCH in "${CHECKPOINTS[@]}"; do
    echo "===== ${TAG}: generate epoch ${EPOCH} ====="
    bash run_stage.sh "schedule_probe_generate_epoch${EPOCH}_${TAG}" SCHEDULE_PROBE_GENERATE_DONE run_schedule_probe.py \
        --stage generate --epoch "$EPOCH" "${HP[@]}" || exit $?
done

echo "===== ${TAG}: report ====="
bash run_stage.sh "schedule_probe_report_${TAG}" SCHEDULE_PROBE_REPORT_DONE run_schedule_probe.py \
    --stage report "${HP[@]}" || exit $?

echo "S4_SCHEDULE_PROBE_DONE tag=${TAG}"
