#!/bin/bash

set -euo pipefail

# Resolve both supported layouts:
#   workspace/run_stage4_five_seed.sh + workspace/vision-unlearning/
#   checkout/run_stage4_five_seed.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/vision-unlearning/vision_unlearning/benchmarks/u_care" ]]; then
  WORKSPACE_DIR="$SCRIPT_DIR"
  PYTHON_PACKAGE_ROOT="$WORKSPACE_DIR/vision-unlearning"
  PIPELINE_DIR="$PYTHON_PACKAGE_ROOT/vision_unlearning/benchmarks/u_care"
elif [[ -d "$SCRIPT_DIR/vision_unlearning/benchmarks/u_care" ]]; then
  WORKSPACE_DIR="$SCRIPT_DIR"
  PYTHON_PACKAGE_ROOT="$WORKSPACE_DIR"
  PIPELINE_DIR="$PYTHON_PACKAGE_ROOT/vision_unlearning/benchmarks/u_care"
else
  echo "Could not locate vision_unlearning/benchmarks/u_care relative to $SCRIPT_DIR" >&2
  exit 1
fi
cd "$WORKSPACE_DIR"

echo "============================================================"
echo "Activating conda environment: vision_unlearn_stable"
echo "============================================================"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision_unlearn_stable

MODEL_PATH="/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/"
MODEL_OUTPUT="/home/leonardo/soham/models"
DATASET_ROOT="/home/leonardo/soham/huggingface_assets/datasets"
CLASSIFIER_ROOT="/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints"
RESULTS_ROOT="/home/leonardo/soham/huggingface_assets/results"
SEED188_RESULTS="$RESULTS_ROOT/results"
ALL_SEED_RESULTS="${ALL_SEED_RESULTS:-$RESULTS_ROOT/all_seeds/results}"
STYLE_CLASSIFIER="$CLASSIFIER_ROOT/classifier_style.pth"
OBJECT_CLASSIFIER="$CLASSIFIER_ROOT/classifier_object.pth"
SEEDS=(188 288 588 688 888)

for required_path in \
  "$MODEL_PATH" \
  "$STYLE_CLASSIFIER" \
  "$OBJECT_CLASSIFIER" \
  "$SEED188_RESULTS/metadata_filtered.json" \
  "$DATASET_ROOT/generated_baseline_sd_style50"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$ALL_SEED_RESULTS"
if [[ -e "$ALL_SEED_RESULTS/accuracies_original.json" ]] || \
   compgen -G "$ALL_SEED_RESULTS/interferences_caused_by_*_uce_sd_style50.json" > /dev/null; then
  echo "Five-seed results already exist under $ALL_SEED_RESULTS." >&2
  echo "Choose a fresh path with ALL_SEED_RESULTS=/path/to/new/results to avoid overwriting them." >&2
  exit 1
fi

cp "$SEED188_RESULTS/metadata_filtered.json" "$ALL_SEED_RESULTS/metadata_filtered.json"

mapfile -t EMITTERS < <(
  PYTHONPATH="$PYTHON_PACKAGE_ROOT" python3 - <<'PY'
from vision_unlearning.benchmarks.u_care.configuration import UNLEARNABLE_ENTITIES

for emitter in UNLEARNABLE_ENTITIES:
    print(emitter)
PY
)

if [[ ${#EMITTERS[@]} -ne 70 ]]; then
  echo "Expected 70 unlearnable emitters, found ${#EMITTERS[@]}. Aborting." >&2
  exit 1
fi

for emitter in "${EMITTERS[@]}"; do
  checkpoint="$MODEL_OUTPUT/${emitter}_uce_sd_style50/unet_state_dict.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing validated UCE checkpoint for $emitter: $checkpoint" >&2
    exit 1
  fi
  answer_set_folder="$DATASET_ROOT/generated_${emitter}_uce_sd_style50"
  if [[ ! -d "$answer_set_folder" ]]; then
    echo "Missing seed-188 answer-set folder for $emitter: $answer_set_folder" >&2
    exit 1
  fi
done

run_command() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo "Generating/completing baseline answer set for seeds: ${SEEDS[*]}"
run_command env "PYTHONPATH=$PYTHON_PACKAGE_ROOT" python3 \
  "$PIPELINE_DIR/pipeline_04_generate_dataset.py" \
  --model-path "$MODEL_PATH" \
  --output-folder "$DATASET_ROOT/generated_baseline_sd_style50" \
  --seed "${SEEDS[@]}" \
  --prefix off \
  --device cuda

echo "Scoring baseline answer set"
run_command env "PYTHONPATH=$PYTHON_PACKAGE_ROOT" python3 \
  "$PIPELINE_DIR/pipeline_06_compute_interference_per_pair.py" \
  --answer-set-folder "$DATASET_ROOT/generated_baseline_sd_style50" \
  --style-checkpoint "$STYLE_CLASSIFIER" \
  --object-checkpoint "$OBJECT_CLASSIFIER" \
  --output-path "$ALL_SEED_RESULTS/accuracies_original.json" \
  --seed "${SEEDS[@]}" \
  --prefix off \
  --device cuda

for emitter in "${EMITTERS[@]}"; do
  echo "============================================================"
  echo "Completing five-seed UCE evaluation for $emitter"
  echo "============================================================"

  answer_set_folder="$DATASET_ROOT/generated_${emitter}_uce_sd_style50"
  checkpoint="$MODEL_OUTPUT/${emitter}_uce_sd_style50/unet_state_dict.pth"
  pair_output="$ALL_SEED_RESULTS/interferences_caused_by_${emitter}_uce_sd_style50.json"

  run_command env "PYTHONPATH=$PYTHON_PACKAGE_ROOT" python3 \
    "$PIPELINE_DIR/pipeline_04_generate_dataset.py" \
    --model-path "$MODEL_PATH" \
    --output-folder "$answer_set_folder" \
    --emitter "$emitter" \
    --method uce \
    --unet-state-dict "$checkpoint" \
    --seed "${SEEDS[@]}" \
    --prefix on \
    --device cuda

  run_command env "PYTHONPATH=$PYTHON_PACKAGE_ROOT" python3 \
    "$PIPELINE_DIR/pipeline_06_compute_interference_per_pair.py" \
    --answer-set-folder "$answer_set_folder" \
    --style-checkpoint "$STYLE_CLASSIFIER" \
    --object-checkpoint "$OBJECT_CLASSIFIER" \
    --output-path "$pair_output" \
    --seed "${SEEDS[@]}" \
    --prefix on \
    --device cuda \
    --emitter "$emitter" \
    --method uce \
    --baseline-path "$ALL_SEED_RESULTS/accuracies_original.json"
done

echo "Aggregating five-seed per-entity metrics"
run_command env "PYTHONPATH=$PYTHON_PACKAGE_ROOT" python3 \
  "$PIPELINE_DIR/pipeline_07_compute_interference_per_entity.py" \
  --method uce \
  --base-folder "$ALL_SEED_RESULTS"

echo "Five-seed UCE artifacts written to $ALL_SEED_RESULTS"