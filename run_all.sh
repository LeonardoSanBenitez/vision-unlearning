#!/bin/bash

set -euo pipefail

# Resolve paths relative to this script so it can be launched from any directory.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$REPO_DIR")"
cd "$WORKSPACE_DIR"

echo "============================================================"
echo "Activating conda environment: vision_unlearn_stable"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision_unlearn_stable

MODEL_PATH="/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/"
UPSTREAM_SCRIPT="/home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/uce_train_erase.py"
UPSTREAM_WORKDIR="/home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/"
MODEL_OUTPUT="/home/leonardo/soham/models"
DATASET_ROOT="/home/leonardo/soham/huggingface_assets/datasets"
CLASSIFIER_ROOT="/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints"
RESULTS_DIR="/home/leonardo/soham/huggingface_assets/results/results"
BASELINE_PATH="$RESULTS_DIR/accuracies_original.json"
STYLE_CLASSIFIER="$CLASSIFIER_ROOT/classifier_style.pth"
OBJECT_CLASSIFIER="$CLASSIFIER_ROOT/classifier_object.pth"
PIPELINE_DIR="vision-unlearning/vision_unlearning/benchmarks/u_care"
EMITTERS=(
  Abstractionism
  Artist_Sketch
  Blossom_Season
  Bricks
  Van_Gogh
  Architectures
  Bears
)

for required_path in \
  "$MODEL_PATH" \
  "$UPSTREAM_SCRIPT" \
  "$UPSTREAM_WORKDIR" \
  "$STYLE_CLASSIFIER" \
  "$OBJECT_CLASSIFIER" \
  "$BASELINE_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$MODEL_OUTPUT" "$DATASET_ROOT" "$RESULTS_DIR"

for emitter in "${EMITTERS[@]}"; do
  echo "============================================================"
  echo "Running UCE pipeline for $emitter"
  echo "============================================================"

  model_folder="$MODEL_OUTPUT/${emitter}_uce_sd_style50"
  answer_set_folder="$DATASET_ROOT/generated_${emitter}_uce_sd_style50"
  pair_output="$RESULTS_DIR/interferences_caused_by_${emitter}_uce_sd_style50.json"

  echo "Running pipeline_03 for $emitter"
  PYTHONPATH=./vision-unlearning python3 \
    "$PIPELINE_DIR/pipeline_03_unlearn_model.py" \
    --emitter "$emitter" \
    --checkpoint "$MODEL_PATH" \
    --upstream-script "$UPSTREAM_SCRIPT" \
    --output-folder "$MODEL_OUTPUT" \
    --expected-model-folder "$model_folder" \
    --python-executable python \
    --working-directory "$UPSTREAM_WORKDIR" \
    --overwrite

  echo "Running pipeline_04 for $emitter"
  PYTHONPATH=./vision-unlearning python3 \
    "$PIPELINE_DIR/pipeline_04_generate_dataset.py" \
    --model-path "$MODEL_PATH" \
    --output-folder "$answer_set_folder" \
    --emitter "$emitter" \
    --method uce \
    --unet-state-dict "$model_folder/unet_state_dict.pth" \
    --prefix on \
    --device cuda \
    --overwrite

  echo "Running pipeline_06 for $emitter"
  PYTHONPATH=./vision-unlearning python3 \
    "$PIPELINE_DIR/pipeline_06_compute_interference_per_pair.py" \
    --answer-set-folder "$answer_set_folder" \
    --style-checkpoint "$STYLE_CLASSIFIER" \
    --object-checkpoint "$OBJECT_CLASSIFIER" \
    --output-path "$pair_output" \
    --seed 188 \
    --prefix on \
    --device cuda \
    --emitter "$emitter" \
    --method uce \
    --baseline-path "$BASELINE_PATH" \
    --overwrite
done

echo ""
echo "============================================================"
echo "ALL REQUESTED U-CARE UCE TASKS COMPLETED SUCCESSFULLY"
echo "============================================================"