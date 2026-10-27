#!/bin/bash

set -euo pipefail

# Resolve both supported layouts:
#   workspace/run_all2.sh + workspace/vision-unlearning/
#   checkout/run_all2.sh
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
UPSTREAM_WORKDIR="/home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/"
UPSTREAM_SCRIPT="$PIPELINE_DIR/uce_train_erase.py"
MODEL_OUTPUT="/home/leonardo/soham/models"
DATASET_ROOT="/home/leonardo/soham/huggingface_assets/datasets"
CLASSIFIER_ROOT="/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints"
RESULTS_DIR="/home/leonardo/soham/huggingface_assets/results/results"
BASELINE_PATH="$RESULTS_DIR/accuracies_original.json"
STYLE_CLASSIFIER="$CLASSIFIER_ROOT/classifier_style.pth"
OBJECT_CLASSIFIER="$CLASSIFIER_ROOT/classifier_object.pth"

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

echo "Using UCE implementation: $UPSTREAM_SCRIPT"
sha256sum "$UPSTREAM_SCRIPT"

mapfile -t PENDING_EMITTERS < <(
  PYTHONPATH="$PYTHON_PACKAGE_ROOT" python3 - <<'PY'
from vision_unlearning.benchmarks.u_care.configuration import UNLEARNABLE_ENTITIES

for emitter in UNLEARNABLE_ENTITIES:
    print(emitter)
PY
)

if [[ ${#PENDING_EMITTERS[@]} -ne 70 ]]; then
  echo "Expected 70 emitters, found ${#PENDING_EMITTERS[@]}. Aborting." >&2
  exit 1
fi

echo "Regenerating UCE artifacts for all ${#PENDING_EMITTERS[@]} emitters: ${PENDING_EMITTERS[*]}"

for emitter in "${PENDING_EMITTERS[@]}"; do
  echo "============================================================"
  echo "Running UCE pipelines for $emitter"
  echo "============================================================"

  model_folder="$MODEL_OUTPUT/${emitter}_uce_sd_style50"
  answer_set_folder="$DATASET_ROOT/generated_${emitter}_uce_sd_style50"
  pair_output="$RESULTS_DIR/interferences_caused_by_${emitter}_uce_sd_style50.json"

  echo "Running pipeline_03 for $emitter"
  PYTHONPATH="$UPSTREAM_WORKDIR:$PYTHON_PACKAGE_ROOT" python3 \
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
  PYTHONPATH="$PYTHON_PACKAGE_ROOT" python3 \
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
  PYTHONPATH="$PYTHON_PACKAGE_ROOT" python3 \
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
echo "ALL 70 U-CARE UCE EMITTERS REGENERATED SUCCESSFULLY"
echo "============================================================"