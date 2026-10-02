#!/bin/bash

# Stop immediately if any command fails.
set -e

# ============================================================
# Activate conda environment
# ============================================================

echo "============================================================"
echo "Activating conda environment: vision_unlearn_stable"
echo "============================================================"

# Make conda available inside the bash script.
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate vision_unlearn_stable


# ============================================================
# Running pipeline_04 for Blossom_Season
# ============================================================

echo "============================================================"
echo "Running pipeline_04 for Blossom_Season"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_04_generate_dataset.py \
--model-path "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--output-folder "/home/leonardo/soham/huggingface_assets/datasets/Blossom_Season" \
--emitter Blossom_Season \
--method uce \
--unet-state-dict "/home/leonardo/soham/models/Blossom_Season_uce_sd_style50/unet_state_dict.pth" \
--prefix on \
--device cuda \
--overwrite


# ============================================================
# Running pipeline_06 for Blossom_Season
# ============================================================

echo "============================================================"
echo "Running pipeline_06 for Blossom_Season"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_06_compute_interference_per_pair.py \
--answer-set-folder "/home/leonardo/soham/huggingface_assets/datasets/Blossom_Season_uce_sd_style_50" \
--style-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_style.pth" \
--object-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_object.pth" \
--output-path "/home/leonardo/soham/huggingface_assets/results/results/interferences_caused_by_3_uce_sd_style50.json" \
--seed 188 \
--prefix on \
--device cuda \
--baseline-path "/home/leonardo/soham/huggingface_assets/results/results/accuracies_original.json"


# ============================================================
# Running pipeline_04 for Bricks
# ============================================================

echo "============================================================"
echo "Running pipeline_04 for Bricks"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_04_generate_dataset.py \
--model-path "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--output-folder "/home/leonardo/soham/huggingface_assets/datasets/Bricks" \
--emitter Bricks \
--method uce \
--unet-state-dict "/home/leonardo/soham/models/Bricks_uce_sd_style50/unet_state_dict.pth" \
--prefix on \
--device cuda \
--overwrite


# ============================================================
# Running pipeline_06 for Bricks
# ============================================================

echo "============================================================"
echo "Running pipeline_06 for Bricks"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_06_compute_interference_per_pair.py \
--answer-set-folder "/home/leonardo/soham/huggingface_assets/datasets/Bricks_uce_sd_style_50" \
--style-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_style.pth" \
--object-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_object.pth" \
--output-path "/home/leonardo/soham/huggingface_assets/results/results/interferences_caused_by_4_uce_sd_style50.json" \
--seed 188 \
--prefix on \
--device cuda \
--baseline-path "/home/leonardo/soham/huggingface_assets/results/results/accuracies_original.json"


# ============================================================
# Running pipeline_03 for Architectures
# ============================================================

echo "============================================================"
echo "Running pipeline_03 for Architectures"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_03_unlearn_model.py \
--emitter Architectures \
--checkpoint "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--upstream-script "/home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/uce_train_erase.py" \
--output-folder "/home/leonardo/soham/models/" \
--expected-model-folder "/home/leonardo/soham/models/Architectures_uce_sd_style50" \
--python-executable python \
--working-directory /home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/


# ============================================================
# Running pipeline_04 for Architectures
# ============================================================

echo "============================================================"
echo "Running pipeline_04 for Architectures"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_04_generate_dataset.py \
--model-path "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--output-folder "/home/leonardo/soham/huggingface_assets/datasets/Architectures" \
--emitter Architectures \
--method uce \
--unet-state-dict "/home/leonardo/soham/models/Architectures_uce_sd_style50/unet_state_dict.pth" \
--prefix on \
--device cuda \
--overwrite


# ============================================================
# Running pipeline_06 for Architectures
# ============================================================

echo "============================================================"
echo "Running pipeline_06 for Architectures"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_06_compute_interference_per_pair.py \
--answer-set-folder "/home/leonardo/soham/huggingface_assets/datasets/Architectures_uce_sd_style_50" \
--style-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_style.pth" \
--object-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_object.pth" \
--output-path "/home/leonardo/soham/huggingface_assets/results/results/interferences_caused_by_52_uce_sd_style50.json" \
--seed 188 \
--prefix on \
--device cuda \
--baseline-path "/home/leonardo/soham/huggingface_assets/results/results/accuracies_original.json"


# ============================================================
# Running pipeline_03 for Bears
# ============================================================

echo "============================================================"
echo "Running pipeline_03 for Bears"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_03_unlearn_model.py \
--emitter Bears \
--checkpoint "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--upstream-script "/home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/uce_train_erase.py" \
--output-folder "/home/leonardo/soham/models/" \
--expected-model-folder "/home/leonardo/soham/models/Bears_uce_sd_style50" \
--python-executable python \
--working-directory /home/leonardo/soham/UnlearnCanvas/machine_unlearning/mu_unified_concept_editing_uce/


# ============================================================
# Running pipeline_04 for Bears
# ============================================================

echo "============================================================"
echo "Running pipeline_04 for Bears"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_04_generate_dataset.py \
--model-path "/home/leonardo/soham/huggingface_assets/style50_checkpoints/style50_checkpoints/" \
--output-folder "/home/leonardo/soham/huggingface_assets/datasets/Bears" \
--emitter Bears \
--method uce \
--unet-state-dict "/home/leonardo/soham/models/Bears_uce_sd_style50/unet_state_dict.pth" \
--prefix on \
--device cuda \
--overwrite


# ============================================================
# Running pipeline_06 for Bears
# ============================================================

echo "============================================================"
echo "Running pipeline_06 for Bears"
echo "============================================================"

PYTHONPATH=./vision-unlearning python3 \
vision-unlearning/vision_unlearning/benchmarks/u_care/pipeline_06_compute_interference_per_pair.py \
--answer-set-folder "/home/leonardo/soham/huggingface_assets/datasets/Bears_uce_sd_style_50" \
--style-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_style.pth" \
--object-checkpoint "/home/leonardo/soham/huggingface_assets/classifier_checkpoints/classifier_checkpoints/classifier_object.pth" \
--output-path "/home/leonardo/soham/huggingface_assets/results/results/interferences_caused_by_53_uce_sd_style50.json" \
--seed 188 \
--prefix on \
--device cuda \
--baseline-path "/home/leonardo/soham/huggingface_assets/results/results/accuracies_original.json"


# ============================================================
# ALL TASKS COMPLETED
# ============================================================

echo ""
echo "============================================================"
echo "ALL U-CARE TASKS COMPLETED SUCCESSFULLY"
echo "============================================================"
