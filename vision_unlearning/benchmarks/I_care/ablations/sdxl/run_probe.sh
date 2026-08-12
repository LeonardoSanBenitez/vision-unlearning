#!/usr/bin/env bash
# Feasibility spike for SPARE unlearning of Stable Diffusion XL on this machine.
#
# Runs every stage of probe_sdxl_memory.py in one process, cheapest-first, and does NOT stop when a
# stage fails: an out-of-memory failure at one resolution is itself a measurement, and the stages
# after it still carry information. Each stage is its own interpreter, so video memory is fully
# released between them.
#
# Usage (from this directory):  bash run_probe.sh
set -u

PY="/c/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

run_stage () {
    local label="$1"; shift
    echo "=================================================================="
    echo "PROBE_STAGE_START ${label} $(date +%H:%M:%S)"
    echo "=================================================================="
    if "$PY" probe_sdxl_memory.py "$@"; then
        echo "PROBE_STAGE_OK ${label} $(date +%H:%M:%S)"
    else
        echo "PROBE_STAGE_FAILED ${label} exit=$? $(date +%H:%M:%S)"
    fi
}

# 1. Load only: proves the component set fits in video memory at all, and pays the ~7 GB download.
run_stage load --stage load --dtype float16 --tag fp16

# 2. The real question: one full SPARE step at the resolution the proof of concept used.
run_stage step_512_bs1 --stage step --dtype float16 --resolution 512 --batch-size 1 --steps 4 --tag 512_bs1

# 3. Same at the resolution SDXL was actually trained for.
run_stage step_1024_bs1 --stage step --dtype float16 --resolution 1024 --batch-size 1 --steps 4 --tag 1024_bs1

# 4. If 1024 does not fit, gradient checkpointing is the first lever; measure its cost.
run_stage step_1024_bs1_ckpt --stage step --dtype float16 --resolution 1024 --batch-size 1 --steps 4 \
    --gradient-checkpointing --tag 1024_bs1_ckpt

# 5. Generation cost, which dominates the campaign budget: 10 entities x 13 epochs x 2 seeds.
run_stage generate_1024 --stage generate --dtype float16 --resolution 1024 --images 2 --tag 1024
run_stage generate_512 --stage generate --dtype float16 --resolution 512 --images 2 --tag 512

echo "PROBE_ALL_DONE $(date +%H:%M:%S)"
