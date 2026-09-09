#!/usr/bin/env bash
# Second feasibility pass: price each memory lever separately, on cached weights.
#
# The first pass established two facts. One full SPARE step at 512 with batch 1 runs, but peaks at
# 12.80 GB against 11.98 GB of dedicated video memory, so it is only running because the driver
# spills into system memory -- which took free system memory down to 1.58 GB. And every 1024 stage
# died with a HIP launch failure inside the *autoencoder* (the encoder while training, the decoder
# while generating), never inside the denoiser, which completed 50 denoising steps at 1024 before
# the decode killed it.
#
# So the levers are tried in the order the evidence points at, one at a time, so each one's effect
# on peak memory and on seconds per step is attributable:
#
#   A  split backward          one activation graph alive instead of two; same parameter update
#   B  A + autoencoder tiling  the component that actually failed at 1024
#   C  B + precompute          autoencoder and text encoders evicted after one use
#   D  C + gradient checkpointing   the expensive lever, last
#
# Usage (from this directory):  bash run_probe_pass2.sh
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

# --- 512, the resolution the proof of concept used: how far under the limit can we get? ---
run_stage 512_A_split --stage step --resolution 512 --batch-size 1 --steps 4 \
    --split-backward --tag 512_A_split
run_stage 512_C_split_tile_precompute --stage step --resolution 512 --batch-size 1 --steps 4 \
    --split-backward --vae-tiling --precompute --tag 512_C_split_tile_precompute

# --- 1024, the resolution SDXL was trained for and the one the subtle-difference reading needs ---
run_stage 1024_B_split_tile --stage step --resolution 1024 --batch-size 1 --steps 4 \
    --split-backward --vae-tiling --tag 1024_B_split_tile
run_stage 1024_C_split_tile_precompute --stage step --resolution 1024 --batch-size 1 --steps 4 \
    --split-backward --vae-tiling --precompute --tag 1024_C_split_tile_precompute
run_stage 1024_D_all_levers --stage step --resolution 1024 --batch-size 1 --steps 4 \
    --split-backward --vae-tiling --precompute --gradient-checkpointing --tag 1024_D_all_levers

# --- generation at 1024 with the autoencoder tiled: the campaign's dominant cost ---
run_stage generate_1024_tiled --stage generate --resolution 1024 --images 2 --vae-tiling --tag 1024_tiled

echo "PROBE_ALL_DONE $(date +%H:%M:%S)"
