#!/bin/bash
# Phase 0.2 — Rebuild all three per-entity caches from per-pair source of truth.
# Run from: /src/vision_unlearning/benchmarks/I_care/  (inside Docker)
set -e

echo "=== pipeline_07: people (uce 0, munba 200, distil 400) ==="
python pipeline_07_compute_interference_per_entity.py \
    --task people \
    --methods uce munba distil \
    --epochs 0 200 400

echo "=== pipeline_07: scenes (uce 0, munba 100, distil 100) ==="
python pipeline_07_compute_interference_per_entity.py \
    --task scenes \
    --methods uce munba distil \
    --epochs 0 100 100

echo "=== pipeline_07: breeds (uce 0, munba 50, distil 100) ==="
python pipeline_07_compute_interference_per_entity.py \
    --task breeds \
    --methods uce munba distil \
    --epochs 0 50 100

echo "=== All per-entity caches rebuilt ==="
