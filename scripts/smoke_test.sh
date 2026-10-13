#!/usr/bin/env bash
# Smoke test: run each of the 5 methods for 1-2 epochs on a subset of CIFAR-10
# using SimpleCNN. Verifies forward / backward / mask / SAM / checkpoint flow.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/.."

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-256}
SUBSET=${SUBSET:-2048}

run() {
    local method=$1
    local extra=$2
    echo "=== smoke: $method ==="
    python -m nm_sam_cifar10.train \
        --method "$method" \
        --model simple_cnn \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --subset-train "$SUBSET" \
        --subset-test "$SUBSET" \
        --num-workers 0 \
        --mask-update-interval 5 \
        --sharpness-interval 1 \
        --output-dir "./nm_sam_cifar10/results/smoke_${method}" \
        $extra
}

run dense_sgd    ""
run dense_sam    "--rho 0.05"
run nm_sgd       ""
run nm_sam       "--rho 0.05"
run nm_sam_aware "--rho 0.05"

echo "Smoke test OK for all 5 methods."
