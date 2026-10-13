#!/usr/bin/env bash
# Run all 5 methods across 3 seeds on ResNet18-CIFAR-10.
# Designed to be cheap; tune EPOCHS / BATCH via env vars.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/.."

EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-128}
SEEDS=${SEEDS:-"0 1 2"}

run() {
    local method=$1
    local extra=$2
    local seed=$3
    echo "=== run: $method (seed=$seed) ==="
    python -m nm_sam_cifar10.train \
        --method "$method" \
        --model resnet18_cifar \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --num-workers 0 \
        --mask-update-interval 50 \
        --sharpness-interval 1 \
        --seed "$seed" \
        --output-dir "./nm_sam_cifar10/results/${method}_seed${seed}" \
        $extra
}

for seed in $SEEDS; do
    run dense_sgd    ""          "$seed"
    run dense_sam    "--rho 0.05" "$seed"
    run nm_sgd       ""          "$seed"
    run nm_sam       "--rho 0.05" "$seed"
    run nm_sam_aware "--rho 0.05" "$seed"
done

python -m nm_sam_cifar10.plot --results-dir ./nm_sam_cifar10/results
echo "All runs complete. See ./nm_sam_cifar10/results/summary.csv and plots/."
