#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/apbr_gpt_joint_phase4_20261101
handoff="$base/phase4_1/server_bundle_v3_handoff_fix1"
project="$handoff/source/project"
export PYTHONPATH="$handoff/source/vpr_src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -e "$base/seed1_sparse_20261101_v2" ]]; then
  echo 'Refusing existing queue output' >&2
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU occupied before queue launch' >&2
  exit 2
fi
cd "$project"
exec /root/miniconda3/bin/python scripts/run_joint_seed1_sparse_queue_v2.py \
  --project-root "$project" \
  --source-root "$handoff/source/vpr_src" \
  --data-root "$base/data" \
  --protocol-config "$base/phase4_1/config/apbr_gpt_joint_seed0_rtx5090_20261101.yaml" \
  --joint-config "$handoff/config/apbr_gpt_joint_seed0_calibration.json" \
  --calibration-dir "$handoff/calibration" \
  --initial-seed0 "$base/server_bundle_v3/model_assets/initial_common_seed0.pt" \
  --initial-seed1 "$base/seed1_assets_v2/initial_common_seed1.pt" \
  --output-root "$base/seed1_sparse_20261101_v2"
