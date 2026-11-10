#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/apbr_l2_dense_seed1to4_20261103
project="$base/project/vpr_sam_research/experiments/post_training_nm_benchmark"
output="$base/artifacts/run_v1"
if [[ -e "$output" || -e "$base/controller.pid" ]]; then
  echo 'Existing queue output or PID; refusing duplicate launch' >&2
  exit 1
fi
cd "$project"
nohup /root/miniconda3/bin/python -u scripts/run_dense_sam_apbr_l2_seed1to4_queue.py \
  --dataset-root "$base/data" \
  --output-root "$output" \
  --seed1-sam-identity "$project/configs/seed1_sam_reused_identity_20261103.json" \
  > "$base/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$base/controller.pid"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'Controller exited immediately' >&2
  cat "$base/controller.log" >&2
  exit 1
fi
printf 'controller_pid=%s\n' "$pid"
if [[ -f "$output/queue_status.json" ]]; then
  /root/miniconda3/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("status="+d["status"], "first_job="+str(d["jobs"][0]))' "$output/queue_status.json"
else
  echo 'Controller running; status initialization pending'
fi
