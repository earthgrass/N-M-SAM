#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${VPR_FORMAL_APPROVED:-}" != "YES" ]]; then
  echo "Set VPR_FORMAL_APPROVED=YES only after protocol/resource approval." >&2
  exit 2
fi
if [[ $# != 1 ]]; then echo "Usage: $0 configs/approved/ONE.json" >&2; exit 2; fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
.venv/bin/python scripts/preflight.py --require-5090 --config "$1"
exec .venv/bin/python -m vpr_sam train --config "$1" --allow-formal
