#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Default forwards to dry-run; --execute must be supplied deliberately.
python scripts/run_experiment.py "$@"
