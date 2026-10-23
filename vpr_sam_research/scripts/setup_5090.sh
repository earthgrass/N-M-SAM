#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
mkdir -p artifacts
.venv/bin/python -m pip freeze > artifacts/server-pip-freeze.txt
.venv/bin/python scripts/preflight.py --require-5090 > artifacts/server-preflight.json
# Environment only. No training, downloading or job submission.
