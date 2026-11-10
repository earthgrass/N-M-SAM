#!/bin/bash
set -euo pipefail
ROOT=/root/autodl-tmp/hq_apbr_v1_20261105
test ! -e "$ROOT/source_90499e9"
mkdir "$ROOT/source_90499e9"
tar -xzf "$ROOT/source_90499e9.tar.gz" -C "$ROOT/source_90499e9"
cd "$ROOT/source_90499e9"
/root/miniconda3/bin/python - <<'PY'
import hashlib,json,pathlib
identity=json.loads(pathlib.Path('SOURCE_IDENTITY.json').read_text())
for p,h in identity['files'].items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
print('VERIFIED_SOURCE_FILES',len(identity['files']))
PY
test ! -e "$ROOT/seed0_full_override_v1"
test ! -e "$ROOT/seed0_full_override_v1.log"
nohup /root/miniconda3/bin/python -u scripts/run_authorized_seed0.py --config configs/hq_apbr_v1_seed0_gpu_v1.json --archive "$ROOT/data/cifar-100-python.tar.gz" --prior-run "$ROOT/run_v1" --approval approvals/seed0_performance_exception_20261105.json --output "$ROOT/seed0_full_override_v1" > "$ROOT/seed0_full_override_v1.log" 2>&1 < /dev/null &
echo $! > "$ROOT/seed0_full_override_v1.pid"
cat "$ROOT/seed0_full_override_v1.pid"
