#!/usr/bin/env bash
set -uo pipefail
for pair in sam_float64 vpr_float64 vpr_float32; do
 method=${pair%_*}; dtype=${pair#*_}
 .venv/bin/python -u - "$method" "$dtype" <<'PY'
import runpy,sys
from pathlib import Path
m=runpy.run_path('scripts/server_validation.py'); root=Path.cwd(); before=m['source_hashes'](root)
m['diagnostic_run'](root,root/'artifacts/server_acceptance','cuda:0',sys.argv[1],sys.argv[2],12,1e-7,'performance_'+sys.argv[1]+'_'+sys.argv[2])
assert before == m['source_hashes'](root)
PY
 status=$?
 if [ "$status" != 0 ]; then
  if [ "$pair" != vpr_float32 ]; then exit "$status"; fi
  .venv/bin/python - <<'PY'
import json
x=json.load(open('artifacts/server_acceptance/performance_vpr_float32/failure.json'))
assert x['type']=='FloatingPointError' and 'log-volume drift' in x['message'], x
print('FP32_VOLUME_GATE_FAILED; STOP_FP32; independent FP64 diagnostics may proceed',flush=True)
PY
  if [ "$?" != 0 ]; then exit 1; fi
 fi
done
for method in sam vpr; do
 .venv/bin/python -u artifacts/server_acceptance/uninstrumented.py "$method" || exit "$?"
done
.venv/bin/python - <<'PY'
import json,pathlib
p=pathlib.Path('artifacts/server_acceptance'); rows=[json.loads((p/f'uninstrumented_{m}.json').read_text()) for m in ('sam','vpr')]
assert rows[0]['initialization_hash']==rows[1]['initialization_hash'] and rows[0]['order_hash']==rows[1]['order_hash']
(p/'uninstrumented_performance.json').write_text(json.dumps({'independent_process_per_method':True,'initialization_order_matched':True,'results':rows},indent=2))
PY
if [ "$?" != 0 ]; then exit 1; fi
.venv/bin/python -u scripts/server_validation.py gradients --device cuda:0
