"""Check local snapshot hashes and all registry selectors, with no remote or Test access."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'evidence/source_snapshot_manifest.json').read_text(encoding='utf-8'))
snapshots=0
for row in manifest:
    if 'snapshot' in row:
        p=ROOT/row['snapshot'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];snapshots+=1
registry=json.loads((ROOT/'numeric_evidence.json').read_text(encoding='utf-8'))
for row in registry['entries']:
    value=json.loads((ROOT/row['source']).read_text(encoding='utf-8'))
    for key in row['selector']['pointer'].lstrip('/').split('/'):
        value=value[int(key)] if isinstance(value,list) else value[key]
    assert math.isclose(value,row['value'],rel_tol=0,abs_tol=row.get('tolerance',0)),row
history=json.loads((ROOT/'evidence/smoke_history.json').read_text(encoding='utf-8'))
profile=json.loads((ROOT/'evidence/performance_derived.json').read_text(encoding='utf-8'))
train=sum(x['train_seconds'] for x in history)/len(history);cal=sum(x['calibration_duration'] for x in history)/len(history)
assert abs(train-profile['train_seconds_mean'])<1e-10
assert abs(cal-profile['calibration_seconds_mean'])<1e-10
assert abs(100*cal/train-profile['calibration_to_train_percent'])<1e-10
report={'status':'PASS','snapshot_hashes_verified':snapshots,'numeric_registry_selectors_verified':len(registry['entries']),
        'smoke_aggregates_recomputed':True,'scope':'Local copied evidence and deterministic aggregates; not final model validation'}
(ROOT/'review/evidence_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
