"""Review genuine completed smoke evidence without pretending new calibration ran."""
import json
from pathlib import Path
from .common import sha_file,object_sha

# Numerical/coordinate/model/artifact components must match the actual CUDA run.
CORE=('obc.py','pipeline.py','recovery.py','nm.py','model.py','evaluation.py',
      'artifacts.py','upstream.py','common.py','data.py')

def verify(config,current_source):
    c=config['execution']['smoke_lineage']
    report_path=Path(c['report']);audit_path=Path(c['audit'])
    if sha_file(report_path)!=c['report_sha256'] or sha_file(audit_path)!=c['audit_sha256']:
        raise ValueError('Actual CUDA evidence byte digest mismatch')
    report=json.loads(report_path.read_text());audit=json.loads(audit_path.read_text())
    identity=report['identity'];old=identity['source']
    if identity['config_sha256']!=current_source['files']['configs/obc_cuda_smoke.yaml']:
        raise ValueError('Actual smoke configuration changed')
    if report['status']!='CUDA_TECHNICAL_SMOKE_PASS' or audit['status']!='PASS':
        raise ValueError('Actual CUDA smoke/audit not PASS')
    if not all(v['pass'] for v in audit['checks'].values()) or audit['report']!=report:
        raise ValueError('Actual smoke independent audit mismatch')
    if audit['hashes']['SMOKE_REPORT.json']!=c['report_sha256']:
        raise ValueError('Actual smoke audit/report SHA mismatch')
    if identity['commit']!=c['commit'] or object_sha(old['files'])!=old['sha256']:
        raise ValueError('Actual smoke source identity mismatch')
    if identity['gpu_uuid']!=config['gpu_uuid'] or identity['environment']['cuda_available'] is not True:
        raise ValueError('Actual smoke device mismatch')
    if report['partial_layer']!='layer1.0.conv1' or report['recovery']['steps']!=2 or report['recovery']['optimizer_reload']!='EXACT_MATCH':
        raise ValueError('Incomplete technical smoke')
    if audit['checks']['parent_intact']['pass'] is not True:
        raise ValueError('Smoke parent audit failed')
    for name in CORE:
        key='benchmark/'+name
        if old['files'][key]!=current_source['files'][key]:
            raise ValueError('CUDA-tested core changed: '+key)
    return {'status':'PASS','scope':'single-layer CUDA; full-model resources pending pilot',
            'smoke_commit':identity['commit'],'calibration':'new approved1024/10 augmentation; not tested by32-image smoke'}
