"""Environment-native official LR reference; strict equality, never tolerances."""
import ast
import json
import math
import platform
from pathlib import Path

import torch

from scheduler import CosineLRscheduler
from state import atomic_json,environment,object_sha,sha_file


def official_class(root):
    path=Path(root)/'reference/ssam2022/solver/lr_scheduler.py'
    manifest=json.loads((Path(root)/'reference/UPSTREAM_SHA256.json').read_text())
    if sha_file(path)!=manifest['ssam2022']['files']['solver/lr_scheduler.py']:
        raise ValueError('Official scheduler source checksum failed')
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name in ('LRscheduler','CosineLRscheduler'):
            node.decorator_list=[]  # Remove registry/config plumbing only.
            for method in node.body:
                if isinstance(method,ast.FunctionDef):
                    method.decorator_list=[d for d in method.decorator_list
                        if not (isinstance(d,ast.Name) and d.id=='configurable')]
            nodes.append(node)
    namespace={'math':math,'optim':torch.optim}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['CosineLRscheduler']


def sequence(cls,config):
    optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))],lr=config['learning_rate'],
        momentum=config['momentum'],weight_decay=config['weight_decay'],nesterov=config['nesterov'])
    scheduler=cls(optimizer,T_max=config['epochs'],eta_min=config['eta_min'],
        warmup_epoch=config['warmup_epoch'],warmup_init_lr=config['warmup_init_lr'])
    rows=[]
    for epoch in range(config['epochs']):
        used=optimizer.param_groups[0]['lr']
        scheduler.step(epoch)  # Official epoch-end zero-based call, unchanged.
        rows.append({'epoch':epoch+1,'used':used,'after_epoch':optimizer.param_groups[0]['lr']})
    return rows


def strict_compare(reference,project,epochs):
    if len(reference)!=epochs or len(project)!=epochs:
        raise ValueError('LR sequence length mismatch')
    fields=[]
    for epoch,(r,p) in enumerate(zip(reference,project),1):
        if set(r)!={'epoch','used','after_epoch'} or set(p)!=set(r) or r['epoch']!=epoch or p['epoch']!=epoch:
            raise ValueError('LR sequence schema/order mismatch')
        for key in ('used','after_epoch'):
            a,b=r[key],p[key]
            passed=math.isfinite(a) and math.isfinite(b) and a==b
            fields.append({'epoch':epoch,'field':key,'reference':a,'project':b,
                           'exactly_equal':passed,'absolute_difference':abs(a-b)})
    count=sum(not f['exactly_equal'] for f in fields)
    return {'status':'PASS' if count==0 else 'FAIL','comparison':'strict numeric equality; no tolerance',
            'epochs':epochs,'compared_fields':len(fields),'differing_fields':count,'fields':fields}


def generate(root,config,frozen):
    root=Path(root); out=root/'artifacts/lr_reference'
    if out.exists():
        raise FileExistsError('LR reference directory already exists; preserve original evidence')
    reference=sequence(official_class(root),config)
    project=sequence(CosineLRscheduler,config)
    comparison=strict_compare(reference,project,config['epochs'])
    atomic_json(out/'official_reference.json',reference)
    atomic_json(out/'project_sequence.json',project)
    atomic_json(out/'comparison.json',comparison)
    windows=root/'reference/LR_EXPECTED.json'
    cross=strict_compare(json.loads(windows.read_text()),reference,config['epochs'])
    cross['interpretation']='Cross-platform evidence only; environment-native official/project strict comparison is the launch gate'
    atomic_json(out/'windows_vs_native.json',cross)
    atomic_json(out/'resolved_config.json',config)
    metadata={'commit':frozen['commit'],'config_sha256':object_sha(config),
        'config_file_sha256':sha_file(root/'config.yaml'),'environment':environment(),
        'libc':platform.libc_ver(),'official_scheduler_sha256':sha_file(root/'reference/ssam2022/solver/lr_scheduler.py'),
        'project_scheduler_sha256':sha_file(root/'scheduler.py'),'windows_reference_sha256':sha_file(windows),
        'official_reference_sha256':sha_file(out/'official_reference.json'),
        'project_sequence_sha256':sha_file(out/'project_sequence.json'),
        'comparison_sha256':sha_file(out/'comparison.json'),'windows_vs_native_sha256':sha_file(out/'windows_vs_native.json'),
        'status':comparison['status'],'differing_native_fields':comparison['differing_fields'],
        'differing_cross_platform_fields':cross['differing_fields']}
    atomic_json(out/'metadata.json',metadata)
    if comparison['status']!='PASS':
        raise ValueError('Environment-native official/project LR strict comparison failed; see comparison.json')
    return metadata


def verify(root,config,frozen):
    root=Path(root); out=root/'artifacts/lr_reference'
    meta=json.loads((out/'metadata.json').read_text())
    if meta['status']!='PASS' or meta['commit']!=frozen['commit'] or meta['config_sha256']!=object_sha(config):
        raise ValueError('Native LR reference provenance mismatch')
    if meta['config_file_sha256']!=sha_file(root/'config.yaml') or meta['environment']!=environment():
        raise ValueError('Native LR reference config/runtime changed')
    for name,key in [('official_reference.json','official_reference_sha256'),('project_sequence.json','project_sequence_sha256'),
                     ('comparison.json','comparison_sha256'),('windows_vs_native.json','windows_vs_native_sha256')]:
        if sha_file(out/name)!=meta[key]: raise ValueError('Native LR reference digest mismatch: '+name)
    reference=json.loads((out/'official_reference.json').read_text())
    if strict_compare(reference,sequence(CosineLRscheduler,config),config['epochs'])['status']!='PASS':
        raise ValueError('Project LR execution no longer matches native reference')
    return meta,reference
