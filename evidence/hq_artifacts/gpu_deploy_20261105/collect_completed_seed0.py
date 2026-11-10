"""Watch the authorized job in this active task and copy immutable completed artifacts.
No training, evaluation, retry, remote deletion, or configuration mutation.
"""
import json,time,hashlib,traceback,shlex,sys
from pathlib import Path
from remote import connect
BASE='/root/autodl-tmp/hq_apbr_v1_20261105'
RUN='seed0_full_override_v1'
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'artifacts'/'hq_apbr_seed0_full_20261105_remote_copy'
DEST.mkdir(parents=True,exist_ok=True)
STATE=DEST/'LOCAL_TRANSFER_STATUS.json'
def write(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def remote_json(c,code):
    _,out,err=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code),timeout=120)
    data=out.read();error=err.read();exitcode=out.channel.recv_exit_status()
    if exitcode:raise RuntimeError(error.decode(errors='replace'))
    return json.loads(data)
PROBE="""
import pathlib,json,os,time
r=pathlib.Path('/root/autodl-tmp/hq_apbr_v1_20261105');o=r/'seed0_full_override_v1'
s=json.loads((o/'status.json').read_text())
fail=[str(p.relative_to(r)) for p in o.glob('failure*.json')]
def active(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
h=json.loads((o/'training/history.json').read_text())
recent=h[-3:]
print(json.dumps({'status':s,'failures':fail,'training_process_active':active(4411),'comparison_process_active':active(5818),'manifest_exists':(o/'SHA256_EVIDENCE.json').exists(),'history_length':len(h),'recent_epochs':[{k:v for k,v in row.items() if k in ['epoch','steps','train_loss','train_seconds','calibration_duration','test_accessed','eps_norm_min','eps_norm_max']} for row in recent]}))
"""
try:
    c=connect();last_key=None
    while True:
        snapshot=remote_json(c,PROBE);s=snapshot['status']
        write(DEST/'LAST_REMOTE_STATUS.json',snapshot)
        if s['stage']=='failed' or snapshot['failures']:raise RuntimeError('Remote run failed: '+json.dumps(snapshot))
        if any(x.get('test_accessed') is not False for x in snapshot['recent_epochs']):raise RuntimeError('Training Test-isolation violation')
        key=(s['stage'],s.get('completed_epochs',0)//10)
        if key!=last_key:
            print(json.dumps({'event':'remote_progress','stage':s['stage'],'completed_epochs':s.get('completed_epochs'),'steps':s.get('steps'),'remote_utc':s.get('time_utc')}) ,flush=True);last_key=key
        if s['stage']=='complete' and snapshot['manifest_exists'] and not snapshot['training_process_active'] and not snapshot['comparison_process_active']:break
        if not snapshot['training_process_active'] and s['stage']!='complete':raise RuntimeError('Training process ended without completion')
        write(STATE,{'state':'WAITING_FOR_COMPLETION','snapshot':s,'no_new_training_or_test':True})
        c.close();time.sleep(30);c=connect()
    # Independently hash all selected frozen files remotely after both writers stop.
    manifest=remote_json(c,"""
import pathlib,json,hashlib
r=pathlib.Path('/root/autodl-tmp/hq_apbr_v1_20261105')
selected=['seed0_full_override_v1','seed0_full_override_v1_historical_comparison','source_90499e9','comparison_tools','run_v1','evidence']
selected+=['seed0_full_override_v1.log','seed0_full_override_v1.pid','launch_override.sh','source_90499e9.tar.gz','source_ec63a17.tar.gz','pipeline_v1.log','launch.sh']
files=[]
for name in selected:
 p=r/name
 if not p.exists():continue
 files.extend(x for x in (p.rglob('*') if p.is_dir() else [p]) if x.is_file() and '__pycache__' not in x.parts and '.git' not in x.parts)
rows=[]
for p in sorted(set(files)):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 rows.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'status':'REMOTE_FROZEN_MANIFEST','files':rows,'total_bytes':sum(x['bytes'] for x in rows)}))
""")
    old=DEST/'REMOTE_SHA256_MANIFEST.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))!=manifest:raise RuntimeError('Previous transfer manifest differs; refusing overwrite')
    write(old,manifest);print(json.dumps({'event':'transfer_start','files':len(manifest['files']),'bytes':manifest['total_bytes']}),flush=True)
    verified=[]
    for row in manifest['files']:
        rel=Path(row['path']);target=DEST/rel
        if '..' in rel.parts or rel.is_absolute():raise ValueError('Unsafe manifest path')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:raise RuntimeError('Conflicting existing local file: '+str(target))
        else:
            part=target.with_name(target.name+'.part');offset=part.stat().st_size if part.exists() else 0
            if offset>row['bytes']:raise RuntimeError('Partial file longer than remote')
            sftp=c.open_sftp();sftp.get_channel().settimeout(60)
            with sftp.open(BASE+'/'+row['path'],'rb') as src,part.open('ab') as dst:
                src.seek(offset);src.prefetch(file_size=row['bytes'],max_concurrent_requests=16)
                while offset<row['bytes']:
                    data=src.read(min(1024*1024,row['bytes']-offset))
                    if not data:raise EOFError(row['path'])
                    dst.write(data);offset+=len(data)
            sftp.close()
            if sha(part)!=row['sha256']:raise RuntimeError('SHA256 mismatch: '+row['path'])
            part.rename(target)
        verified.append(row)
        write(STATE,{'state':'TRANSFERRING','verified_files':len(verified),'total_files':len(manifest['files']),'last_file':row['path'],'no_remote_deletion':True})
        if row['bytes']>10000000 or len(verified)%50==0:print(json.dumps({'event':'file_verified','file':row['path'],'count':len(verified)}),flush=True)
    # Check the job's original frozen evidence manifest against independently downloaded files.
    original=json.loads((DEST/RUN/'SHA256_EVIDENCE.json').read_text())
    for rel,expected in original.items():
        p=DEST/RUN/rel
        if not p.is_file() or sha(p)!=expected:raise RuntimeError('Original manifest mismatch: '+rel)
    result={'status':'LOCAL_TRANSFER_SHA256_PASS','verified_files':len(verified),'verified_bytes':manifest['total_bytes'],'original_run_manifest_files':len(original),'local_root':str(DEST),'remote_files_deleted':False,'test_repeated':False}
    write(DEST/'LOCAL_SHA256_VERIFICATION.json',result);write(STATE,{'state':'COMPLETE',**result})
    print(json.dumps(result),flush=True);c.close()
except BaseException as e:
    write(STATE,{'state':'FAILED','error':str(e),'traceback':traceback.format_exc(),'automatic_training_retry':False})
    print(traceback.format_exc(),flush=True);sys.exit(1)
