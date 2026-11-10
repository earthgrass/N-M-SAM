import hashlib,json,pathlib,sys,stat
from remote import connect
root='/root/autodl-tmp/hq_apbr_v1_20261105'
local=pathlib.Path.cwd();c=connect();s=c.open_sftp()
code="""import pathlib,hashlib,json
root=pathlib.Path('/root/autodl-tmp/hq_apbr_v1_20261105')
dest=root/'evidence/run_v1_sha256.json'
files=list((root/'run_v1').rglob('*'))+[root/'pipeline_v1.log',root/'source_ec63a17.tar.gz']
result={}
for p in files:
 if not p.is_file():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 result[str(p.relative_to(root))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
if dest.exists():
 assert json.loads(dest.read_text())==result,'Remote evidence changed'
else:dest.write_text(json.dumps(result,indent=2)+'\\n')
print(len(result))
"""
_,o,e=c.exec_command("/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",timeout=120)
output=o.read().decode();errors=e.read().decode()
if o.channel.recv_exit_status():raise RuntimeError(errors)
print('Remote manifest files',output.strip(),flush=True)
dest=local/'reports/gpu_3090_v1/remote_run_v1';dest.mkdir(parents=True,exist_ok=True)
s.get(root+'/evidence/run_v1_sha256.json',str(dest/'REMOTE_SHA256.json'))
manifest=json.loads((dest/'REMOTE_SHA256.json').read_text());verified={}
for rel,meta in manifest.items():
 if rel.endswith('.tar.gz'):continue
 if rel.endswith('.pt'):
  if not rel.endswith('smoke_final.pt'):continue
  target=local/'artifacts/gpu_deploy_20261105/smoke_final.pt'
 else:target=dest/rel
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  part=target.with_suffix(target.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
  if offset>meta['bytes']:raise ValueError('Invalid partial size')
  with s.open(root+'/'+rel,'rb') as remote,part.open('ab') as f:
   remote.seek(offset)
   remote.prefetch(file_size=meta['bytes'],max_concurrent_requests=32)
   while True:
    b=remote.read(1048576)
    if not b:break
    f.write(b)
  part.replace(target)
 h=hashlib.sha256()
 with target.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 if h.hexdigest()!=meta['sha256']:raise ValueError('Hash mismatch '+rel)
 verified[rel]={'local_path':str(target),'sha256':h.hexdigest(),'verified':True}
 print('VERIFIED',rel,flush=True)
(dest/'LOCAL_VERIFICATION.json').write_text(json.dumps(verified,indent=2)+'\n')
c.close()
