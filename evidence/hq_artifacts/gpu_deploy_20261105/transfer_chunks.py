"""Bounded compressed SSH streams, independently hashed; no dataset or experiment execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,shutil,shlex,zlib,time,traceback,sys
from remote import connect
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'artifacts/hq_apbr_seed0_full_20261105_remote_copy';BASE='/root/autodl-tmp/hq_apbr_v1_20261105/'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def download(r):
 target=DEST/r['path'];target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.part')
 offset=part.stat().st_size if part.exists() else 0
 if offset>r['bytes']:raise ValueError('Oversized partial')
 c=connect();c.get_transport().set_keepalive(15)
 try:
  with part.open('ab') as f:
   while offset<r['bytes']:
    n=min(1048576,r['bytes']-offset)
    code='import sys,zlib;f=open('+repr(BASE+r['path'])+",'rb');f.seek("+str(offset)+');sys.stdout.buffer.write(zlib.compress(f.read('+str(n)+'),1))'
    _,out,err=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code),timeout=45)
    packed=out.read();errors=err.read();status=out.channel.recv_exit_status()
    if status:raise RuntimeError(errors.decode(errors='replace'))
    data=zlib.decompress(packed)
    if len(data)!=n:raise ValueError('Short chunk '+r['path'])
    f.write(data);f.flush();offset+=n
    if r['bytes']>10000000 and offset//33554432 != (offset-n)//33554432:
     print(json.dumps({'event':'chunk_progress','file':r['path'],'bytes':offset,'total':r['bytes']}),flush=True)
 finally:c.close()
 if sha(part)!=r['sha256']:raise ValueError('SHA256 mismatch '+r['path'])
 part.rename(target);return r
try:
 m=json.loads((DEST/'REMOTE_SHA256_MANIFEST.json').read_text(encoding='utf-8'));cache={};groups={};verified=0
 for r in m['files']:
  p=DEST/r['path'];part=p.with_name(p.name+'.part')
  if not p.exists() and part.exists() and part.stat().st_size==r['bytes'] and sha(part)==r['sha256']:part.rename(p)
  if p.exists():
   if sha(p)!=r['sha256']:raise ValueError('Existing file mismatch '+r['path'])
   cache[r['sha256']]=p;verified+=1
 for r in m['files']:
  p=DEST/r['path']
  if p.exists():continue
  if r['sha256'] in cache:
   p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cache[r['sha256']],p);verified+=1
  else:groups.setdefault(r['sha256'],[]).append(r)
 representatives=[v[0] for v in groups.values()]
 representatives.sort(key=lambda r:r['bytes'],reverse=True)
 print(json.dumps({'event':'bounded_transfer_start','already_verified':verified,'unique_remaining':len(representatives),'workers':8}),flush=True)
 with ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(download,r):r for r in representatives}
  for future in as_completed(futures):
   r=future.result();verified+=1
   for alias in groups[r['sha256']][1:]:
    p=DEST/alias['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/r['path'],p);verified+=1
   write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'BOUNDED_SSH_TRANSFERRING','verified_files':verified,'total_files':len(m['files']),'last_verified':r['path']})
   if r['bytes']>10000000 or verified%50==0:print(json.dumps({'event':'file_verified','file':r['path'],'verified':verified}),flush=True)
 for r in m['files']:
  p=DEST/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
  part=p.with_name(p.name+'.part')
  if part.exists():part.unlink()
 original=json.loads((DEST/'seed0_full_override_v1/SHA256_EVIDENCE.json').read_text())
 for rel,expected in original.items():assert sha(DEST/'seed0_full_override_v1'/rel)==expected,rel
 result={'status':'LOCAL_TRANSFER_SHA256_PASS','verified_files':len(m['files']),'verified_bytes':m['total_bytes'],'original_run_manifest_files':len(original),'local_root':str(DEST),'remote_files_deleted':False,'test_repeated':False,'transport':'bounded compressed SSH channels; previous stalls retained'}
 write(DEST/'LOCAL_SHA256_VERIFICATION.json',result);write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'COMPLETE',**result});print(json.dumps(result),flush=True)
except BaseException as e:
 write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'FAILED','error':str(e),'traceback':traceback.format_exc(),'remote_data_changed':False});print(traceback.format_exc(),flush=True);sys.exit(1)
