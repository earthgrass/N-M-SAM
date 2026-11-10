import pathlib,json,hashlib,shutil,subprocess,time,traceback,sys
from remote import connect
ROOT=pathlib.Path(__file__).resolve().parents[2]
DEST=ROOT/'artifacts/hq_apbr_seed0_full_20261105_remote_copy'
BASE='/root/autodl-tmp/hq_apbr_v1_20261105'
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
try:
 m=json.loads((DEST/'REMOTE_SHA256_MANIFEST.json').read_text(encoding='utf-8'));cache={};todo=[];copied=0
 for r in m['files']:
  f=DEST/r['path']
  if f.exists():
   if sha(f)!=r['sha256']:raise ValueError('Existing file mismatch '+r['path'])
   cache[r['sha256']]=f
 for r in m['files']:
  f=DEST/r['path'];f.parent.mkdir(parents=True,exist_ok=True)
  if f.exists():continue
  if r['sha256'] in cache:
   shutil.copyfile(cache[r['sha256']],f);assert sha(f)==r['sha256'];copied+=1
   part=f.with_name(f.name+'.part')
   if part.exists():part.unlink()
  else:todo.append(r)
 scratch=pathlib.Path.home()/'.codex/tmp/hq_seed0_native_transfer_20261105';scratch.mkdir(parents=True,exist_ok=True)
 c=connect();key=c.get_transport().get_remote_server_key();c.close()
 known=scratch/'known_hosts';known.write_text('[connect.nmb2.seetacloud.com]:25375 '+key.get_name()+' '+key.get_base64()+'\n',encoding='ascii')
 batch=scratch/'batch.txt';batch.write_text('\n'.join('reget "'+BASE+'/'+r['path']+'" "'+r['path']+'.part"' for r in todo)+'\n',encoding='utf-8')
 write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'NATIVE_SFTP_RESUME','missing_files':len(todo),'copied_verified_duplicates':copied,'no_remote_deletion':True})
 print(json.dumps({'event':'native_resume','files':len(todo),'copied_duplicates':copied}),flush=True)
 cmd=['sftp','-q','-C','-B','131072','-R','64','-P','25375','-i',str(pathlib.Path.home()/'.ssh/codex_apbr_seed245_nopass_20261103'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-b',str(batch),'root@connect.nmb2.seetacloud.com']
 with (DEST/'NATIVE_SFTP_TRANSFER.log').open('wb') as log:
  result=subprocess.run(cmd,cwd=DEST,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Native SFTP failed; see log, resumable partials preserved')
 for i,r in enumerate(todo):
  f=DEST/r['path'];part=f.with_name(f.name+'.part')
  if part.stat().st_size!=r['bytes'] or sha(part)!=r['sha256']:raise ValueError('Downloaded hash mismatch '+r['path'])
  part.rename(f)
  if r['bytes']>10000000:print(json.dumps({'event':'sha_verified','file':r['path']}),flush=True)
 for r in m['files']:
  f=DEST/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 original=json.loads((DEST/'seed0_full_override_v1/SHA256_EVIDENCE.json').read_text())
 for rel,expected in original.items():assert sha(DEST/'seed0_full_override_v1'/rel)==expected,rel
 out={'status':'LOCAL_TRANSFER_SHA256_PASS','verified_files':len(m['files']),'verified_bytes':m['total_bytes'],'original_run_manifest_files':len(original),'local_root':str(DEST),'remote_files_deleted':False,'test_repeated':False,'transport':'OpenSSH SFTP resume; Paramiko stall evidence retained'}
 write(DEST/'LOCAL_SHA256_VERIFICATION.json',out);write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'COMPLETE',**out});print(json.dumps(out),flush=True)
except BaseException as e:
 write(DEST/'LOCAL_TRANSFER_STATUS.json',{'state':'FAILED','error':str(e),'traceback':traceback.format_exc(),'no_remote_mutation':True});print(traceback.format_exc(),flush=True);sys.exit(1)
