from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,time,os
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
if os.name=='nt': ROOT=Path('\\\\?\\'+str(ROOT));OUT=Path('\\\\?\\'+str(OUT))
items=json.loads((OUT/'external_assets_to_hash.json').read_text(encoding='utf-8'))
by_path={x['workspace_path']:x for x in items}
for n in ['apbr_original_sparsegpt_ckpts','fixedmask_secondorder_inputs_20261028','seed3_dense_transfer_20261029']:
    for d,dirs,fs in os.walk(ROOT/n):
        dirs[:]=[x for x in dirs if x not in {'.git','__pycache__','.venv'}]
        for fn in fs:
            p=Path(d)/fn
            if p.suffix.lower() in {'.pt','.pth','.ckpt','.npz','.npy'}:
                rel=p.relative_to(ROOT).as_posix();by_path.setdefault(rel,{'workspace_path':rel,'bytes':p.stat().st_size})
def work(x):
    p=ROOT/x['workspace_path'];before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    after=p.stat()
    return dict(x,sha256=h.hexdigest(),local_read_verified=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,uploaded_to_github=False)
start=time.time();results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for i,x in enumerate(pool.map(work,by_path.values()),1):
        results.append(x)
        if i%100==0:print('hashed',i,flush=True)
manifest={'date':'2026-11-10','scope':'Selected research checkpoint/bundle assets; datasets excluded; bytes remain local, not a remote backup','assets':results,'count':len(results),'bytes':sum(x['bytes'] for x in results),'elapsed_seconds':time.time()-start}
(OUT/'EXTERNAL_ASSETS_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='assets'},ensure_ascii=True),flush=True)
