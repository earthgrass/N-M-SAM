from pathlib import Path
import os, re, json, hashlib, shutil, subprocess, csv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if os.name == 'nt':
    ROOT = Path('\\\\?\\'+str(ROOT))
    OUT = Path('\\\\?\\'+str(OUT))
P = ROOT/'APBR-post-training-benchmark'/'vpr_sam_research'
E = P/'experiments'/'post_training_nm_benchmark'
PACK = OUT/'packages_v2'
SKIP = {'.git','__pycache__','.pytest_cache','.venv','venv','data','runs','python_deps','upstream','node_modules','literature','preview','tmp','torch28_cpu_env','bundle_cache_quarantine','bundle_extract_verification','bundle_v3_extract_verification','server_bundle','server_bundle_v3','before'}
TEXT = {'.py','.json','.jsonl','.yaml','.yml','.sh','.md','.txt','.log','.csv','.toml','.ps1','.diff','.patch','.tex','.bib','.sty','.bst','.xml','.sha256','.gitignore','.gitattributes'}
BINARY = {'.pt','.pth','.ckpt','.npz','.npy','.gz','.zip','.bundle','.bin','.safetensors'}
SECRETS = re.compile(r'''(?i)(?:password|passwd|passphrase|api[_-]?key|access[_-]?token|auth[_-]?token|secret)\s*[=:]\s*['\"]([^'\"\r\n]{6,})['\"]|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?m:^\s*密码\s*[:：]\s*\S+)''')
SOURCE_REPOS = ['APBR-post-training-benchmark','HQ-APBR-V1','paper/narrative_revision_20261106','paper/joint_second_order_draft_20261102','HC-SAM-paper']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def files(base):
    if not base.exists(): return
    if base.is_file(): yield base; return
    for d,dirs,fs in os.walk(base):
        dirs[:]=sorted(x for x in dirs if x not in SKIP and not x.startswith(('pytest','bundle-pytest','env_')))
        for n in sorted(fs): yield Path(d)/n

def git(repo,*args):
    p=subprocess.run(['git','-C',str(repo).removeprefix('\\\\?\\'),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
    return p.stdout.strip() if p.returncode==0 else None

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if PACK.exists(): raise SystemExit('Packages already exist; do not overwrite archive')
    PACK.mkdir(parents=True)
    repos=[]
    for rel in SOURCE_REPOS:
        r=ROOT/rel
        repos.append({'workspace_path':rel,'head':git(r,'rev-parse','HEAD'),'branch':git(r,'branch','--show-current'),'status_porcelain':git(r,'status','--porcelain')})
    write_json(OUT/'source_repositories.json',repos)
    common=[]
    for path in [P/'src',P/'scripts',P/'tests',P/'configs',P/'mechanism_study']:
        common.extend(files(path))
    for path in P.iterdir():
        if path.is_file() and path.suffix not in {'.pyc'}:common.append(path)
    for d in ['baseline_adapters','benchmark','blockwise_apbr','joint_optimization','patches','configs','scripts','tests','docs']:
        common.extend(files(E/d))
    for f in E.iterdir():
        if f.is_file():common.append(f)
    art=E/'artifacts'
    available=[x.name for x in art.iterdir() if x.is_dir() and not x.name.startswith('pytest')]
    apbr={n for n in available if n.startswith(('cifar100_','apbr_full_baselines','ablation_apbr_parent','apbr_geometry_vs','apbr_seed3','apbr_nearoptimal_conditional','apbr_vpr_nearoptimal','sam_','conditional_greedy_obs_apbr'))}
    apbr.discard('sam_geo_dynamic_rtx5090_20261102_remote_copy')
    l2={n for n in available if n.startswith(('apbr_fixed_l2','apbr_l2_','dense_sam_apbr_l2'))}
    joint={n for n in available if n.startswith('apbr_gpt_joint_')}
    block={'apbr_blockwise_perturbation_cifar100_20261028'}
    obs={n for n in available if n.startswith(('apbr_gpt_mask','apbr_radius','apbr_seed0_fixedmask','apbr_sparsegpt_compensation','apbr_unified','apbr_weights_original','fixed_mask_second_order','geo_','gpt_score_conditional','group_joint_obs','sam_geo_dynamic'))}
    scopes={
      'apbr-implemented':{'roots':sorted(apbr),'inventory':'apbr'},
      'apbr-l2':{'roots':sorted(l2),'inventory':'apbr'},
      'apbr-paper':{'roots':[],'inventory':'apbr'},
      'joint':{'roots':sorted(joint),'inventory':'joint'},
      'obs-explorations':{'roots':sorted(obs),'inventory':'explorations'},
      'blockwise':{'roots':sorted(block),'inventory':'explorations'},
      'hc-sam':{'roots':[],'inventory':'explorations'},
      'research-index':{'roots':[],'inventory':None},
    }
    exclusions=[]; allrecords=[]; assets={}; stats=[]
    for name,spec in scopes.items():
        dest=PACK/name;dest.mkdir()
        selected={}
        def add(f,t): selected[str(t).replace('\\','/')]=f
        if name not in {'hc-sam','apbr-paper','research-index'}:
            for f in common:add(f,Path('project/vpr_sam_research')/f.relative_to(P))
        for n in spec['roots']:
            for f in files(art/n): add(f,Path('evidence/experimental_artifacts')/f.relative_to(art))
        extra=[]
        if name=='joint': extra=[(ROOT/'joint_seed0_copy',Path('evidence/workspace/joint_seed0_copy')),(ROOT/'paper'/'joint_second_order_draft_20261102',Path('manuscript'))]
        if name=='apbr-paper':extra=[(ROOT/'paper'/'narrative_revision_20261106',Path('manuscript'))]
        if name=='hc-sam':extra=[(ROOT/'HQ-APBR-V1',Path('project/HQ-APBR-V1')),(ROOT/'HC-SAM-paper',Path('manuscript'))]
        for base,target in extra:
            for f in files(base):
                relative=f.relative_to(base)
                if name=='hc-sam' and base.name=='HQ-APBR-V1' and relative.parts[0]=='artifacts':
                    add(f,Path('evidence/hq_artifacts')/Path(*relative.parts[1:]))
                else:add(f,target/relative)
        for f in files(OUT/'inventory'):
            if name=='research-index' or f.stem==spec['inventory']:add(f,Path('inventory')/f.name)
        records=[];omitted=[];blocked=[]
        for target,f in sorted(selected.items()):
            size=f.stat().st_size
            rel=str(f.relative_to(ROOT)).replace('\\','/')
            suffix=f.suffix.lower()
            if suffix in BINARY or '.tar.part.' in f.name:
                assets[rel]={'workspace_path':rel,'bytes':size}
                omitted.append({'workspace_path':rel,'bytes':size,'reason':'external_binary_not_uploaded'})
                continue
            if suffix in {'.pyc','.exe','.aux','.bbl','.blg','.out','.brf'} or 'known_hosts' in f.name:
                continue
            if size>2_000_000 or (suffix not in TEXT|{'.pdf','.png','.svg'} and f.name not in {'.gitignore','.gitattributes','LICENSE','README'}):
                omitted.append({'workspace_path':rel,'bytes':size,'reason':'outside_curated_small_evidence_scope'})
                continue
            if suffix in TEXT or f.name in {'.gitignore','.gitattributes'}:
                try:content=f.read_text(encoding='utf-8-sig')
                except UnicodeError:continue
                hits=list(SECRETS.finditer(content))
                if hits:
                    item={'workspace_path':rel,'bytes':size,'sha256':sha(f),'lines':[content.count('\n',0,m.start())+1 for m in hits],'reason':'possible_embedded_credential_quarantined_no_content_uploaded'}
                    blocked.append(item);exclusions.append(item);continue
            out=dest/target;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,out)
            records.append({'archive_path':target,'workspace_path':rel,'bytes':size,'sha256':sha(f),'byte_identical_copy':True})
        write_json(dest/'archive_metadata'/'SOURCE_FILE_MAP.json',records)
        write_json(dest/'archive_metadata'/'OMITTED_LOCAL_ASSETS.json',omitted)
        write_json(dest/'archive_metadata'/'CREDENTIAL_EXCLUSIONS.json',blocked)
        write_json(dest/'archive_metadata'/'SOURCE_REPOSITORIES.json',repos)
        write_json(dest/'archive_metadata'/'SCOPE.json',dict(spec,name=name,date='2026-11-10',branch='archive/'+name+'-20261110',new_experiments_run=False,original_files_modified=False,common_source_library_is_not_run_identity=True))
        stats.append({'name':name,'files_copied':len(records),'bytes_copied':sum(x['bytes'] for x in records),'omitted_files':len(omitted),'credential_exclusions':len(blocked)})
        allrecords.extend(dict(x,branch=name) for x in records)
    write_json(OUT/'candidate_secret_exclusions.json',exclusions)
    write_json(OUT/'external_assets_to_hash.json',list(assets.values()))
    write_json(OUT/'archive_stats.json',stats)
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
