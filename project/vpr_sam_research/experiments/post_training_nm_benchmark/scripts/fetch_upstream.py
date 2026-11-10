"""Materialize exact official commits; never fall back to branch tip."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.upstream import source

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',choices=['OBC','SNOWS','CAP','MOONSHOT','UniPTS'],default='OBC');a=p.parse_args()
    lock=json.loads((ROOT/'configs/upstream_lock.json').read_text(encoding='utf-8'))[a.repo]
    target=ROOT/'artifacts/upstream'/a.repo;target.parent.mkdir(parents=True,exist_ok=True)
    def git(*args):subprocess.run(['git','-c','core.autocrlf=false',*args],check=True)
    if not target.exists():
        git('clone','--no-checkout',lock['origin'],str(target))
        git('-C',str(target),'checkout','--detach',lock['commit'])
    head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    if head!=lock['commit']:raise ValueError('Existing upstream HEAD differs; preserve it and inspect manually')
    for name in lock['files']:source(a.repo,name)
    print(a.repo,head,'locked file bytes verified')

if __name__=='__main__':main()
