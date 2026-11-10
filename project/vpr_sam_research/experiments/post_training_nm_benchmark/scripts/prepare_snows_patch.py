"""Create independent review diff, plus fresh official-ref evidence; not a run."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.snows_patch import candidate_source
from benchmark.common import atomic_json,sha_file

def main():
    text,patch=candidate_source();folder=ROOT/'patches';folder.mkdir(exist_ok=True)
    target=folder/'snows_minimal_diagnostic.patch'
    if target.exists():raise FileExistsError('Preserve reviewed patch; create a separate revision')
    target.write_text(patch,encoding='utf-8',newline='')
    refs=subprocess.check_output(['git','ls-remote','https://github.com/mazumder-lab/SNOWS.git'],text=True)
    history=subprocess.check_output(['git','-C',str(ROOT/'artifacts/upstream/SNOWS'),'log','-8','--format=%H %s'],text=True)
    folder=ROOT/'artifacts/smoke_readiness_20261025';folder.mkdir(exist_ok=False)
    atomic_json(folder/'official_snows_check.json',{'checked_date_Asia_Shanghai':'2026-10-25',
        'repository':'https://github.com/mazumder-lab/SNOWS','advertised_refs':refs.splitlines(),
        'recent_history':history.splitlines(),'no_newer_advertised_ref_than_pin':all('750497fe13454253056307956a56d581fedf1128' in line for line in refs.splitlines()),
        'patch_sha256':sha_file(target),'official_checkout_modified':False,
        'scope':'advertised refs, not a claim about private/unpublished fixes'})
    print('Independent diagnostic patch and official-ref evidence created; full SNOWS remains BLOCKED.')

if __name__=='__main__':main()
