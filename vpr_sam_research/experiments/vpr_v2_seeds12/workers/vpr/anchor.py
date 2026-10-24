"""Read-only verification of the just-completed paired SAM trajectory."""
import json
from pathlib import Path
import torch
from state import sha_file,validate_checkpoint,source_freeze

def verify_anchor(config):
    root=Path(config['anchor']['root']);run=root/config['anchor']['run'];path=run/'best_test.pt'
    status=json.loads((run/'status.json').read_text());audit=json.loads((run/'FINAL_AUDIT.json').read_text())
    hashes=json.loads((run/'checkpoint_hashes.json').read_text());digest=sha_file(path)
    if status['status']!='completed' or audit['status']!='PASS' or status['steps']!=78000:
        raise ValueError('Paired SAM is not completely audited')
    if hashes['best_test.pt']!=digest or audit['selected']['best_test.pt']['sha256']!=digest:
        raise ValueError('Paired SAM publication digest mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False);validate_checkpoint(state)
    identity=state['identity'];frozen=source_freeze(Path(__file__).resolve().parent)
    if identity['commit']!=frozen['commit'] or identity['seed']!=config['seed']:
        raise ValueError('Paired SAM commit/seed mismatch')
    excluded={'experiment','output','minimum_free_disk_bytes'}
    differences={k:[v,config.get(k)] for k,v in identity['config'].items() if k not in excluded and config.get(k)!=v}
    if differences:raise ValueError('SAM/VPR common protocol mismatch: '+str(differences))
    history=[json.loads(s) for s in (run/'epoch_metrics.jsonl').read_text().splitlines()]
    if len(history)!=200 or history[-1]['steps']!=78000:raise ValueError('SAM epoch coverage')
    best=max(history,key=lambda r:(r['test_accuracy'],-r['epoch']))
    if state['epoch']!=best['epoch'] or state['history'][-1]!=best:raise ValueError('SAM Best Test selection mismatch')
    return state,history,{'path':str(path),'sha256':digest,'producer_commit':identity['commit'],
        'seed':config['seed'],'epoch':state['epoch'],'dense_accuracy':best['test_accuracy'],
        'dense_loss':best['test_loss'],'initialization_hash':identity['initialization_hash'],
        'data_manifest_sha256':identity['data_manifest_sha256'],'common_config_differences':differences,
        'history_sha256':sha_file(run/'epoch_metrics.jsonl')}
