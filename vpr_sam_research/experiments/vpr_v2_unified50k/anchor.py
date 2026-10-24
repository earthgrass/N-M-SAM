"""Read-only import checks; never silently relabel an old checkpoint."""
import json
from pathlib import Path
import torch
from state import object_sha,sha_file,validate_checkpoint


def verify_anchor(config):
    source=config['anchor'];root=Path(source['root']);run=root/source['run'];path=run/source['checkpoint']
    if sha_file(path)!=source['sha256']: raise ValueError('SAM anchor file hash')
    state=torch.load(path,map_location='cpu',weights_only=False);validate_checkpoint(state)
    identity=state['identity']
    if identity['commit']!=source['commit'] or state['epoch']!=193 or state['history'][-1]['test_accuracy']!=.9677:
        raise ValueError('Wrong SAM anchor')
    excluded={'experiment','output','minimum_free_disk_bytes'}
    differences={k:[v,config.get(k)] for k,v in identity['config'].items() if k not in excluded and config.get(k)!=v}
    if differences: raise ValueError('SAM/VPR common protocol mismatch: '+str(differences))
    history=[json.loads(s) for s in (run/'epoch_metrics.jsonl').read_text().splitlines()]
    if len(history)!=200 or history[-1]['steps']!=78000: raise ValueError('Anchor complete trajectory missing')
    return state,history,{'path':str(path),'sha256':source['sha256'],'producer_commit':source['commit'],
          'epoch':state['epoch'],'dense_accuracy':state['history'][-1]['test_accuracy'],
          'dense_loss':state['history'][-1]['test_loss'],'initialization_hash':identity['initialization_hash'],
          'data_manifest_sha256':identity['data_manifest_sha256'],'common_config_differences':differences,
          'history_sha256':sha_file(run/'epoch_metrics.jsonl')}
