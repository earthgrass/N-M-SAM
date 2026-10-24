"""Read-only V1 augmentation reference, executed using ORIGINAL V1 runtime.

Writes only a specified file in the new workspace; never loads a checkpoint.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch
from vpr_sam.data import build_data,loader
from vpr_sam.repro import source_hashes,atomic_json

if __name__=='__main__':
    old=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
    if target.is_relative_to(old):raise ValueError('Must write into the NEW workspace only')
    gates=json.loads((old/'runs/main200_seeds012_v1/control/gates.json').read_text())
    if source_hashes(old)!=gates['source_hashes']:raise RuntimeError('Original V1 provenance changed')
    torch.set_num_threads(4)
    c=json.loads((old/'configs/approved_main200/vpr_seed0.json').read_text())
    datasets,manifest=build_data(c,old)
    values={}
    for epoch in range(200):
        x,y,ids=next(iter(loader(datasets['train'],128,0,epoch,workers=0,shuffle=True)))
        values[str(epoch)]=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()+ids.numpy().tobytes()).hexdigest()
    atomic_json(target,dict(v1_commit=gates['commit'],dataset_fingerprint=manifest['fingerprint'],
        source_hashes=source_hashes(old),augmentation_first_batch_sha256=values))
    print('V1 original-runtime augmentation reference complete; no checkpoint loaded')
