"""Run with ORIGINAL f742733 runtime to verify and export immutable inputs.

No checkpoint is silently re-labelled as having the new source snapshot.
The new importer separately creates a derived artifact and preserves this chain.
"""
import argparse,json,subprocess
from pathlib import Path
import torch
import vpr_sam
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.repro import atomic_json,atomic_save,file_hash,tensor_hash,source_hashes


def original_files(root):
    paths=list((root/'runs/vpr_l2_seed0_v2').rglob('*'))
    paths+=list((root/'artifacts/vpr_l2').rglob('*'))
    paths+=list((root/'artifacts/geometry_guided_mask_seed0').rglob('*'))
    paths+=list(root.glob('VPR_L2_*.md'))
    return {str(p.relative_to(root)):file_hash(p) for p in paths if p.is_file()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    root=a.original.resolve();out=a.output.resolve()
    if out.is_relative_to(root) or out.exists():raise RuntimeError('Export must be new and outside original workspace')
    if not Path(vpr_sam.__file__).resolve().is_relative_to(root/'src'):raise RuntimeError('Use ORIGINAL runtime for original checkpoint verification')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!='f7427339de6d6589f53b0191068448cf642901b2':raise RuntimeError('Wrong original producer')
    gate=json.loads((root/'artifacts/vpr_l2/control/gates.json').read_text())
    if source_hashes(root)!=gate['source_hashes']:raise RuntimeError('Original source identity mismatch')
    before=original_files(root)
    base=root/'runs/vpr_l2_seed0_v2/fixed_l2_seed0';diag=root/'artifacts/geometry_guided_mask_seed0'
    audit=json.loads((diag/'final_audit.json').read_text())
    if audit['status']!='PASS':raise RuntimeError('Three-mask audit missing')
    final,fp=read_checkpoint(base/'final.pt',root)
    if final['epoch']!=200 or final['stepper']['steps']!=70400 or file_hash(fp)!=audit['final_checkpoint_sha256']:
        raise RuntimeError('Dense final identity mismatch')
    masks_path=diag/'masks.pt'
    if file_hash(masks_path)!=audit['masks_file_sha256']:raise RuntimeError('Frozen mask file mismatch')
    masks=torch.load(masks_path,map_location='cpu',weights_only=True)['geometry_guided']
    guided=next(r for r in audit['methods'] if r['method']=='geometry_guided')
    if tensor_hash(masks)!=guided['mask_sha256']:raise RuntimeError('Not the frozen Geometry-guided mask')
    geom=final['stepper']['geometry']
    if geom['method']!='fixed_l2' or not all(s.dtype==torch.float64 for s in geom['logs']):raise RuntimeError('Geometry origin invalid')
    if tensor_hash({name:s for (name,_),s in zip(geom['mapping'],geom['logs'])})!=audit['geometry_sha256']:
        raise RuntimeError('Geometry coordinate identity mismatch')
    magpath=base/'pruning/n2_m4_exclude_first_last_magnitude_dense_final/recovery/final.pt'
    mag,mp=read_checkpoint(magpath,root)
    mag_hard,hp=read_checkpoint(magpath.parent.parent/'pruned.pt',root)
    mag_evidence=json.loads((root/'artifacts/vpr_l2/recovery_extension/final_audit.json').read_text())
    if mag['epoch']!=20 or mag['stepper']['steps']!=7040 or file_hash(mp)!=mag_evidence['final_recovery_sha256']:
        raise RuntimeError('Magnitude recovery baseline invalid')
    if tensor_hash(masks)==tensor_hash(mag_hard['masks']):raise RuntimeError('Guided mask unexpectedly replaced by Magnitude')
    payload=dict(schema='verified-guided-recovery-inputs-v1',model=final['model'],masks=masks,
        dense_config=final['config'],manifest=final['manifest'],initialization_hash=final['initialization_hash'],
        original_environment=final['environment'],dense_metrics=final['epoch_rows'][-1]['validation'],
        guided_raw=guided['raw'],guided_bn=guided['bn_calibrated'],
        guided_raw_parameter_sha256=guided['bn_evidence']['parameter_sha256'],
        magnitude_config=mag['config'],magnitude_epochs=mag['epoch_rows'],magnitude_order_hash=mag['order_hash'],
        magnitude_raw=mag_hard['report']['hard_validation'],
        magnitude_bn=next(r for r in audit['methods'] if r['method']=='magnitude')['bn_calibrated'],
        producer_commit=commit,dense_checkpoint_sha256=file_hash(fp),guided_masks_file_sha256=file_hash(masks_path),
        guided_mask_sha256=tensor_hash(masks),diagnostic_audit_sha256=file_hash(diag/'final_audit.json'),
        diagnostic_commit=audit['diagnostic_commit'],geometry_sha256=audit['geometry_sha256'],
        magnitude_recovery_sha256=file_hash(mp),magnitude_mask_sha256=tensor_hash(mag_hard['masks']),
        original_sources=gate['source_hashes'],original_files_before=before,original_project=str(root))
    atomic_save(out,payload)
    atomic_json(out.with_suffix('.receipt.json'),dict(status='PASS',input_sha256=file_hash(out),dense_sha256=file_hash(fp),mask_sha256=tensor_hash(masks),original_files=len(before)))
    if original_files(root)!=before:raise RuntimeError('Original artifacts changed during export')
    print(json.dumps(dict(status='PASS',input_sha256=file_hash(out),mask_sha256=tensor_hash(masks),original_files=len(before))))
