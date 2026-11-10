"""Fail-closed V2 gates; writes evidence only, never launches formal training."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import torch
from vpr_sam.data import build_data, loader
from vpr_sam.models import build_model
from vpr_sam.repro import atomic_json, environment, seed_all, seed_for, source_hashes, tensor_hash, file_hash
from vpr_sam.l2_protocol import CONFIG, GPU_UUID, validate_protocol, full_order_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vpr_l2/control'


def main():
    c, diff = validate_protocol(ROOT)
    test_receipt = json.loads((OUT/'cuda_test_receipt.json').read_text())
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if test_receipt['exit_code'] != 0 or test_receipt['commit'] != commit or test_receipt['log_sha256'] != file_hash(OUT.parent/'cuda_tests.log'):
        raise RuntimeError('Missing/mismatched CUDA test receipt')
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Frozen experiment requires clean worktree')
    torch.set_num_threads(c['threads'])
    seed_all(seed_for(0,'initialization'), True)
    env = environment(c)
    ref = json.loads((ROOT/'docs/vpr_l2_reference/v1_gates.json').read_text())
    if env != ref['environment']:
        raise RuntimeError('V1/V2 execution environment mismatch')
    if env['execution']['gpu_uuid'].removeprefix('GPU-') != GPU_UUID:
        raise RuntimeError('Unapproved CUDA GPU UUID')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    others = [p for p in processes.splitlines() if p.strip().isdigit() and int(p) != __import__('os').getpid()]
    if others:
        raise RuntimeError(f'GPU already occupied: {others}')
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise RuntimeError('Insufficient space: require 8 GiB including >4 GiB safety reserve')
    # No legacy editable import path is allowed in this independent runtime.
    import vpr_sam
    if not Path(vpr_sam.__file__).resolve().is_relative_to(ROOT/'src'):
        raise RuntimeError('Package imported from another workspace')
    old_sources = ref['source_hashes']
    sources = source_hashes(ROOT)
    for name, sha in old_sources.items():
        if name.startswith('src/vpr_sam/') and sources[name] != sha:
            raise RuntimeError(f'Frozen training module changed: {name}')
    datasets, manifest = build_data(c, ROOT)
    expected = ref['identities']['0']
    if manifest['fingerprint'] != expected['dataset_fingerprint']:
        raise RuntimeError('Dataset/split identity mismatch')
    model = build_model(c['model']).to(c['device'])
    init = tensor_hash(model.state_dict())
    if init != expected['initialization_hash']:
        raise RuntimeError('Random initialization mismatch')
    order = full_order_hash(manifest)
    if order != expected['order_hash']:
        raise RuntimeError('Full 200-epoch sample order mismatch')
    ids = json.loads((ROOT/'docs/vpr_l2_reference/fixed_sample_ids.json').read_text())['0']
    if ids['bn_train'] != manifest['train_indices'][:4096] or ids['validation'] != manifest['validation_indices'][:1024]:
        raise RuntimeError('Frozen diagnostic sample mismatch')
    if expected['probe_validation_ids'] != manifest['validation_indices'][:len(expected['probe_validation_ids'])]:
        raise RuntimeError('Public sphere probe sample mismatch')
    augmentation = {}
    # Same byte-identical Images/seed_for + same raw official data, seed, epoch
    # and IDs establishes the complete stream. Record first batch every epoch.
    for epoch in range(200):
        x,y,indices = next(iter(loader(datasets['train'],128,0,epoch,workers=0,shuffle=True)))
        augmentation[str(epoch)] = hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()+indices.numpy().tobytes()).hexdigest()
    baseline = json.loads((OUT/'v1_augmentation_reference.json').read_text())
    if baseline['source_hashes'] != ref['source_hashes'] or baseline['augmentation_first_batch_sha256'] != augmentation:
        raise RuntimeError('V1 original-runtime / V2 augmented batch mismatch')
    gates = dict(status='PASS', commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config_sha256=file_hash(ROOT/CONFIG), config_diff=diff, source_hashes=sources, environment=env,
        initialization_hash=init, order_hash=order, dataset_fingerprint=manifest['fingerprint'],
        augmentation_first_batch_sha256=augmentation, augmentation_rule='unchanged epoch/sample keyed source; exact seed/IDs/raw data',
        v1_augmentation_reference_sha256=file_hash(OUT/'v1_augmentation_reference.json'),
        cuda_test_receipt=test_receipt,
        v1_commit=ref['commit'], free_bytes=shutil.disk_usage(ROOT).free,
        authorized_methods=['fixed_l2'], authorized_seeds=[0], recovery_authorized=False, test_authorized=False)
    atomic_json(OUT/'gates.json',gates)
    atomic_json(OUT/'config_diff.json',diff)
    atomic_json(OUT/'source_sha256.json',sources)
    (ROOT/'VPR_L2_PRECHECK.md').write_text('# VPR-L2 precheck\n\nPASS. See artifacts/vpr_l2/control/gates.json and cuda_tests.log.\n'
        f'\nFrozen commit: {gates["commit"]}\nConfig SHA256: {gates["config_sha256"]}\n'
        '\nOriginal Geometry, Stepper, data, model, engine and RNG modules remain byte-identical to V1. '
        'The approved fixed_l2 path alone sets the new perturbation norm. BN and optimizer updates are unchanged.\n',encoding='utf8')
    print(json.dumps({'status':'PASS','commit':gates['commit'],'initialization_hash':init,'order_hash':order}),flush=True)


if __name__ == '__main__':
    main()
