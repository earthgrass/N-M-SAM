"""Future gated CUDA entry point; not exercised or authorized as a formal run yet."""
import argparse,json,subprocess,traceback
from pathlib import Path
import torch,yaml
from torch.utils.data import Subset
from . import artifacts
from .audit_inputs import load_original
from .calibration import approved_indices,index_sha
from .common import atomic_json,environment,object_sha,seed_all,sha_file
from .data import Images,make_loader,real_data,verify_worker_rng
from .evaluation import evaluate,recalibrate_copy
from .model import build_model
from .pipeline import prune_obc
from .preflight import ROOT,gates,source_identity,require_config_hash
from .recovery import recover
from .resource_monitor import Monitor

def run(args):
    require_config_hash(args.config,args.expected_config_sha256)
    actual_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=args.expected_commit:raise ValueError('Reviewed source Git commit mismatch')
    c=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));errors=gates(c,gpu=True)
    if errors:raise ValueError('; '.join(errors))
    identity={'source':source_identity(),'config_sha256':sha_file(args.config),'seed':args.seed,
        'method':'OBC','commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    if args.seed not in c['seeds']:raise ValueError('Seed not registered')
    if args.seed!=0:
        if args.pilot_review is None:raise ValueError('Reviewed seed0 pilot required before seeds1/2')
        review=json.loads(Path(args.pilot_review).read_text(encoding='utf-8'))
        if review.get('approved') is not True or review.get('source_sha256')!=identity['source']['sha256'] or review.get('config_sha256')!=identity['config_sha256']:
            raise ValueError('Pilot review provenance mismatch')
    out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'artifacts'):raise ValueError('New outputs must stay inside this experiment artifacts/')
    if out.exists() and not args.resume:raise FileExistsError('Output exists; explicit healthy resume required')
    out.mkdir(parents=True,exist_ok=True)
    monitor=None
    try:
        old=yaml.safe_load((ROOT/'configs/historical_protocol.yaml').read_text(encoding='utf-8'))
        data=dict(c['data'],data_root=args.data_root)
        seed_all(args.seed,c['threads']);worker=verify_worker_rng()
        train,test,manifest=real_data(data)
        records=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))['checkpoints']
        item=next(r for r in records if r['seed']==args.seed and r['role']=='standard_dense_sam')
        path=Path(args.input_root)/item['relative_path'];parent,digest=load_original(path,item['sha256'])
        if 'geometry' in parent or parent['identity']['seed']!=args.seed:raise ValueError('Only standard Dense-SAM parent allowed')
        if parent['identity']['data_manifest_sha256']!=object_sha(manifest):raise ValueError('Historical dataset identity mismatch')
        identity.update(environment=environment(),data_manifest_sha256=object_sha(manifest),worker_policy=worker['policy'])
        if args.resume:
            previous=json.loads((out/'identity.json').read_text(encoding='utf-8'))
            if previous!=identity:raise ValueError('Run source/config/environment changed')
        else:atomic_json(out/'identity.json',identity)
        torch.cuda.reset_peak_memory_stats()
        monitor=Monitor(out/'resources.jsonl',c['device'],c['minimum_free_disk_bytes'],2)
        monitor.__enter__()
        def watched(iterator):
            for batch in iterator:
                monitor.check();yield batch
        ids=approved_indices(c)
        seed_all(c['calibration']['random_seed'],c['threads'])
        calibration=Subset(Images(train.dataset,data,False),ids)
        calibration_receipts=[]
        def calibration_factory():
            calibration_data=dict(data,evaluation_batch_size=c['calibration']['batch_size'])
            for _ in range(c['calibration']['rounds']):yield from watched(make_loader(calibration,calibration_data,workers=0))
        if c['calibration']['augmentation']=='obc_random_resized_crop32_flip':
            from .obc_calibration import factory
            calibration_factory=factory(train.dataset,ids,data,c['calibration'],monitor.check,calibration_receipts)
        # Reference model is never pruned; official Conv Unfold accumulation is retained.
        dense=build_model().to(c['device']);dense.load_state_dict(parent['model'])
        relation={'sha256':digest,'model_sha256':parent['model_sha256'],'stage':'existing_dense_sam_best_test'}
        native,masks,receipts=prune_obc(dense,calibration_factory,out/'native',identity,relation,
            parallel=c['OBC']['parallel'],resume=args.resume)
        monitor.check()
        if calibration_receipts:
            if any(row!=calibration_receipts[0] for row in calibration_receipts):
                raise ValueError('Layer calibration views/order changed')
            if calibration_receipts[0]['samples_including_rounds']!=c['calibration']['samples']*c['calibration']['rounds']:
                raise ValueError('Augmented calibration coverage')
            atomic_json(out/'calibration_replay.json',calibration_receipts)
        native_path=out/'native/native_complete.pt';native_sha=sha_file(native_path)
        metrics={'dense':evaluate(dense,watched(make_loader(test,data)),c['device']),
                 'native_post_pruning':evaluate(native,watched(make_loader(test,data)),c['device'])}
        bn_ids=json.loads((ROOT/c['common_bn']['index_file']).read_text(encoding='utf-8'))
        if index_sha(bn_ids)!=c['common_bn']['index_sha256']:raise ValueError('Frozen BN identities changed')
        if c['common_bn'].get('enabled',True):
            bn_data=Subset(Images(train.dataset,data,False),bn_ids)
            bn,bn_info=recalibrate_copy(native,watched(make_loader(bn_data,data,workers=0)),c['device'])
            if not (out/'common_bn.pt').exists():
                artifacts.save(out/'common_bn.pt',{'identity':identity,'stage':'common_bn','model':bn.state_dict(),
                    'masks':masks,'parent':{'sha256':native_sha,'stage':'native_complete'},'bn':bn_info})
            else:
                existing=artifacts.load(out/'common_bn.pt',identity)
                if existing['parent']['sha256']!=native_sha:raise ValueError('BN parent mismatch')
            metrics['common_bn']=evaluate(bn,watched(make_loader(test,data)),c['device'])
        else:
            metrics['common_bn']={'status':'SKIPPED_BY_USER','no_bn_copy_or_evaluation':True}
        atomic_json(out/'post_pruning_metrics.json',metrics)
        # Match historical fresh-constructor RNG consumption, then load pre-BN native weights.
        seed_all(args.seed,c['threads']);recovered=build_model().to(c['device']);recovered.load_state_dict(native.state_dict())
        rc=dict(c,seed=args.seed)
        recovery_dir=out/'recovery'
        result=recover(recovered,masks,lambda:watched(make_loader(train,data,True)),lambda:watched(make_loader(test,data)),
            rc,recovery_dir,identity,{'sha256':native_sha,'stage':'native_complete'},
            resume=args.resume and (recovery_dir/'last.pt').exists())
        monitor.__exit__(None,None,None);monitor=None
        atomic_json(out/'completion.json',{'status':'COMPLETE','identity':identity,'metrics':metrics,'recovery':result,
            'native_sha256':native_sha,'layer_receipts':receipts,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'pilot_review_required_before_next_seed':args.seed==0})
    except Exception as exc:
        # Preserve every failure attempt; never overwrite a preceding failure record.
        import time
        atomic_json(out/f'failure_{time.time_ns()}.json',{'error':repr(exc),'traceback':traceback.format_exc(),'identity':identity})
        raise
    finally:
        if monitor is not None:monitor.__exit__(None,None,None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/post_training_nm_benchmark.yaml')
    p.add_argument('--input-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--pilot-review')
    p.add_argument('--expected-config-sha256',required=True)
    p.add_argument('--expected-commit',required=True)
    run(p.parse_args())

if __name__=='__main__':main()
