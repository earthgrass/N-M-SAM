"""Independent, bounded partial-layer technical smoke. Never a formal experiment."""
import argparse,copy,json,shutil,subprocess,time,traceback
from pathlib import Path
import torch,yaml
from torch import nn
from torch.utils.data import Subset
from . import artifacts
from .audit_inputs import load_original
from .calibration import approved_indices
from .common import atomic_json,environment,finite,object_sha,rng_state,seed_all,sha_file,tensor_sha
from .data import Images,make_loader,real_data
from .evaluation import recalibrate_copy
from .model import build_model
from .nm import eligible_layers,project,validate_masks
from .obc import load_official,prune_with_hessian
from .preflight import ROOT,source_identity,require_config_hash
from .resource_monitor import Monitor
from .smoke_provenance import verify as verify_provenance

def review_errors(c,formal):
    errors=[]
    if c.get('purpose')!='cuda_technical_smoke_only' or c.get('method')!='OBC' or c.get('seed')!=0:errors.append('Only OBC seed0 technical smoke accepted')
    if c['review']['approved'] is not True:errors.append('BLOCKED_CONFIG_REVIEW')
    try:
        ids=approved_indices(dict(c,data=formal['data']))
        if len(ids)>c['sample_resource_cap']:errors.append('Smoke sample resource cap exceeded')
        if (len(ids)+c['calibration']['batch_size']-1)//c['calibration']['batch_size']<c['recovery_steps']:
            errors.append('Not enough distinct batches for bounded technical recovery')
        verify_provenance(c,formal)
    except (ValueError,KeyError,FileNotFoundError) as exc:errors.append('BLOCKED_CALIBRATION_APPROVAL: '+str(exc))
    if c['layer_name']!='layer1.0.conv1' or c['recovery_steps']!=2 or c['parallel']!=2:
        errors.append('Technical scope changed; not the reviewed bounded smoke')
    return errors

def micro_recovery(model,masks,batches,rc,steps,path,identity,parent,check=lambda:None):
    optimizer=torch.optim.SGD(model.parameters(),lr=rc['learning_rate'],momentum=rc['momentum'],
        weight_decay=rc['weight_decay'],nesterov=rc['nesterov'])
    before=tensor_sha(masks);count=0;model.train();losses=[]
    for x,y,*_ in batches:
        check();optimizer.zero_grad(set_to_none=True);device=next(model.parameters()).device
        loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));finite([loss],'Technical SGD loss');loss.backward()
        project(model,masks,optimizer);finite((p.grad for p in model.parameters() if p.grad is not None),'Technical SGD gradients')
        optimizer.step();project(model,masks,optimizer);validate_masks(model,masks,complete=False)
        for name,p in model.named_parameters():
            if name in masks and torch.count_nonzero(optimizer.state[p]['momentum_buffer'][~masks[name].to(p.device)]):
                raise ValueError('Smoke masked momentum regrowth')
        finite(model.state_dict().values(),'Technical SGD model');count+=1;losses.append(float(loss.detach()))
        if tensor_sha(masks)!=before:raise ValueError('Technical Mask changed')
        if count==steps:break
    if count!=steps:raise ValueError('Not enough approved smoke batches')
    value={'stage':'recovery','identity':identity,'parent':parent,'model':model.state_dict(),'masks':masks,
           'optimizer':optimizer.state_dict(),'rng':rng_state(),'completed_steps':count,'train_losses':losses,
           'selection':'none; train-only technical smoke; not formal recovery'}
    digest=artifacts.save(path,value);restored=artifacts.load(path,identity)
    fresh=copy.deepcopy(model);fresh.load_state_dict(restored['model']);validate_masks(fresh,masks,complete=False)
    optimizer2=torch.optim.SGD(fresh.parameters(),lr=rc['learning_rate'],momentum=rc['momentum'],weight_decay=rc['weight_decay'])
    optimizer2.load_state_dict(restored['optimizer'])
    old_state=optimizer.state_dict();new_state=optimizer2.state_dict()
    if old_state['param_groups']!=new_state['param_groups'] or set(old_state['state'])!=set(new_state['state']):
        raise ValueError('Smoke optimizer reload groups/state mapping')
    for key,state in old_state['state'].items():
        for field,value in state.items():
            reloaded=new_state['state'][key][field]
            if isinstance(value,torch.Tensor):
                if not torch.equal(value.cpu(),reloaded.cpu()):raise ValueError('Smoke optimizer tensor reload mismatch')
            elif value!=reloaded:raise ValueError('Smoke optimizer scalar reload mismatch')
    for p,state in optimizer2.state.items():
        if 'momentum_buffer' in state:finite([state['momentum_buffer']],'Reloaded momentum')
    if tensor_sha(fresh.state_dict())!=tensor_sha(model.state_dict()):raise ValueError('Smoke model reload mismatch')
    return {'steps':count,'checkpoint_sha256':digest,'mask_sha256':before,'losses':losses,'reload':'PASS','optimizer_reload':'EXACT_MATCH'}

def run(args):
    c=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    formal=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    errors=review_errors(c,formal)
    if errors:raise ValueError('; '.join(errors)) # no CUDA initialization before approval
    require_config_hash(args.config,args.expected_config_sha256)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=args.expected_commit or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Reviewed clean source commit required')
    if not torch.cuda.is_available() or c['device']!='cuda:0':raise ValueError('Approved CUDA device unavailable; no CPU fallback')
    props=torch.cuda.get_device_properties(0)
    if str(props.uuid).removeprefix('GPU-')!=c['gpu_uuid'].removeprefix('GPU-') or props.name!=c['gpu_name']:raise ValueError('Wrong approved GPU')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
    import os
    if any(p.strip()!=str(os.getpid()) for p in processes if p.strip()):raise ValueError('Existing GPU task; do not overlap')
    if torch.cuda.mem_get_info(0)[0]<c['minimum_free_cuda_bytes'] or shutil.disk_usage(ROOT).free<c['minimum_free_disk_bytes']:
        raise ValueError('Insufficient smoke resource reserve')
    out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'artifacts'):raise ValueError('Smoke output outside isolated artifacts')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    seed_all(0,c['threads'])
    identity={'purpose':c['purpose'],'commit':commit,'config_sha256':sha_file(args.config),'source':source_identity(),
              'environment':environment(),'seed':0,'gpu_uuid':c['gpu_uuid']}
    try:
        seed_all(0,c['threads']);torch.cuda.reset_peak_memory_stats()
        records=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))['checkpoints']
        record=next(r for r in records if r['seed']==0 and r['role']=='standard_dense_sam')
        parent,digest=load_original(Path(args.input_root)/record['relative_path'],record['sha256'])
        if 'geometry' in parent or parent['identity']['seed']!=0:raise ValueError('Wrong dense parent')
        data=dict(formal['data'],data_root=args.data_root,evaluation_batch_size=c['calibration']['batch_size'])
        train,_,manifest=real_data(data)
        if object_sha(manifest)!=parent['identity']['data_manifest_sha256']:raise ValueError('Data provenance changed')
        proof=verify_provenance(c,formal)
        if proof['dataset_manifest_sha256']!=parent['identity']['data_manifest_sha256']:raise ValueError('Smoke dataset/parent mismatch')
        if [name for name,_ in train.dataset.train_list]!=proof['training_file_order']:raise ValueError('CIFAR training file/index order changed')
        ids=approved_indices(dict(c,data=formal['data']));subset=Subset(Images(train.dataset,data,False),ids)
        def batches():return make_loader(subset,data,workers=0)
        dense=build_model().to(c['device']).eval();dense.load_state_dict(parent['model']);before=tensor_sha(dense.state_dict())
        if any(p.dtype!=torch.float32 for p in dense.parameters()) or torch.is_autocast_enabled('cuda'):
            raise ValueError('Smoke requires FP32 and no AMP')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('Smoke TF32 enabled')
        teacher=copy.deepcopy(dense);layer=teacher.get_submodule(c['layer_name']);native=copy.deepcopy(dense)
        if c['layer_name'] not in eligible_layers(native):raise ValueError('Excluded smoke layer')
        handle=load_official(c['device'])(layer,rel_damp=0.)
        relation={'sha256':digest,'stage':'existing_dense_sam_best_test'}
        with Monitor(out/'resources.jsonl',c['device'],c['minimum_free_disk_bytes'],c['monitor_interval_seconds']) as monitor:
            hook=layer.register_forward_hook(lambda module,inp,output:handle.add_batch(inp[0].detach(),output.detach()))
            try:
                with torch.no_grad():
                    for _ in range(c['calibration']['rounds']):
                        for x,*_ in batches():monitor.check();teacher(x.to(c['device']))
            finally:hook.remove()
            finite([handle.H],'Smoke Hessian')
            original_weight=native.get_submodule(c['layer_name']).weight.detach().clone()
            value,mask,receipt=prune_with_hessian(native.get_submodule(c['layer_name']),handle.H,parallel=c['parallel'])
            raw_error=(original_weight*mask-original_weight).flatten(1).double()
            compensated_error=(value-original_weight).flatten(1).double()
            raw_quadratic=.5*((raw_error@handle.H)*raw_error).sum()
            compensated_quadratic=.5*((compensated_error@handle.H)*compensated_error).sum()
            finite([raw_quadratic,compensated_quadratic],'Technical reconstruction diagnostics')
            receipt.update(raw_zeroing_quadratic=float(raw_quadratic),compensated_quadratic=float(compensated_quadratic),
                surviving_weight_change_l2=float((value-original_weight)[mask].norm()))
            native.get_submodule(c['layer_name']).weight.data.copy_(value);masks={c['layer_name']+'.weight':mask.cpu()}
            validate_masks(native,masks,complete=False)
            native_path=out/'native_partial.pt'
            native_sha=artifacts.save(native_path,{'identity':identity,'parent':relation,'stage':'native_partial',
                'model':native.state_dict(),'masks':masks,'receipt':receipt,'partial_layer_only':True})
            calibrated,bn_info=recalibrate_copy(native,batches(),c['device'])
            artifacts.save(out/'bn_only_smoke.pt',{'identity':identity,'parent':{'sha256':native_sha},'stage':'common_bn',
                'model':calibrated.state_dict(),'masks':masks,'bn':bn_info})
            recovery=micro_recovery(copy.deepcopy(native),masks,batches(),formal['recovery'],c['recovery_steps'],
                out/'recovery_smoke.pt',identity,{'sha256':native_sha,'stage':'native_partial'},monitor.check)
            if sha_file(native_path)!=native_sha or tensor_sha(dense.state_dict())!=before or tensor_sha(teacher.state_dict())!=before:
                raise ValueError('Native output or dense reference overwritten')
            monitor.check()
        torch.cuda.synchronize()
        atomic_json(out/'SMOKE_REPORT.json',{'status':'CUDA_TECHNICAL_SMOKE_PASS','identity':identity,'partial_layer':c['layer_name'],
            'native_sha256':native_sha,'bn':bn_info,'recovery':recovery,'seconds':time.monotonic()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'not_formal_accuracy_or_20epoch_acceptance':True})
    except Exception:
        atomic_json(out/'FAILURE.json',{'identity':identity,'traceback':traceback.format_exc()});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/obc_cuda_smoke.yaml')
    p.add_argument('--expected-config-sha256',required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--input-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())

if __name__=='__main__':main()
