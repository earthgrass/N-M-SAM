"""Explicit-execute single-device training/probe dispatch. Never called by default."""
from dataclasses import asdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import subprocess
from uuid import uuid4
import torch
from torch.utils.data import DataLoader
from nmsam.config.experiment import resolved_path
from nmsam.core.batch import StepBatch
from nmsam.core.functional_model import EffectiveLoss
from nmsam.data.cifar10 import cifar_loaders
from nmsam.data.sampler import StatefulBatchSampler
from nmsam.models.tiny_cnn import TinyCNN
from nmsam.models.resnet_cifar import resnet18_cifar
from nmsam.sparsity.registry import MaskRegistry
from nmsam.diagnostics.candidate_mask import block_candidates,joint_candidates
from nmsam.diagnostics.direction_gap import evaluate_direction_gap
from nmsam.diagnostics.objective_gap import evaluate_objective_gap
from nmsam.diagnostics.compensation import compare_compensation
from nmsam.diagnostics.coupling import cross_block_coupling
from nmsam.utils.capabilities import require_execution_backend
from nmsam.utils.seed import seed_everything
from .training import BaselineTrainer
from .checkpoint import save_checkpoint,load_checkpoint
from .evaluation import evaluate
from .artifacts import save_result,config_fingerprint


def _scientific_hash(config):
    return config_fingerprint(asdict(config))


def _git_sha():
    root=Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()


def _model(config):
    model=resnet18_cifar() if config.model=="resnet18" else TinyCNN()
    return model.to(device=config.device,dtype=torch.float64 if config.precision=="fp64" else torch.float32)


def execute_training(config,raw_config,*,selected_method=None,selected_seed=None,resume=None):
    require_execution_backend(config)
    methods=[selected_method] if selected_method else config.methods
    seeds=[selected_seed] if selected_seed is not None else config.seeds
    if any(m not in config.methods for m in methods) or any(s not in config.seeds for s in seeds):
        raise ValueError("selected method/seed outside declared matrix")
    resume=resume or config.resume
    if resume and len(methods)*len(seeds)!=1:raise ValueError("resume requires one --method and --seed")
    summaries=[]
    for seed in seeds:
        seed_everything(seed)
        data=cifar_loaders(resolved_path(config.archive_path),resolved_path(config.dataset_cache_dir),
                           batch_size=config.batch_size,seed=config.split_seed)
        for method in methods:
            seed_everything(seed);model=_model(config)
            dataset=data["train"].dataset
            full_updates=(len(dataset)//config.batch_size)*config.epochs
            target=min(full_updates,config.max_batches) if config.max_batches else full_updates
            if target<1:raise ValueError("batch exceeds training split")
            trainer=BaselineTrainer(model,method,lr=config.lr,rho=config.rho,l2_reg=config.l2_reg,
                mask_interval=config.mask_interval,beta=config.momentum,nesterov=config.nesterov,
                gradient_clipping=config.gradient_clipping,ste=config.ste,scheduler=config.scheduler,total_steps=target)
            sampler=StatefulBatchSampler(len(dataset),config.batch_size,seed=seed,drop_last=True)
            loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
            metadata={"config_sha256":_scientific_hash(config),"split_sha256":data["provenance"]["split_sha256"],
                      "archive_sha256":data["provenance"]["archive_sha256"],"git_sha":_git_sha(),
                      "config_raw":raw_config,"source_config_sha256":hashlib.sha256(raw_config.encode()).hexdigest(),
                      "training_config":asdict(config),"seed":seed,"run_id":uuid4().hex}
            if resume:
                restored=load_checkpoint(resolved_path(resume),trainer,sampler,metadata)
                metadata["run_id"]=restored["run_id"]
            safe=method.replace(":","to")
            directory=Path(config.output_dir)/"runs"/f"seed{seed}_{safe}";directory.mkdir(parents=True,exist_ok=True)
            history=[]
            with gzip.open(directory/"steps.jsonl.gz","at",encoding="utf-8") as stream:
                while trainer.step<target:
                    for x,y in loader:
                        batch=StepBatch(x.to(device=config.device,dtype=next(model.parameters()).dtype),y.to(config.device))
                        metric=trainer.train(batch);metric.update(seed=seed,git_sha=metadata["git_sha"],
                            run_id=metadata["run_id"],config_sha256=metadata["config_sha256"],
                            source_config_sha256=metadata["source_config_sha256"],augmentation="crop32_pad4_flip_normalize",**data["provenance"])
                        stream.write(json.dumps(metric,allow_nan=False)+"\n");stream.flush()
                        if trainer.step>=target:break
                    if sampler.drop_last and sampler.size-sampler.cursor<sampler.batch_size:
                        sampler.cursor=sampler.size
                    if sampler.cursor==sampler.size:
                        history.append({"epoch":sampler.epoch+1,"validation":evaluate(model,data["validation"],trainer.registry,
                                               max_batches=config.eval_max_batches)})
                        sampler.next_epoch()
                    save_checkpoint(directory/"latest.pt",trainer,sampler,metadata)
                final=evaluate(model,data["validation"],trainer.registry,max_batches=config.eval_max_batches)
                test=evaluate(model,data["test"],trainer.registry) if config.max_batches is None else None
            summaries.append({"method":method,"seed":seed,"updates":trainer.step,"validation":final,
                              "test":test,"epochs":history,"data":data["provenance"],"status":"EXECUTED"})
    return save_result("gpu_baselines",asdict(config),{"runs":summaries,"config_raw":raw_config},config.output_dir)


def execute_probe(config,raw_config):
    require_execution_backend(config)
    if len(config.seeds)!=1:raise ValueError("one seed/checkpoint per mechanism probe")
    seed=config.seeds[0];seed_everything(seed)
    data=cifar_loaders(resolved_path(config.archive_path),resolved_path(config.dataset_cache_dir),batch_size=config.batch_size,seed=config.split_seed)
    checkpoint=torch.load(resolved_path(config.checkpoint),map_location=config.device,weights_only=False)
    metadata=checkpoint["metadata"]
    for key in ("archive_sha256","split_sha256"):
        if metadata.get(key)!=data["provenance"][key]:raise ValueError("probe dataset fingerprint mismatch")
    if checkpoint["mask"] is None:raise ValueError("candidate probe requires explicit sparse checkpoint support")
    model=_model(config);model.load_state_dict(checkpoint["model"])
    registry=MaskRegistry(model,ste=config.ste);registry.load_state_dict(checkpoint["mask"])
    x,y=next(iter(data["calibration"]))
    context=EffectiveLoss(model,StepBatch(x.to(device=config.device,dtype=next(model.parameters()).dtype),y.to(config.device)),
                          registry,state_mode=config.state_mode)
    payload={"data":data["provenance"],"config_raw":raw_config,"checkpoint_source_git":metadata["git_sha"],"labels":context.labels,"rows":[],"summaries":[]}
    if config.kind in ("direction_gap","objective_gap"):
        for block in config.block_ids:
            candidates=block_candidates(context,config.target_layer,block)
            if config.kind=="direction_gap":
                result=evaluate_direction_gap(context,context.z,context.master,context.support,candidates,rho=config.rho,labels=context.labels)
            else:
                if config.perturbation_domain=="old_support_control":
                    candidates=({**c,"support":context.support} for c in candidates)
                result=evaluate_objective_gap(context,context.z,context.support,candidates,rho=config.rho,
                          steps=config.inner_steps,restarts=config.restarts,labels=context.labels)
            payload["rows"].extend({**r,"block_id":block,"perturbation_domain":config.perturbation_domain} for r in result["rows"])
            payload["summaries"].append({"block_id":block,**result["rankings"]})
    elif config.kind=="compensation":
        blocks=[(config.target_layer,b) for b in config.block_ids]
        first_slice=context.parameter_slice(config.target_layer)
        a_start=first_slice.start+4*config.block_ids[0]
        for index,candidate in enumerate(joint_candidates(context,blocks,max_combinations=36)):
            if index>=config.joint_candidate_limit:break
            q=candidate["q"];qa=torch.zeros_like(q);qa[a_start:a_start+4]=q[a_start:a_start+4]
            coupling=cross_block_coupling(context,context.z,qa,q-qa)
            comparisons=compare_compensation(context,context.z,q,candidate["support"],config.compensation_budget,
                config.rho,max_hvps=config.max_hvps,inner_steps=config.inner_steps,
                outer_steps=config.outer_steps,restarts=config.restarts,damping=config.damping)
            for method,result in comparisons.items():
                row={k:v for k,v in result.items() if isinstance(v,(str,int,float,bool))}
                if "derivative" in row:row["optimization_derivative"]=row.pop("derivative")
                payload["rows"].append({**context.labels,**row,**coupling,"method":method,
                    "candidate_id":str(candidate["candidate_id"]),"damping":config.damping})
    else:raise ValueError("probe kind required")
    path,record=save_result("gpu_probe",asdict(config),payload,config.output_dir)
    rows=record["rows"]
    csv_path=path.with_suffix(".csv")
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)
    return path,record
