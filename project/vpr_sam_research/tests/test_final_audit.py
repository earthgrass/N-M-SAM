"""Only device consistency and selected-checkpoint plotting regressions."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vpr_sam.config import comparison_protocol, validate
from vpr_sam.repro import execution_record, environment, validate_execution_record, atomic_json
from vpr_sam.training import train, prune
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.analysis.summary import summarize
from vpr_sam.analysis.report import plot


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_formal_requires_explicit_approved_cuda_index(config, device, isolated_root):
    c=copy.deepcopy(config)
    c.update(purpose="formal",model="resnet18",device=device)
    c["data"]["kind"]="cifar10"
    with pytest.raises(ValueError,match="approved explicit CUDA"):
        validate(c,formal_allowed=True)
    with pytest.raises(ValueError,match="approved explicit CUDA"):
        train(c,isolated_root,formal_allowed=True)  # rejected before dataset/model IO
    c["device"]="cuda:2"
    assert validate(c,formal_allowed=True)["device"]=="cuda:2"  # syntax only, no CUDA execution


def test_gpu_visibility_does_not_mean_cpu_run_used_cuda(config,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:"NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda,"get_arch_list",lambda:["sm_120"])
    value=environment(config,torch.device("cpu"))
    assert value["execution"]["kind"]=="cpu" and value["gpu"] is None
    assert value["execution"]["resolved_device"]=="cpu"
    assert value["execution"]["threads"]==config["threads"]
    assert value["execution"]["interop_threads"]==torch.get_num_interop_threads()


def test_formal_cuda_records_the_target_not_gpu_zero(config,monkeypatch):
    c=copy.deepcopy(config);c.update(device="cuda:2",purpose="formal")
    calls=[]
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    def properties(index):
        calls.append(index)
        return SimpleNamespace(name="NVIDIA GeForce RTX 5090",uuid="GPU-approved-test",major=12,minor=0)
    monkeypatch.setattr(torch.cuda,"get_device_properties",properties)
    record=execution_record(c,torch.device("cuda:2"))  # mocked metadata only; no CUDA tensors/kernels
    assert calls==[2] and record["resolved_device"]=="cuda:2"
    assert record["gpu_uuid"]=="GPU-approved-test" and record["compute_capability"]==[12,0]
    validate_execution_record(c,record)
    for field,value in (("resolved_device","cpu"),("resolved_device","cuda:1"),
                        ("gpu_name","NVIDIA RTX 4090"),("gpu_uuid",None),("threads",2)):
        bad=copy.deepcopy(record);bad[field]=value
        with pytest.raises(ValueError):validate_execution_record(c,bad)


def test_unavailable_cuda_never_falls_back_to_cpu(config,monkeypatch):
    c=copy.deepcopy(config);c["device"]="cuda:0"
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="fallback forbidden"):execution_record(c)
    with pytest.raises(ValueError,match="device mismatch"):execution_record(c,"cpu")


def test_comparison_retains_device_and_threads(config):
    protocol=comparison_protocol(config)
    assert protocol["device"]=="cpu" and protocol["threads"]==1
    same=copy.deepcopy(config);same["optimizer"]["method"]="sam"
    assert comparison_protocol(same)==protocol
    for key,value in (("device","cuda:0"),("threads",2)):
        changed=copy.deepcopy(same);changed[key]=value
        assert comparison_protocol(changed)!=protocol


def test_summary_rejects_mixed_devices_even_with_identical_host_fingerprint(config,isolated_root,monkeypatch):
    from vpr_sam.analysis import summary as module
    paths=[]
    for method in ("sam","vpr"):
        c=copy.deepcopy(config);c["optimizer"]["method"]=method;c["output"]=f"runs/{method}"
        train(c,isolated_root);paths.append(c["output"])
    assert summarize(isolated_root,paths,"runs/valid")["paired"]["vpr"]["n"]==1
    original=module.completed_run
    def simulate_cuda_at_artifact_boundary(*args,**kwargs):
        summary,c,state,path=original(*args,**kwargs)
        if c["optimizer"]["method"]=="vpr":
            c=copy.deepcopy(c);c["device"]="cuda:0"
        # Deliberately identical host/environment objects; policy must reject
        # the requested devices independently. This does not execute CUDA.
        return summary,c,state,path
    monkeypatch.setattr(module,"completed_run",simulate_cuda_at_artifact_boundary)
    with pytest.raises(ValueError,match="Mixed CPU/CUDA"):
        summarize(isolated_root,paths,"runs/mixed")


def test_summary_rejects_different_thread_settings(config,isolated_root):
    paths=[]
    for method,threads in (("sam",1),("vpr",2)):
        c=copy.deepcopy(config);c.update(threads=threads,output=f"runs/{method}")
        c["optimizer"]["method"]=method
        train(c,isolated_root);paths.append(c["output"])
    with pytest.raises(ValueError,match="Thread settings differ"):
        summarize(isolated_root,paths,"runs/invalid")


def test_schema_two_checks_execution_metadata(config,isolated_root):
    from vpr_sam.provenance import verify_artifact
    train(config,isolated_root)
    state,_=read_checkpoint(config["output"]+"/final.pt",isolated_root)
    assert state["schema"]==2 and state["environment"]["execution"]["resolved_device"]=="cpu"
    verify_artifact(state,isolated_root,check_environment=True)
    state["environment"]["execution"]["resolved_device"]="cuda:0"
    with pytest.raises(ValueError,match="device mismatch"):verify_artifact(state,isolated_root)


@pytest.mark.parametrize("choice",["final","best_validation"])
def test_all_checkpoint_plots_use_selected_snapshot(config,isolated_root,monkeypatch,choice):
    import matplotlib.axes
    from vpr_sam.training import engine
    train(config,isolated_root)
    selected,_=read_checkpoint(config["output"]+f"/{choice}.pt",isolated_root)
    final,_=read_checkpoint(config["output"]+"/final.pt",isolated_root)
    best,_=read_checkpoint(config["output"]+"/best_validation.pt",isolated_root)
    assert best["stepper"]["steps"] < final["stepper"]["steps"]  # fixture actually distinguishes snapshots
    hard=prune(config["output"]+f"/{choice}.pt",isolated_root)
    parent,_=read_checkpoint(hard["checkpoint"],isolated_root)
    train(parent["config"],isolated_root,recover_from=hard["checkpoint"])
    summarize(isolated_root,[config["output"]],"runs/summary",checkpoint=choice,recovery_checkpoint=choice)
    calls=[];histograms=[]
    original=engine.read_checkpoint;original_hist=matplotlib.axes.Axes.hist
    def read(path,root):calls.append(str(path));return original(path,root)
    def hist(self,values,*args,**kwargs):histograms.append(np.asarray(values).copy());return original_hist(self,values,*args,**kwargs)
    monkeypatch.setattr(engine,"read_checkpoint",read)
    monkeypatch.setattr(matplotlib.axes.Axes,"hist",hist)
    result=plot(isolated_root,"runs/summary/summary.json","runs/figures")
    dense_calls=[path for path in calls if "/pruning/" not in path.replace("\\","/")]
    recovery_calls=[path for path in calls if "/recovery/" in path.replace("\\","/")]
    assert dense_calls and all(path.endswith(f"{choice}.pt") for path in dense_calls)
    assert recovery_calls and all(path.endswith(f"{choice}.pt") for path in recovery_calls)
    expected=torch.cat([s.flatten().double() for s in selected["stepper"]["geometry"]["logs"]]).numpy()
    assert len(histograms)==1 and np.array_equal(histograms[0],expected)
    sources=json.loads((isolated_root/"runs/figures/figure_sources.json").read_text(encoding="utf-8"))
    assert sources["checkpoint"]==result["checkpoint"]==choice
    assert sources["runs"][0]["epoch"]==selected["epoch"]
    assert sources["runs"][0]["steps"]==selected["stepper"]["steps"]
    assert sources["runs"][0]["pruning"][0]["recovery_checkpoint"]==choice
    summary=json.loads((isolated_root/"runs/summary/summary.json").read_text(encoding="utf-8"))
    summary["runs"][0]["checkpoint_sha256"]="0"*64
    atomic_json(isolated_root/"runs/summary/summary.json",summary)
    with pytest.raises(ValueError,match="checkpoint/hash mismatch"):
        plot(isolated_root,"runs/summary/summary.json","runs/refuse_mixed")
