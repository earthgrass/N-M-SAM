import copy
import json
import pytest
import torch
import numpy as np
from vpr_sam.config import validate
from vpr_sam.data import build_data,loader
from vpr_sam.training import train,prune,evaluate_checkpoint
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.analysis.report import summarize,plot
from vpr_sam.repro import project_path,source_hashes


def nested_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def test_resume_exact_mid_epoch(config,isolated_root):
    continuous=copy.deepcopy(config);continuous["output"]="runs/continuous"
    split=copy.deepcopy(config);split["output"]="runs/split"
    a=train(continuous,isolated_root)
    train(split,isolated_root,stop_after=2)
    b=train(split,isolated_root,resume="runs/split/last.pt")
    left,_=read_checkpoint("runs/continuous/final.pt",isolated_root)
    right,_=read_checkpoint("runs/split/final.pt",isolated_root)
    for field in ("model","stepper","scheduler","rng","epoch_rows","order_hash"):
        assert nested_equal(left[field],right[field]),field
    assert a["initialization_hash"]==b["initialization_hash"]
    bad=copy.deepcopy(split);bad["geometry"]["dtype"]="float32"
    with pytest.raises(ValueError,match="config"):train(bad,isolated_root,resume="runs/split/last.pt")
    with pytest.raises(FileExistsError):train(split,isolated_root)


def test_data_order_worker_independence(config,isolated_root):
    datasets,_=build_data(config,isolated_root)
    a=list(loader(datasets["train"],4,17,epoch=2,shuffle=True,workers=0))
    b=list(loader(datasets["train"],4,17,epoch=2,shuffle=True,workers=1))
    assert all(torch.equal(x,y) for aa,bb in zip(a,b) for x,y in zip(aa,bb))


def test_end_to_end_comparison(config,isolated_root):
    paths=[]
    for method in ("sgd","sam","vpr"):
        c=copy.deepcopy(config);c["optimizer"]["method"]=method;c["output"]=f"runs/{method}"
        train(c,isolated_root)
        result=prune(f"runs/{method}/final.pt",isolated_root)
        parent,_=read_checkpoint(result["checkpoint"],isolated_root)
        recovery=train(parent["config"],isolated_root,recover_from=result["checkpoint"])
        assert recovery["complete"] and recovery["stage"]=="recovery"
        evaluated=evaluate_checkpoint(result["checkpoint"],isolated_root,"test")
        assert evaluated["metrics"]["count"]==8
        evaluate_checkpoint(f"runs/{method}/final.pt",isolated_root,"test")
        evaluate_checkpoint(str((isolated_root/result["checkpoint"]).parent/"recovery/final.pt"),isolated_root,"test")
        paths.append(f"runs/{method}")
    report=summarize(isolated_root,paths,"runs/analysis")
    assert all(item["paired_identity_match"] for item in report["fairness"])
    assert report["paired"]["vpr"]["n"]==1
    assert report["paired"]["vpr"]["paired_bootstrap_95_percentile_interval"] is None
    test_report=summarize(isolated_root,paths,"runs/test_analysis",split="test")
    assert test_report["split"]=="test"
    figures=plot(isolated_root,"runs/analysis/summary.json","runs/analysis/figures")
    assert figures["figures"]==7


def test_guards(config,isolated_root):
    with pytest.raises(ValueError):project_path(isolated_root,"../escape")
    bad=copy.deepcopy(config);bad["purpose"]="formal"
    with pytest.raises(ValueError,match="allow-formal"):validate(bad)
    bad["seed"]=None
    with pytest.raises(ValueError,match="Unresolved"):validate(bad,formal_allowed=True)
    bad=copy.deepcopy(config);bad["pruning"]["n"]=None
    with pytest.raises(ValueError):validate(bad)


def test_real_data_never_falls_back(config,isolated_root,monkeypatch):
    from torchvision.datasets import CIFAR10
    def fail(*a,**k):raise RuntimeError("missing real data")
    monkeypatch.setattr(CIFAR10,"__init__",fail)
    config["data"]["kind"]="cifar10"
    with pytest.raises(RuntimeError,match="missing real data"):build_data(config,isolated_root)


def test_recovery_resume_exact(config,isolated_root):
    config["recovery"]["epochs"]=2
    config["recovery"]["method"]="sam"
    train(config,isolated_root)
    a=prune(config["output"]+"/final.pt",isolated_root)
    base,_=read_checkpoint(a["checkpoint"],isolated_root)
    train(base["config"],isolated_root,recover_from=a["checkpoint"])
    final_a,_=read_checkpoint(str((isolated_root/a["checkpoint"]).parent/"recovery/final.pt"),isolated_root)
    # A second explicit hard-pruning artifact from best-validation, same model
    # values for source equality are not assumed; compare its own split run.
    import shutil
    original=(isolated_root/a["checkpoint"]).parent
    second=original.parent/"resume_parent"
    second.mkdir()
    shutil.copy2(original/"pruned.pt",second/"pruned.pt")
    second_relative=str((second/"pruned.pt").relative_to(isolated_root))
    train(base["config"],isolated_root,recover_from=second_relative,stop_after=2)
    resume_relative=str((second/"recovery/last.pt").relative_to(isolated_root))
    train(base["config"],isolated_root,recover_from=second_relative,resume=resume_relative)
    final_b,_=read_checkpoint(str((second/"recovery/final.pt").relative_to(isolated_root)),isolated_root)
    for key in ("model","stepper","scheduler","order_hash"):
        assert nested_equal(final_a[key],final_b[key]),key


def test_summary_refuses_unfair_protocol(config,isolated_root):
    a=copy.deepcopy(config);a["output"]="runs/a";a["optimizer"]["method"]="sam"
    b=copy.deepcopy(config);b["output"]="runs/b";b["optimizer"]["lr"]*=2
    train(a,isolated_root);train(b,isolated_root)
    with pytest.raises(ValueError,match="protocol mismatch"):
        summarize(isolated_root,["runs/a","runs/b"],"runs/analysis")


def test_provenance_excludes_build_artifacts(isolated_root):
    (isolated_root/"src/vpr_sam/module.py").write_text("VALUE=1\n")
    metadata=isolated_root/"src/vpr_sam_research.egg-info"
    metadata.mkdir();(metadata/"PKG-INFO").write_text("generated metadata")
    initial=source_hashes(isolated_root)
    (metadata/"PKG-INFO").write_text("regenerated metadata")
    assert source_hashes(isolated_root)==initial
    (isolated_root/"src/vpr_sam/module.py").write_text("VALUE=2\n")
    assert source_hashes(isolated_root)!=initial
