import copy
import json
import pytest
import torch

from vpr_sam.analysis.summary import summarize, postprocessing_statistics
from vpr_sam.analysis.statistics import seed_statistics
from vpr_sam.config import validate, method_protocol
from vpr_sam.training import train, prune, evaluate_checkpoint
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.repro import atomic_json


def fixture_runs(config):
    runs = []
    for seed in (1, 2):
        for method in ("sam", "vpr"):
            hard = {("sam", 1): .7, ("sam", 2): .6, ("vpr", 1): .8, ("vpr", 2): .75}[method, seed]
            item = {"protocol": copy.deepcopy(config["pruning"]), "selection": {"layout": "test_fixture", "bn_calibration": "none"},
                    "hard_accuracy": hard, "hard_loss": 1-hard, "hard_loss_increase": .1 if method=="vpr" else .2,
                    "accuracy_drop": .05, "recovered_accuracy": .8+.05*(seed-1)+(.05 if method=="vpr" else 0),
                    "recovered_loss": .1, "recovery_protocol": copy.deepcopy(config["recovery"]),
                    "recovery_checkpoint": "final", "recovery_order_hash": f"order_seed_{seed}"}
            runs.append({"seed": seed, "method": method, "pruning": [item]})
    return runs


def test_hard_and_recovery_paired_statistics(config):
    report = postprocessing_statistics(fixture_runs(config), "sam")
    hard = report["hard"]["accuracy"]["paired"]["vpr"]
    recovery = report["recovery"]["accuracy"]["paired"]["vpr"]
    assert hard["seeds"] == [1, 2] and hard["n"] == 2
    assert hard["mean_difference"] == pytest.approx(.125)
    assert hard["paired_bootstrap_95_percentile_interval"] == pytest.approx([.1, .15])
    assert recovery["mean_difference"] == pytest.approx(.05)
    assert report["hard"]["loss_increase"]["paired"]["vpr"]["mean_difference"] == pytest.approx(-.1)
    assert report["hard"]["accuracy"]["statistics"]["vpr"]["sample_std"] == pytest.approx(.05 / 2**.5)


@pytest.mark.parametrize("field,value", [("n", 2), ("m", 5), ("scope", "exclude_first"), ("tail", "error"), ("score", "different_rule")])
def test_postprocessing_rejects_mixed_protocols(config, field, value):
    runs = fixture_runs(config)
    runs[-1]["pruning"][0]["protocol"][field] = value
    with pytest.raises(ValueError, match="Pruning protocol"):
        postprocessing_statistics(runs, "sam")


def test_postprocessing_rejects_incomplete_seeds_recovery_and_order(config):
    runs = fixture_runs(config)
    with pytest.raises(ValueError, match="matched seed"):
        postprocessing_statistics(runs[:-1], "sam")
    runs[-1]["pruning"][0]["recovery_order_hash"] = "wrong_order"
    with pytest.raises(ValueError, match="data-order"):
        postprocessing_statistics(runs, "sam")
    runs = fixture_runs(config)
    runs[-1]["pruning"][0].pop("recovered_accuracy")
    with pytest.raises(ValueError, match="complete"):
        postprocessing_statistics(runs, "sam")
    runs = fixture_runs(config)
    runs[-1]["pruning"][0]["recovery_protocol"]["epochs"] += 1
    with pytest.raises(ValueError, match="Recovery budget"):
        postprocessing_statistics(runs, "sam")


@pytest.mark.parametrize("field,value", [("gamma", .01), ("warmup_steps", 2), ("dtype", "float32")])
def test_same_method_hyperparameters_not_poolable(config, isolated_root, field, value):
    a=copy.deepcopy(config);a["output"]="runs/a"
    b=copy.deepcopy(config);b.update(seed=18, output="runs/b")
    b["geometry"][field]=value
    if field=="dtype":b["geometry"].update(volume_atol=1e-5,volume_rtol=1e-6)
    train(a,isolated_root);train(b,isolated_root)
    with pytest.raises(ValueError, match="Within-method"):
        summarize(isolated_root,["runs/a","runs/b"],"runs/summary")


def test_cross_method_geometric_parameters_are_allowed(config, isolated_root):
    a=copy.deepcopy(config);a["optimizer"]["method"]="sam";a["output"]="runs/sam"
    b=copy.deepcopy(config);b["geometry"]["gamma"]*=2;b["output"]="runs/vpr"
    train(a,isolated_root);train(b,isolated_root)
    assert summarize(isolated_root,["runs/sam","runs/vpr"],"runs/summary")["paired"]["vpr"]["n"]==1


def test_full_workflow_formal_gate(config):
    c=copy.deepcopy(config);c.update(purpose="formal",model="resnet18")
    c["data"]["kind"]="cifar10"
    validate(c,formal_allowed=True)  # config validation only; does not load data/train
    for section in ("pruning", "recovery"):
        for field in c[section]:
            bad=copy.deepcopy(c);bad[section][field]=None
            with pytest.raises(ValueError,match="Unresolved protocol"):
                validate(bad,formal_allowed=True)
    c["recovery"]["lr"]=float("nan")
    with pytest.raises(ValueError,match="recovery.lr"):validate(c,formal_allowed=True)


@pytest.mark.parametrize("operation", ["evaluate", "prune", "recover"])
def test_postprocessing_refuses_changed_source(config, isolated_root, operation):
    module=isolated_root/"src/vpr_sam/module.py"
    module.write_text("VERSION=1\n")
    train(config,isolated_root)
    checkpoint=config["output"]+"/final.pt"
    if operation=="recover":checkpoint=prune(checkpoint,isolated_root)["checkpoint"]
    module.write_text("VERSION=2\n")
    with pytest.raises(ValueError,match="source/config/scripts identity"):
        if operation=="evaluate":evaluate_checkpoint(checkpoint,isolated_root)
        elif operation=="prune":prune(checkpoint,isolated_root)
        else:train(config,isolated_root,recover_from=checkpoint)


@pytest.mark.parametrize("identity", ["model", "config", "masks"])
def test_artifact_identity_tampering_rejected(config, isolated_root, identity):
    train(config,isolated_root)
    path=config["output"]+"/final.pt"
    if identity=="masks":path=prune(path,isolated_root)["checkpoint"]
    state,absolute=read_checkpoint(path,isolated_root)
    if identity=="model":next(iter(state["model"].values())).flatten()[0]+=1
    elif identity=="config":state["config"]["seed"]+=1
    else:next(iter(state["masks"].values())).flatten()[0].logical_not_()
    torch.save(state,absolute)
    with pytest.raises(ValueError,match="identity mismatch"):read_checkpoint(path,isolated_root)


def test_changed_evaluation_provenance_rejected(config, isolated_root):
    c=copy.deepcopy(config);c["output"]="runs/a"
    train(c,isolated_root)
    evaluate_checkpoint("runs/a/final.pt",isolated_root,"test")
    path=isolated_root/"runs/a/evaluation_final_test.json"
    value=json.loads(path.read_text(encoding="utf-8"));value["provenance"]["source_hashes"]={}
    atomic_json(path,value)
    with pytest.raises(ValueError,match="provenance"):
        summarize(isolated_root,["runs/a"],"runs/summary",split="test")


def test_two_seed_pruning_recovery_integration(config, isolated_root):
    paths=[]
    for seed in (17,18):
        for method in ("sam","vpr"):
            c=copy.deepcopy(config);c.update(seed=seed,output=f"runs/{method}_{seed}")
            c["optimizer"]["method"]=method
            train(c,isolated_root)
            hard=prune(c["output"]+"/final.pt",isolated_root)
            state,_=read_checkpoint(hard["checkpoint"],isolated_root)
            train(state["config"],isolated_root,recover_from=hard["checkpoint"])
            paths.append(c["output"])
    report=summarize(isolated_root,paths,"runs/paired")
    for stage in ("hard","recovery"):
        item=report["postprocessing"][stage]["accuracy"]["paired"]["vpr"]
        assert item["n"]==2 and item["paired_bootstrap_95_percentile_interval"] is not None
        field="hard_accuracy" if stage=="hard" else "recovered_accuracy"
        actual={}
        for run in report["runs"]:actual.setdefault(run["seed"],{})[run["method"]]=run["pruning"][0][field]
        expected=sum(row["vpr"]-row["sam"] for row in actual.values())/2
        assert item["mean_difference"]==pytest.approx(expected)
    assert (isolated_root/"runs/paired/postprocessing.csv").is_file()


def test_integration_rejects_different_pruning_override(config, isolated_root):
    paths=[]
    for method in ("sam","vpr"):
        c=copy.deepcopy(config);c["optimizer"]["method"]=method;c["output"]=f"runs/{method}"
        train(c,isolated_root)
        protocol=copy.deepcopy(c["pruning"])
        if method=="vpr":protocol["n"]=2
        prune(c["output"]+"/final.pt",isolated_root,protocol)
        paths.append(c["output"])
    with pytest.raises(ValueError,match="Pruning protocol/selection"):
        summarize(isolated_root,paths,"runs/invalid")
