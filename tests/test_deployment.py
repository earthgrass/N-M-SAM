from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import pytest
import torch
from torch import nn
from nmsam.config.experiment import ExperimentConfig,load_experiment_config,plan
from nmsam.utils.capabilities import require_execution_backend
from nmsam.experiment.evaluation import evaluate


def test_configs_parse_without_dataset_or_gpu():
    root=Path(__file__).resolve().parents[1]
    files=["gpu_baselines.yaml","gpu_smoke.yaml","research_direction_gap.yaml","research_objective_gap.yaml",
           "research_compensation.yaml","ablation_pure_masked.yaml","ablation_old_support.yaml"]
    for filename in files:
        config,raw=load_experiment_config(root/"configs"/filename)
        result=plan(config);assert result["dry_run"] and raw and result["dataset_status"]=="DATASET_PENDING"
    config,_=load_experiment_config(root/"configs/gpu_baselines.yaml")
    assert len(plan(config)["matrix"])==18 and config.split_seed==0


def test_default_cli_never_queries_cuda_or_opens_data(monkeypatch,capsys):
    root=Path(__file__).resolve().parents[1]
    def forbidden(*args,**kwargs):raise AssertionError("dry-run accessed runtime")
    monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    from nmsam.data import cifar_archive
    monkeypatch.setattr(cifar_archive,"prepare_archive",forbidden)
    spec=importlib.util.spec_from_file_location("run_experiment",root/"scripts/run_experiment.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.main(["--config",str(root/"configs/gpu_baselines.yaml")])==0
    output=json.loads(capsys.readouterr().out);assert output["dry_run"] and len(output["matrix"])==18


@pytest.mark.parametrize("kwargs",[{"amp":True},{"ddp":True}])
def test_unsupported_amp_ddp_fail_before_gpu_query(kwargs,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:(_ for _ in ()).throw(AssertionError("GPU queried")))
    with pytest.raises(NotImplementedError):require_execution_backend(ExperimentConfig(**kwargs))


def test_small_tensor_evaluation_mode_restore():
    model=nn.Linear(2,2);model.train()
    data=[(torch.tensor([[1.,2.],[2.,1.]]),torch.tensor([0,1]))]
    result=evaluate(model,data)
    assert model.training and result["samples"]==2 and result["objective"]=="task_CE"


@pytest.mark.parametrize("kwargs",[{"rho":float("nan")},{"batch_size":1},{"methods":["unknown"]},
    {"num_workers":2},{"seeds":[0,0]},{"kind":"compensation","block_ids":[0]}])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):ExperimentConfig(**kwargs)
