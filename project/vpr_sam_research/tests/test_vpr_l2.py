import copy
import json
from pathlib import Path
import pytest
import torch
from vpr_sam.optim import Stepper
from vpr_sam.optim.geometry import Geometry
from vpr_sam.models import build_model
from vpr_sam.l2_observer import ObservedStepper
from vpr_sam.l2_protocol import validate_protocol, differences
from vpr_sam.repro import tensor_hash, rng_state, restore_rng


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_nonunit_l2_direction_feedback(config, device):
    p = torch.nn.Parameter(torch.tensor([.4, -.8, .2, 1.], device=device))
    p.grad = torch.tensor([1., -.4, 0., .6], device=device)
    a = Geometry([("p", p)], config["geometry"], "vpr")
    b = Geometry([("p", p)], config["geometry"], "fixed_l2")
    s = torch.tensor([.8, -.3, -.7, .2], dtype=torch.float64, device=device)
    a.logs[0].copy_(s); b.logs[0].copy_(s)
    ed, _ = a.perturbations(.05); el, stats = b.perturbations(.05)
    reference = .05*s.mul(2).exp()*p.grad.double()
    reference /= (s.mul(2).exp()*p.grad.double()).norm()
    # Reference above already includes rho; normalization denominator excludes it.
    assert torch.allclose(el[0].double(), reference, rtol=2e-6, atol=1e-10)
    assert stats["eps_norm"] == pytest.approx(.05, rel=2e-5, abs=1e-12)
    assert stats["ellipsoid_norm"] != pytest.approx(.05, rel=1e-3)
    assert float(torch.nn.functional.cosine_similarity(ed[0].double(),el[0].double(),dim=0)) == pytest.approx(1.,abs=2e-14)
    pd = ed[0].double().square()/ed[0].double().square().sum()
    pl = el[0].double().square()/el[0].double().square().sum()
    assert torch.allclose(pd, pl, rtol=5e-7, atol=2e-15)
    a.prepare(ed, 0); b.prepare(el, 0)
    assert torch.allclose(a.candidate[0][0], b.candidate[0][0], rtol=0., atol=config["geometry"]["gamma"]*4*1e-6)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_observer_bitwise_bn_rng_and_resume(config, device):
    config["optimizer"]["method"] = "fixed_l2"
    torch.manual_seed(123)
    model = build_model("tiny").to(device); other = copy.deepcopy(model)
    plain = Stepper(model, config); observed = ObservedStepper(other, config)
    x, y = torch.randn(4,3,32,32,device=device), torch.arange(4,device=device)
    start = rng_state(); plain.step(x,y); expected_rng = torch.get_rng_state().clone()
    restore_rng(start); info = observed.step(x,y)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert tensor_hash(model.state_dict()) == tensor_hash(other.state_dict())
    assert int(other.bn.num_batches_tracked) == 1 and other.bn.track_running_stats
    assert info["delta_w_l2"] == pytest.approx(info["eps_norm"], abs=info["addition_roundoff_l2"]+1e-12)
    assert info["observed_addition_roundoff_l2"] == pytest.approx(info["addition_roundoff_l2"], rel=1e-12)
    assert info["addition_roundoff_l2"] <= info["addition_roundoff_bound_l2"]
    restored_model = copy.deepcopy(other)
    restored = ObservedStepper(restored_model,config)
    restored.load_state_dict(copy.deepcopy(observed.state_dict()))
    observed.step(x,y); restored.step(x,y)
    assert tensor_hash(other.state_dict()) == tensor_hash(restored_model.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(observed.geometry.logs,restored.geometry.logs))
    assert restored.geometry.method == "fixed_l2" and restored.geometry.updates == 2


def test_protocol_only_authorized_changes():
    root = Path(__file__).resolve().parents[1]
    c, diff = validate_protocol(root)
    assert c["seed"] == 0 and [d["field"] for d in diff] == ["optimizer.method","output"]
    altered = copy.deepcopy(c); altered["geometry"]["gamma"] *= 2
    assert differences(c, altered)[0]["field"] == "geometry.gamma"


def test_radius_gate_is_fail_closed(config):
    config["optimizer"]["method"] = "fixed_l2"
    model = build_model("tiny"); opt = ObservedStepper(model,config)
    original = opt.geometry.perturbations
    # Corruption injected below the observer, so the normal gate must reject.
    import vpr_sam.optim.geometry as gm
    real = gm.Geometry.perturbations
    def corrupt(self, rho, masks=None):
        eps, stats = real(self,rho,masks);stats["eps_norm"] *= 1.01
        return eps,stats
    # The closure captured a bound original method, patch its function temporarily.
    from unittest.mock import patch
    with patch.object(gm.Geometry,"perturbations",corrupt):
        opt = ObservedStepper(model,config)
        with pytest.raises(FloatingPointError,match="radius gate"):
            opt.step(torch.randn(4,3,32,32),torch.arange(4))
    assert opt.failed and opt.steps == 0 and not opt.base.state


def test_observed_checkpoint_pipeline_resume(config, isolated_root, monkeypatch):
    from vpr_sam.training import engine
    from test_pipeline import nested_equal
    config["optimizer"]["method"] = "fixed_l2"
    monkeypatch.setattr(engine, "Stepper", ObservedStepper)
    continuous=copy.deepcopy(config);continuous['output']='runs/l2_complete'
    split=copy.deepcopy(config);split['output']='runs/l2_resume'
    engine.train(continuous,isolated_root)
    engine.train(split,isolated_root,stop_after=2)
    engine.train(split,isolated_root,resume='runs/l2_resume/last.pt')
    a,_=engine.read_checkpoint('runs/l2_complete/final.pt',isolated_root)
    b,_=engine.read_checkpoint('runs/l2_resume/final.pt',isolated_root)
    for field in ('model','stepper','scheduler','rng','epoch_rows','order_hash'):
        assert nested_equal(a[field],b[field]),field
    assert a['schema']==2 and a['stepper']['geometry']['method']=='fixed_l2'
    pruned=engine.prune('runs/l2_complete/final.pt',isolated_root)
    result=engine.evaluate_checkpoint(pruned['checkpoint'],isolated_root,'validation',True)
    assert result['metrics']['count']==config['data']['validation_size']


def test_new_readonly_evaluation_helpers():
    import importlib.util
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('l2_evaluator',root/'scripts/vpr_l2_evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from vpr_sam.data import Images
    images=np.zeros((4096,32,32,3),dtype=np.uint8)
    labels=np.arange(4096)%10
    training=Images(images,labels,np.arange(4096),0)
    validation=Images(images,labels,np.arange(16),0)
    model=build_model('tiny');before=tensor_hash(dict(model.named_parameters()))
    result=module.bn_calibrate(model,training,validation)
    assert result['optimizer_steps']==0 and result['batches_per_bn']==32
    assert tensor_hash(dict(model.named_parameters()))==before
    assert not any(p.requires_grad for p in model.parameters())


def test_report_generation_full_history_schema(tmp_path):
    """Exercise deferred reporting now rather than discover a mismatch after 200ep."""
    import importlib.util, shutil
    from vpr_sam.repro import atomic_json,file_hash
    from vpr_sam.l2_protocol import OUTPUT
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('l2_report',root/'scripts/vpr_l2_report.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=tmp_path;module.OUT=tmp_path/'artifacts/vpr_l2';module.OUT.mkdir(parents=True)
    shutil.copytree(root/'docs/vpr_l2_reference',tmp_path/'docs/vpr_l2_reference')
    run=tmp_path/OUTPUT;run.mkdir(parents=True)
    for name in ('final','best_validation','last'):(run/f'{name}.pt').write_bytes(b'synthetic-report-fixture-only')
    hard=run/'pruning/n2_m4_exclude_first_last_magnitude_dense_final/pruned.pt'
    hard.parent.mkdir(parents=True);hard.write_bytes(b'synthetic-report-fixture-only')
    metrics=dict(loss=.2,accuracy=.9,count=5000)
    geom=dict(log_radius_min=-.2,log_radius_max=.1,log_radius_rms=.05)
    ep=[dict(epoch=i,train_loss=.2,adversarial_train_loss=.3,validation=metrics,geometry=geom) for i in range(1,201)]
    steps=[dict(step=i,epsilon_l2_over_rho=1.,log_volume=0.,direction_cosine=.9,linear_strength_over_spherical=.9) for i in range(1,70401)]
    atomic_json(run/'epochs.json',ep);atomic_json(run/'steps.json',steps)
    atomic_json(run/'summary.json',dict(elapsed_seconds=1.,peak_cuda_bytes=0))
    endpoint=[dict(R_epsilon=1.,direction_cosine=.9,linear_actual_ratio=.9,actual_loss_increase=.01,spherical_loss_increase=.02)]
    result=dict(status='PASS',commit='synthetic-fixture',config_sha256='fixture',dense={'metrics':metrics},
        raw_hard={'hard_validation':metrics,'hard_loss_increase':0.,'accuracy_drop':0.},
        bn_calibrated_hard={'metrics':metrics},geometry=geom,endpoint=endpoint,
        checkpoint_sha256={'final':file_hash(run/'final.pt')})
    atomic_json(module.OUT/'evaluation.json',result)
    module.main()
    assert (tmp_path/'VPR_L2_FINAL_AUDIT.md').exists()
    assert len(list((module.OUT/'plots').glob('*.png')))==7
    assert json.loads((module.OUT/'final_audit.json').read_text())['status']=='PASS'
