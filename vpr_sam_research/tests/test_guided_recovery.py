import copy,json
from pathlib import Path
import pytest
import torch
from vpr_sam.config import recovery_config,validate_pruning
from vpr_sam.models import build_model
from vpr_sam.pruning import make_masks,apply_masks,validate_masks
from vpr_sam.guided_recovery_adapter import AuditedRecoveryStepper,arithmetic_aulc
from vpr_sam.repro import tensor_hash
from vpr_sam.optim import Stepper


def test_import_only_never_silently_magnitude(config):
    p=copy.deepcopy(config['pruning']);p['score']='geometry_guided'
    validate_pruning(p)
    with pytest.raises(ValueError,match='frozen mask import'):make_masks(build_model('tiny'),p)


@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
def test_sgd_mask_gradient_momentum_bn_and_scheduler(config,device):
    torch.manual_seed(18)
    c=recovery_config(config);c['pruning']['score']='geometry_guided'
    model=build_model('tiny').to(device)
    p=copy.deepcopy(c['pruning']);p['score']='magnitude'
    masks,_=make_masks(model,p)
    # Simulate imported frozen masks, independent from any score recomputation.
    apply_masks(model,masks);reference=copy.deepcopy(model)
    observed=AuditedRecoveryStepper(model,c,masks);plain=Stepper(reference,c,masks)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(observed.base,T_max=20)
    x,y=torch.randn(4,3,32,32,device=device),torch.arange(4,device=device)
    for _ in range(3):
        info=observed.step(x,y);plain.step(x,y)
        assert tensor_hash(model.state_dict())==tensor_hash(reference.state_dict())
        assert all(info[k]==0 for k in ['mask_violations','regrown_parameters','masked_gradient_violations','masked_momentum_violations'])
        validate_masks(model,masks,c['pruning'])
    assert int(model.bn.num_batches_tracked)==3 and observed.geometry.logs==[]
    schedule.step()


def test_arithmetic_aulc_exact_definition():
    rows=[dict(epoch=i,validation={'accuracy':.5+.02*i}) for i in range(1,21)]
    assert arithmetic_aulc(.5,rows)==pytest.approx(.7)
    assert arithmetic_aulc(.9,[dict(epoch=i,validation={'accuracy':.9}) for i in range(1,21)])==pytest.approx(.9)
    with pytest.raises(ValueError):arithmetic_aulc(.5,rows[1:])


def test_only_mask_and_identity_config_diff():
    from vpr_sam.l2_protocol import differences
    root=Path(__file__).resolve().parents[1]
    new=json.loads((root/'configs/approved_guided_recovery/seed0.json').read_text())
    old=json.loads((root/'docs/guided_recovery_reference/magnitude_recovery_config.json').read_text())
    assert [r['field'] for r in differences(old,recovery_config(new))]==['output','pruning.score']
