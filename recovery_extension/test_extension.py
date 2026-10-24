import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('recovery_extension_runner',Path(__file__).with_name('runner.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_constant_curve_area_and_gaps():
    rows=[dict(epoch=i,validation=dict(accuracy=.8,loss=.4)) for i in range(1,21)]
    result=module.curve_metrics(rows,dict(accuracy=.8,loss=.4),dict(accuracy=.9,loss=.3))
    assert result['accuracy_aulc_0_20']==pytest.approx(16.)
    assert result['accuracy_aulc_normalized']==pytest.approx(.8)
    assert result['accuracy_aulc_1_20']==pytest.approx(.8)
    assert result['hard_to_recovery_accuracy_gain']==0
    assert result['dense_to_recovery_accuracy_gap']==pytest.approx(.1)


def test_linear_curve_and_invalid_epochs():
    rows=[dict(epoch=i,validation=dict(accuracy=.5+i*.02,loss=1-i*.03)) for i in range(1,21)]
    result=module.curve_metrics(rows,dict(accuracy=.5,loss=1.),dict(accuracy=.95,loss=.25))
    assert result['accuracy_aulc_normalized']==pytest.approx(.7)
    assert result['hard_to_recovery_accuracy_gain']==pytest.approx(.4)
    assert result['dense_to_recovery_accuracy_gap']==pytest.approx(.05)
    with pytest.raises(ValueError):module.curve_metrics(rows[:-1],{'accuracy':.5,'loss':1.},{'accuracy':.95,'loss':.25})
    broken=copy.deepcopy(rows);broken[2]['validation']['loss']=float('nan')
    with pytest.raises(ValueError):module.curve_metrics(broken,{'accuracy':.5,'loss':1.},{'accuracy':.95,'loss':.25})


def test_historical_curves_are_audited_endpoints():
    import json
    from vpr_sam.config import recovery_config
    root=Path(__file__).resolve().parents[1]/'vpr_sam_research'
    historical=json.loads((root/'docs/vpr_l2_reference/v1_final_audit.json').read_text())
    c=json.loads((root/'docs/vpr_l2_reference/v1_vpr_seed0.json').read_text())
    rc=recovery_config(c)
    assert rc['optimizer']['method']=='sgd' and rc['optimizer']['lr']==.005 and rc['training']['epochs']==20
    for record in historical['records']:
        if record['seed']!=0:continue
        epochs=json.loads((Path(__file__).parent/f'reference/{record["method"]}_seed0_recovery_epochs.json').read_text())
        assert epochs[-1]['validation']==record['recovered']
        assert epochs[-1]['step']==7040
        assert module.curve_metrics(epochs,record['hard'],record['dense'])['final']==record['recovered']
