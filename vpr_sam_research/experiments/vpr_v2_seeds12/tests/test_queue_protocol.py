import importlib.util,sys
from pathlib import Path
import pytest,yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aggregate import mean_sd_ci
import overnight

def test_all_four_configs_pair_and_keep_frozen_protocol():
    for seed in (1,2):
        sam=yaml.safe_load((ROOT/'configs'/f'sam_seed{seed}.yaml').read_text())
        vpr=yaml.safe_load((ROOT/'configs'/f'vpr_seed{seed}.yaml').read_text())
        for key,value in sam.items():
            if key not in ('experiment','output'):assert vpr[key]==value,(seed,key)
        assert sam['seed']==vpr['seed']==seed
        assert sam['epochs']==200 and sam['train_size']==50000 and sam['test_size']==10000
        assert sam['bn_rule']=='both_train_forwards_update_running_buffers'
        assert vpr['optimizer_method']=='fixed_l2' and vpr['geometry']['gamma']==5.482145661475941e-8
        assert vpr['recovery']['epochs']==20 and vpr['recovery']['save_each_epoch']
        assert f'sam_seed{seed}' in vpr['anchor']['root']

def test_uncertainty_unit_and_sample_sd():
    result=mean_sd_ci([1.,2.,3.]);assert result['mean']==2 and result['sample_sd']==1 and result['n_seeds']==3
    with pytest.raises(ValueError):mean_sd_ci([1.,2.])
    with pytest.raises(ValueError):mean_sd_ci([1.,2.,float('nan')])

def test_source_manifest_keeps_official_data_code_and_excludes_artifacts(tmp_path):
    source=tmp_path/'workers/vpr/reference/ssam2022/data/build.py';source.parent.mkdir(parents=True);source.write_text('pass\n')
    output=tmp_path/'workers/vpr/artifacts/test.json';output.parent.mkdir(parents=True);output.write_text('{}')
    dataset=tmp_path/'data/metadata.json';dataset.parent.mkdir();dataset.write_text('{}')
    assert set(overnight.source_files(tmp_path))=={'workers/vpr/reference/ssam2022/data/build.py'}

def test_deployment_records_byte_identical_config_alias(tmp_path,monkeypatch):
    root=tmp_path/'bundle';root.mkdir();(root/'configs').mkdir()
    for method in ('sam','vpr'):
        worker=root/'workers'/method;worker.mkdir(parents=True)
        (worker/'train.py').write_text('pass\n');(worker/'config.yaml').write_text('seed: 0\n')
        for seed in (1,2):(root/'configs'/f'{method}_seed{seed}.yaml').write_text(f'seed: {seed}\n')
    monkeypatch.setattr(overnight,'ROOT',root);monkeypatch.setattr(overnight,'HOME',tmp_path)
    freeze={'commit':'frozen','branch':'test','source_sha256':overnight.source_files(root)}
    paths=overnight.deploy(freeze)
    assert [p.parent.name for p in paths]==['sam_seed1','vpr_seed1','sam_seed2','vpr_seed2']
    import json
    first=json.loads((paths[0]/'SOURCE_FREEZE.json').read_text())
    assert first['deployment_git_paths']['config.yaml']=='configs/sam_seed1.yaml'
    assert (paths[0]/'config.yaml').read_bytes()==(root/'configs/sam_seed1.yaml').read_bytes()
    with pytest.raises(FileExistsError):overnight.deploy(freeze)
