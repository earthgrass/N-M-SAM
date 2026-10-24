import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import aggregate as a

def test_synthetic_three_seed_report_no_model_execution(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';bundle.mkdir();seed0=tmp_path/'seed0'
    monkeypatch.setattr(a,'ROOT',bundle);monkeypatch.setattr(a,'HOME',tmp_path);monkeypatch.setattr(a,'SEED0',seed0)
    monkeypatch.setattr(a,'verify_bundle',lambda:{'commit':'synthetic-only'})
    for seed in range(3):
        project=seed0 if seed==0 else tmp_path/'runs'/f'vpr_seed{seed}'/'project'
        comparison=project/'artifacts/comparison';comparison.mkdir(parents=True)
        parents={}
        for kind in ('sam','vpr'):
            p=tmp_path/'parents'/str(seed)/kind;p.mkdir(parents=True)
            (p/'best_test.pt').write_bytes((kind+str(seed)).encode())
            (p/'epoch_metrics.jsonl').write_text(''.join(json.dumps({'steps':i*390,'elapsed_seconds_before_publication':i,
                'peak_cuda_allocated_bytes':1000})+'\n' for i in range(1,201)))
            (p/'source_manifest.json').write_text(json.dumps({'identity':{'seed':seed,'commit':'synthetic',
                'config_sha256':'config','config_file_sha256':'file'}}))
            parents[kind]={'path':str(p/'best_test.pt'),'sha256':a.sha(p/'best_test.pt')}
        data=[]
        for method in a.METHODS:
            rows=[{'epoch':i,'steps':i*390,'test_accuracy':.94+i*.001+seed*.0001,'test_loss':.3-i*.001,
                   'order_sha256':f'{seed}_{i}','elapsed_seconds':i,'peak_cuda_allocated_bytes':100} for i in range(1,21)]
            data.append({'method':method,'dense':{'accuracy':.97,'loss':.1},'raw_hard':{'accuracy':.9,'loss':.4},
                'bn_only':{'accuracy':.93,'loss':.3},'recovery_best':rows[-1],'recovery_final':rows[-1],
                'recovery':{'epochs':rows,'best_sha256':'best','last_sha256':'last'},
                'parent':parents['sam' if method=='sam_magnitude' else 'vpr'],'mask_sha256':method,
                'accuracy_damage_pp':7.,'recovery_best_gain_pp':6.})
        (comparison/'summary.json').write_text(json.dumps(data))
        (comparison/'FINAL_AUDIT.json').write_text(json.dumps({'status':'PASS','checkpoint_sha256':{}}))
    a.main();out=bundle/'artifacts/final_seeds012'
    assert (out/'recovery_three_seed_curves.png').stat().st_size>1000
    import csv
    assert len(list(csv.DictReader((out/'per_seed_metrics.csv').open())))==9
    assert json.loads((out/'FINAL_AUDIT.json').read_text())['status']=='PASS'
