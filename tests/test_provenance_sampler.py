import json
from nmsam.experiment.artifacts import save_result,config_fingerprint
from nmsam.data.sampler import StatefulBatchSampler


def test_fixture_artifact_device_config_and_source_hash(tmp_path):
    config={"device":"cuda","rho":.05,"archive_path":"${ARCHIVE}","seed":3}
    alternate={**config,"archive_path":"${OTHER_ARCHIVE}"}
    assert config_fingerprint(config)==config_fingerprint(alternate)
    assert config_fingerprint(config)!=config_fingerprint({**config,"rho":.1})
    path,result=save_result("unit_fixture",config,{"status":"UNIT_FIXTURE","config_raw":"rho: 0.05"},tmp_path)
    assert result["device"]=="cuda" and result["source_config_sha256"]
    assert json.loads(path.read_text())["status"]=="UNIT_FIXTURE"


def test_drop_last_sampler_progress_without_training():
    sampler=StatefulBatchSampler(5,2,drop_last=True)
    batches=list(sampler)
    assert len(batches)==2 and all(len(batch)==2 for batch in batches) and sampler.cursor==5
    sampler.next_epoch();assert sampler.epoch==1 and sampler.cursor==0
