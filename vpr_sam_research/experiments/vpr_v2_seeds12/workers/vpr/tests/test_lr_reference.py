import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from lr_reference import official_class,sequence,strict_compare
from scheduler import CosineLRscheduler

ROOT=Path(__file__).resolve().parents[1]


def test_native_all_400_fields_exact_and_one_ulp_rejected():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    official=sequence(official_class(ROOT),c); project=sequence(CosineLRscheduler,c)
    report=strict_compare(official,project,c['epochs'])
    assert report['status']=='PASS' and report['compared_fields']==400
    project[49]['used']=math.nextafter(project[49]['used'],math.inf)
    report=strict_compare(official,project,c['epochs'])
    assert report['status']=='FAIL' and report['differing_fields']==1


def test_native_schema_and_length_cannot_skip_fields():
    c=yaml.safe_load((ROOT/'config.yaml').read_text()); rows=sequence(official_class(ROOT),c)
    with pytest.raises(ValueError): strict_compare(rows,rows[:-1],200)
    malformed=copy.deepcopy(rows); malformed[0]['epoch']=2
    with pytest.raises(ValueError): strict_compare(rows,malformed,200)
    malformed=copy.deepcopy(rows); del malformed[0]['after_epoch']
    with pytest.raises(ValueError): strict_compare(rows,malformed,200)
