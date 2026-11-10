import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def pytest_sessionfinish(session,exitstatus):
    from benchmark.common import atomic_json,environment
    xml=session.config.getoption('xmlpath')
    target=Path(xml).with_suffix('.environment.json') if xml else Path(__file__).resolve().parents[1]/'artifacts/test_environment.json'
    atomic_json(target,
                {'environment_at_test_session_end':environment(),'exitstatus':exitstatus})

@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(2);torch.manual_seed(0)
    yield
