import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from state import seed_all

@pytest.fixture
def device():
    value=os.environ.get('ANCHOR_TEST_DEVICE','cpu')
    if value.startswith('cuda') and not torch.cuda.is_available():
        pytest.fail('CUDA explicitly required, unavailable')
    seed_all(0,threads=2)
    return value
