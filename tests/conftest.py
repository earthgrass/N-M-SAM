import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_tests():
    torch.set_num_threads(1)
    torch.manual_seed(11)

