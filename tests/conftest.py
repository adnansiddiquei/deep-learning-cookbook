import pytest
import torch


@pytest.fixture
def device():
    return (
        'cude'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
