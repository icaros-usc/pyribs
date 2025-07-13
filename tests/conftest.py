"""Shared functionality for all tests."""
import numpy as np
import pytest
from array_api_compat import array_namespace

# Array API backend handling. Adapted from:
# https://github.com/scipy/scipy/blob/888ca356eda34481e0e32b1be48c1262077d79a7/scipy/conftest.py#L283
xp_available_backends = [
    pytest.param(
        (np, None),  # `None` should default to cpu for numpy.
        id="numpy-cpu",
    ),
]

try:
    import torch

    xp_available_backends.append(
        pytest.param((torch, torch.device("cpu")), id="torch-cpu"))

    if torch.cuda.is_available():
        xp_available_backends.append(
            pytest.param((torch, torch.device("cuda")), id="torch-cuda"))
except ImportError:
    pass


@pytest.fixture(params=xp_available_backends)
def xp_and_device(request):
    """Run the test that uses this fixture on each available array API library
    and device."""
    xp, device = request.param
    xp = array_namespace(xp.empty(0))
    return xp, device
