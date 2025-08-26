"""Custom data types for pyribs."""

# Developer Note: When adding new types, make sure to update the API listing in
# `docs/api/ribs.typing.rst`.

from __future__ import annotations

from typing import Any, Union

import numpy as np

try:
    import torch

    is_torch_available = True
except ImportError:
    is_torch_available = False

try:
    import cupy as cp

    is_cp_availalbe = True
except ImportError:
    is_cp_availalbe = False

#: Represents data about a batch of solutions. The first dimension of each entry should
#: be the batch dimension.
BatchData = dict[str, np.ndarray]

#: Represents data about a single solution.
SingleData = dict[str, Any]

#: General type for integers.
Int = Union[int, np.integer]

#: General type for floats.
Float = Union[float, np.floating]

#: Represents an array compatible with pyribs.
Array = np.ndarray
if is_torch_available:
    Array = Union[Array, torch.Tensor]
if is_cp_availalbe:
    Array = Union[Array, cp.ndarray]

#: Represents an array's device.
Device = Union[str, int]
if is_torch_available:
    Device = Union[Device, torch.device]
if is_cp_availalbe:
    Device = Union[Device, cp.cuda.Device]
