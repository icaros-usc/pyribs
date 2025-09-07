"""Custom data types for pyribs."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import DTypeLike

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

## General types ##

#: General type for integers.
Int = Union[int, np.integer]

#: General type for floats.
Float = Union[float, np.floating]

#: Represents data about a batch of solutions. The first dimension of each entry should
#: be the batch dimension.
BatchData = dict[str, np.ndarray]

#: Represents data about a single solution.
SingleData = dict[str, Any]

#: Description of fields for archives.
FieldDesc = dict[str, tuple[Int | tuple[Int, ...], DTypeLike]]

## Types related to array API ##

#: (For array API use only.) Represents an array compatible with pyribs.
Array = np.ndarray
#: (For array API use only.) Represents a dtype compatible with pyribs.
DType = np.dtype
#: (For array API use only.) Represents an array's device.
Device = Union[str, int]

# Modify types based on which array backends are available.
if is_torch_available:
    Array = Union[Array, torch.Tensor]
    DType = Union[DType, torch.dtype]
    Device = Union[Device, torch.device]
if is_cp_availalbe:
    Array = Union[Array, cp.ndarray]
    DType = Union[DType, cp.dtype]
    Device = Union[Device, cp.cuda.Device]
