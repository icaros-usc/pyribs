"""Custom data types for pyribs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike

try:
    import torch

    IS_TORCH_AVAILABLE = True
except ImportError:
    IS_TORCH_AVAILABLE = False

try:
    import cupy as cp

    IS_CP_AVAILALBE = True
except ImportError:
    IS_CP_AVAILALBE = False

## General types ##

#: General type for integers.
Int = int | np.integer

#: General type for floats.
Float = float | np.floating

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
Device = str | int

# Modify types based on which array backends are available.
if IS_TORCH_AVAILABLE:
    Array = Array | torch.Tensor
    DType = DType | torch.dtype
    Device = Device | torch.device
if IS_CP_AVAILALBE:
    Array = Array | cp.ndarray
    DType = DType | cp.dtype
    Device = Device | cp.cuda.Device
