"""Custom data types for pyribs."""

# Developer Note: When adding new types, make sure to update the API listing in
# `docs/api/ribs.typing.rst`.

from __future__ import annotations

from typing import Any, Literal, TypeVar, Union

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

#: General type for integers.
Int = Union[int, np.integer]

#: General type for floats.
Float = Union[float, np.floating]

#: Represents an array compatible with pyribs.
Array = np.ndarray
#: Represents a dtype compatible with pyribs.
DType = np.dtype
#: Represents an array's device.
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

#: TypeVar for arrays; can be used, e.g., to indicate that input and output arrays are
#: the same type.
ArrayVar = TypeVar("ArrayVar")

#: Represents data about a batch of solutions. The first dimension of each entry should
#: be the batch dimension.
BatchData = dict[str, Array]

#: Same as above but allowing for optional values.
OptionalBatchData = dict[str, Array | None]

#: Represents data about a single solution.
SingleData = dict[str, Any]

#: Description of fields for archives.
FieldDesc = dict[str, tuple[Int | tuple[Int, ...], DTypeLike]]

#: Description of dtypes for archives.
ArchiveDType = Union[
    Literal["f", "d"], type[Union[np.float32, np.float64]], dict[str, DTypeLike]
]
