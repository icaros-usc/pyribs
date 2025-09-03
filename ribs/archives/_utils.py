"""Utilities specific to archives."""

from __future__ import annotations

from types import ModuleType

import numpy as np
from numpy.typing import DTypeLike


def parse_dtype(dtype: DTypeLike, xp: ModuleType) -> DTypeLike:
    """Makes any necessary modifications to the input dtype.

    Any dtypes that are `None` are set to the default "real floating" dtype for the
    provided array backend.
    """
    if dtype is None:
        return xp.__array_namespace_info__().default_dtypes()["real floating"]  # ty: ignore[unresolved-attribute]
    return dtype


def validate_cma_mae_settings(learning_rate, threshold_min, dtype):
    """Checks variables related to CMA-MAE, i.e., learning_rate and threshold_min."""
    if threshold_min != -np.inf and learning_rate is None:
        raise ValueError(
            "threshold_min was set without setting learning_rate. "
            "Please note that threshold_min is only used in CMA-MAE; "
            "it is not intended to be used for only filtering archive "
            "solutions. To run CMA-MAE, please also set learning_rate."
        )
    if learning_rate is None:
        learning_rate = 1.0  # Default value.
    if threshold_min == -np.inf and learning_rate != 1.0:
        raise ValueError("threshold_min can only be -np.inf if learning_rate is 1.0")
    learning_rate = np.asarray(learning_rate, dtype=dtype)
    threshold_min = np.asarray(threshold_min, dtype=dtype)
    return learning_rate, threshold_min


def fill_sentinel_values(occupied, data):
    """Fills unoccupied entries in data with sentinel values.

    Operates in-place on `data`.
    """
    unoccupied = ~occupied

    for name, arr in data.items():
        if arr.dtype == object:
            fill_val = None
        elif name == "index":
            fill_val = -1
        elif np.issubdtype(arr.dtype, np.integer):
            fill_val = 0
        else:  # Floating-point and other fields.
            fill_val = np.nan
        arr[unoccupied] = fill_val
