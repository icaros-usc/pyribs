"""Utilities specific to archives."""
import numpy as np

from ribs._utils import np_scalar


def parse_dtype(dtype):
    """Parses dtypes for the archive."""
    # Convert str dtypes to np.dtype.
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # np.dtype is not np.float32 or np.float64, but it compares equal.
    if dtype in [np.float32, np.float64]:
        return {
            "solution": dtype,
            "objective": dtype,
            "measures": dtype,
        }
    elif isinstance(dtype, dict):
        if (
            "solution" not in dtype
            or "objective" not in dtype
            or "measures" not in dtype
        ):
            raise ValueError(
                "If dtype is a dict, it must contain 'solution',"
                "'objective', and 'measures' keys."
            )
        return dtype
    else:
        raise ValueError(
            "Unsupported dtype. Must be np.float32 or np.float64, "
            "or dict of the form "
            '{"solution": <dtype>, "objective": <dtype>, "measures": <dtype>}'
        )


def validate_cma_mae_settings(learning_rate, threshold_min, dtype):
    """Checks variables related to CMA-MAE, i.e., learning_rate and
    threshold_min."""
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
        raise ValueError(
            "threshold_min can only be -np.inf if " "learning_rate is 1.0"
        )
    learning_rate = np_scalar(learning_rate, dtype)
    threshold_min = np_scalar(threshold_min, dtype)
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
