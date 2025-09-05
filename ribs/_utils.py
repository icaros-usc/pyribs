"""Miscellaneous internal utilities."""

from __future__ import annotations

import numbers
from types import ModuleType

import array_api_compat.numpy as np_compat
import numpy as np
from array_api_compat import array_namespace
from numpy.typing import ArrayLike

from ribs.archives._archive_base import ArchiveBase
from ribs.typing import ArrayVar, BatchData, Int, SingleData


def check_finite(x: ArrayLike, name: str) -> None:
    """Checks that x is finite (i.e. not infinity or NaN).

    `x` must be either a scalar or NumPy array.
    """
    if not np.all(np.isfinite(x)):
        if np.isscalar(x):
            raise ValueError(
                f"{name} must be finite (infinity and NaN values are not supported)."
            )
        raise ValueError(
            f"All elements of {name} must be finite (infinity "
            "and NaN values are not supported)."
        )


def check_batch_shape(
    array: np.ndarray,
    array_name: str,
    dim: Int | tuple[Int, ...],
    dim_name: str,
    extra_msg: str = "",
) -> None:
    """Checks that the array has shape (batch_size, dim) or (batch_size, *dim).

    `batch_size` can be any value.

    `array` must be a numpy array, and `dim` must be an int or tuple of int.
    """
    if isinstance(dim, numbers.Integral):
        dim = (dim,)
    if array.ndim != 1 + len(dim) or array.shape[1:] != dim:
        dim_str = ", ".join(map(str, dim))
        raise ValueError(
            f"Expected {array_name} to be an array with shape "
            f"(batch_size, {dim_str}) (i.e. shape "
            f"(batch_size, {dim_name})) but it had shape "
            f"{array.shape}.{extra_msg}"
        )


def check_shape(
    array: np.ndarray,
    array_name: str,
    dim: Int | tuple[Int, ...],
    dim_name: str,
    extra_msg: str = "",
) -> None:
    """Checks that the array has shape dim.

    `array` must be a numpy array, and `dim` must be an int or tuple of int.
    """
    if isinstance(dim, numbers.Integral):
        dim = (dim,)
    if array.ndim != len(dim) or array.shape != dim:
        comma = "," if len(dim) == 1 else ""
        raise ValueError(
            f"Expected {array_name} to be an array with shape "
            f"{dim} (i.e. shape ({dim_name}{comma})) but it had shape "
            f"{array.shape}.{extra_msg}"
        )


def check_is_1d(array: np.ndarray, array_name: str, extra_msg: str = "") -> None:
    """Checks that an array is 1D."""
    if array.ndim != 1:
        raise ValueError(
            f"Expected {array_name} to be a 1D array but it had "
            f"shape {array.shape}.{extra_msg}"
        )


def check_solution_batch_dim(
    array: np.ndarray,
    array_name: str,
    batch_size: int,
    is_1d: bool = False,
    extra_msg: str = "",
) -> None:
    """Checks the batch dimension of an array with respect to the solutions."""
    if array.shape[0] != batch_size:
        raise ValueError(
            f"{array_name} does not match the batch dimension of "
            "solution -- since solution has shape "
            f"({batch_size}, ..), {array_name} should have shape "
            f"({batch_size},{'' if is_1d else ' ..'}), but it has "
            f"shape {array.shape}.{extra_msg}"
        )


def validate_batch(
    archive: ArchiveBase,
    data: BatchData,
    add_info: BatchData | None = None,
    jacobian: ArrayLike = None,
    none_objective_ok: bool = False,
):  # No return annotation because it's quite complicated.
    """Preprocesses and validates batch arguments.

    ``data`` is a dict containing arrays with the data of each solution, e.g., objective
    and measures. The batch size of each argument in the data is validated with respect
    to data["solution"].

    The arguments are assumed to come directly from users, so they may not be arrays.
    Thus, we preprocess each argument by converting it into a numpy array. We then
    perform checks on the array, including seeing if its batch size matches the batch
    size of data["solution"].
    """
    # Process and validate solutions.
    data["solution"] = np.asarray(data["solution"])
    check_batch_shape(
        data["solution"], "solution", archive.solution_dim, "solution_dim", ""
    )
    batch_size = data["solution"].shape[0]

    # Process and validate the other data.
    for name, arr in data.items():
        if name == "solution":
            # Already checked above.
            continue

        if name == "objective":
            if arr is None:
                # `objective` allowed to be None for diversity optimization.
                if not none_objective_ok:
                    raise ValueError("objective cannot be None")
            else:
                arr = np.asarray(arr)
                check_is_1d(arr, "objective", "")
                check_solution_batch_dim(
                    arr, "objective", batch_size, is_1d=True, extra_msg=""
                )
                check_finite(arr, "objective")

        elif name == "measures":
            arr = np.asarray(arr)
            check_batch_shape(arr, "measures", archive.measure_dim, "measure_dim", "")
            check_solution_batch_dim(
                arr, "measures", batch_size, is_1d=False, extra_msg=""
            )
            if np.issubdtype(arr.dtype, np.number):
                check_finite(arr, "measures")

        else:
            arr = np.asarray(arr)
            check_solution_batch_dim(arr, name, batch_size, is_1d=False, extra_msg="")

        data[name] = arr

    extra_returns = []

    # add_info is optional; check it if provided.
    if add_info is not None:
        for name, arr in add_info.items():
            if name == "status":
                arr = np.asarray(arr)
                check_is_1d(arr, "status", "")
                check_solution_batch_dim(
                    arr, "status", batch_size, is_1d=True, extra_msg=""
                )
                check_finite(arr, "status")

            elif name == "value":
                arr = np.asarray(arr)
                check_is_1d(arr, "value", "")
                check_solution_batch_dim(
                    arr, "value", batch_size, is_1d=True, extra_msg=""
                )

            else:
                arr = np.asarray(arr)
                check_solution_batch_dim(
                    arr, name, batch_size, is_1d=False, extra_msg=""
                )

            add_info[name] = arr

        extra_returns.append(add_info)

    # jacobian is optional; check it if provided.
    if jacobian is not None:
        jacobian = np.asarray(jacobian)
        check_batch_shape(
            jacobian,
            "jacobian",
            (archive.measure_dim + 1, archive.solution_dim),
            "measure_dim + 1, solution_dim",
        )
        check_finite(jacobian, "jacobian")
        extra_returns.append(jacobian)

    if extra_returns:
        return data, *extra_returns
    else:
        return data


def validate_single(
    archive: ArchiveBase, data: SingleData, none_objective_ok: bool = False
) -> SingleData:
    """Performs preprocessing and checks for arguments to add_single()."""
    data["solution"] = np.asarray(data["solution"])
    check_shape(data["solution"], "solution", archive.solution_dim, "solution_dim")

    if data["objective"] is None:
        if not none_objective_ok:
            raise ValueError("objective cannot be None")
    else:
        data["objective"] = np.asarray(
            data["objective"], dtype=archive.dtypes["objective"]
        )
        check_finite(data["objective"], "objective")

    data["measures"] = np.asarray(data["measures"])
    check_shape(data["measures"], "measures", archive.measure_dim, "measure_dim")
    if np.issubdtype(data["measures"].dtype, np.number):
        check_finite(data["measures"], "measures")

    return data


def arr_readonly(arr: ArrayVar, view: bool = False) -> ArrayVar:
    """Sets an array to be readonly if possible.

    Intended to support arrays across libraries; currently only supports numpy. Other
    arrays are returned as is.

    Pass `view` to call `arr.view` when the array is a numpy array. This is useful if
    you will still need to modify the array, as you can return a readonly view that does
    not make the original array readonly.
    """
    if isinstance(arr, np.ndarray):
        readonly_arr = arr.view() if view else arr
        readonly_arr.flags.writeable = False
        return readonly_arr  # ty: ignore[invalid-return-type]
    else:
        return arr


def xp_namespace(xp: ModuleType | None) -> ModuleType:
    """Utility for retrieving a namespace compatible with the array API.

    Expects to receive an argument like `torch` or `numpy`.

    Adapted from scipy:
    https://github.com/scipy/scipy/blob/4d3dcc103612a2edaec7069638b7f8d0d75cab8b/scipy/signal/windows/_windows.py#L44-L50

    For more context, see:
    https://github.com/data-apis/array-api-compat/issues/342
    """
    return np_compat if xp is None else array_namespace(xp.empty(0))  # ty: ignore[unresolved-attribute]


## Deprecations ##


def deprecate_dtype(dtype: None) -> None:
    if dtype is not None:
        raise ValueError(
            "dtype is deprecated as of pyribs 0.9.0. Please specify solution_dtype, "
            "objective_dtype, and/or measures_dtype instead."
        )
