"""Miscellaneous internal utilities."""
import numpy as np


def check_finite(x, name):
    """Checks that x is finite (i.e. not infinity or NaN).

    `x` must be either a scalar or NumPy array.
    """
    if not np.all(np.isfinite(x)):
        if np.isscalar(x):
            raise ValueError(f"{name} must be finite (infinity "
                             "and NaN values are not supported).")
        raise ValueError(f"All elements of {name} must be finite (infinity "
                         "and NaN values are not supported).")


def check_batch_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (batch_size, dim).

    `batch_size` can be any value.

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"Expected {array_name} to be a 2D array with shape "
                         f"(batch_size, {dim}) (i.e. shape "
                         f"(batch_size, {dim_name})) but it had shape "
                         f"{array.shape}.{extra_msg}")


def check_1d_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (dim,).

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 1 or array.shape[0] != dim:
        raise ValueError(
            f"Expected {array_name} to be a 1D array with shape "
            f"({dim},) (i.e. shape ({dim_name},)) but it had shape "
            f"{array.shape}.{extra_msg}")


def check_is_1d(array, array_name, extra_msg=""):
    """Checks that an array is 1D."""
    if array.ndim != 1:
        raise ValueError(f"Expected {array_name} to be a 1D array but it had "
                         f"shape {array.shape}.{extra_msg}")


def check_batch_shape_3d(array,
                         array_name,
                         dim1,
                         dim1_name,
                         dim2,
                         dim2_name,
                         extra_msg=""):
    """Checks that the array has shape (batch_size, dim1, dim2).

    `batch_size` can be any value.

    `array` must be a numpy array, `dim1` and `dim2` must be int.
    """
    if array.ndim != 3 or array.shape[1] != dim1 or array.shape[2] != dim2:
        raise ValueError(f"Expected {array_name} to be a 3D array with shape "
                         f"(batch_size, {dim1}, {dim2}) (i.e. shape "
                         f"(batch_size, {dim1_name}, {dim2_name})) but it had"
                         f"shape {array.shape}.{extra_msg}")


def check_solution_batch_dim(array,
                             array_name,
                             batch_size,
                             is_1d=False,
                             extra_msg=""):
    """Checks the batch dimension of an array with respect to solution_batch."""
    if array.shape[0] != batch_size:
        raise ValueError(f"{array_name} does not match the batch dimension of "
                         "solution_batch -- since solution_batch has shape "
                         f"({batch_size}, ..), {array_name} should have shape "
                         f"({batch_size},{'' if is_1d else ' ..'}), but it has "
                         f"shape {array.shape}.{extra_msg}")


_BATCH_WARNING = (" Note that starting in pyribs 0.5.0, add() and tell() take"
                  " in a batch of solutions unlike in pyribs 0.4.0, where add()"
                  " and tell() only took in a single solution.")


def validate_batch_args(archive, solution_batch, **batch_kwargs):
    """Preprocesses and validates batch arguments.

    The batch size of each argument in batch_kwargs is validated with respect to
    solution_batch.

    The arguments are assumed to come directly from users, so they may not be
    arrays. Thus, we preprocess each argument by converting it into a numpy
    array. We then perform checks on the array, including seeing if its batch
    size matches the batch size of solution_batch. The arguments are then
    returned in the same order that they were passed into the kwargs, with
    solution_batch coming first.

    Note that we can guarantee the order is the same as when passed in due to
    PEP 468 (https://peps.python.org/pep-0468/), which guarantees that kwargs
    will preserve the same order as they are listed.

    See the for loop for the list of supported kwargs.
    """
    # List of args to return.
    returns = []

    # Process and validate solution_batch.
    solution_batch = np.asarray(solution_batch)
    check_batch_shape(solution_batch, "solution_batch", archive.solution_dim,
                      "solution_dim", _BATCH_WARNING)
    returns.append(solution_batch)

    # Process and validate the other batch arguments.
    batch_size = solution_batch.shape[0]
    for name, arg in batch_kwargs.items():
        if name == "objective_batch":
            objective_batch = np.asarray(arg)
            check_is_1d(objective_batch, "objective_batch", _BATCH_WARNING)
            check_solution_batch_dim(objective_batch,
                                     "objective_batch",
                                     batch_size,
                                     is_1d=True,
                                     extra_msg=_BATCH_WARNING)
            check_finite(objective_batch, "objective_batch")
            returns.append(objective_batch)
        elif name == "measures_batch":
            measures_batch = np.asarray(arg)
            check_batch_shape(measures_batch, "measures_batch",
                              archive.measure_dim, "measure_dim",
                              _BATCH_WARNING)
            check_solution_batch_dim(measures_batch,
                                     "measures_batch",
                                     batch_size,
                                     is_1d=False,
                                     extra_msg=_BATCH_WARNING)
            check_finite(measures_batch, "measures_batch")
            returns.append(measures_batch)
        elif name == "jacobian_batch":
            jacobian_batch = np.asarray(arg)
            check_batch_shape_3d(jacobian_batch, "jacobian_batch",
                                 archive.measure_dim + 1, "measure_dim + 1",
                                 archive.solution_dim, "solution_dim")
            check_finite(jacobian_batch, "jacobian_batch")
            returns.append(jacobian_batch)
        elif name == "status_batch":
            status_batch = np.asarray(arg)
            check_is_1d(status_batch, "status_batch", _BATCH_WARNING)
            check_solution_batch_dim(status_batch,
                                     "status_batch",
                                     batch_size,
                                     is_1d=True,
                                     extra_msg=_BATCH_WARNING)
            check_finite(status_batch, "status_batch")
            returns.append(status_batch)
        elif name == "value_batch":
            value_batch = np.asarray(arg)
            check_is_1d(value_batch, "value_batch", _BATCH_WARNING)
            check_solution_batch_dim(value_batch,
                                     "value_batch",
                                     batch_size,
                                     is_1d=True,
                                     extra_msg=_BATCH_WARNING)
            returns.append(value_batch)
        elif name == "metadata_batch":
            # Special case -- metadata_batch defaults to None in our methods,
            # but we make it into an array of None if it is not provided.
            metadata_batch = (np.empty(batch_size, dtype=object)
                              if arg is None else np.asarray(arg, dtype=object))
            check_is_1d(metadata_batch, "metadata_batch", _BATCH_WARNING)
            check_solution_batch_dim(metadata_batch,
                                     "metadata_batch",
                                     batch_size,
                                     is_1d=True,
                                     extra_msg=_BATCH_WARNING)
            returns.append(metadata_batch)

    return returns


def validate_single_args(archive, solution, objective, measures):
    """Performs preprocessing and checks for arguments to add_single()."""
    solution = np.asarray(solution)
    check_1d_shape(solution, "solution", archive.solution_dim, "solution_dim")

    objective = archive.dtype(objective)
    check_finite(objective, "objective")

    measures = np.asarray(measures)
    check_1d_shape(measures, "measures", archive.measure_dim, "measure_dim")
    check_finite(measures, "measures")

    return solution, objective, measures


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr
