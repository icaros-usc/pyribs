"""Miscellaneous internal utilities."""


def _format_extra_msg(extra_msg):
    """Formats extra_msg if it exists."""
    return f" {extra_msg}" if len(extra_msg) > 0 else ""


def check_batch_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (batch_size, dim).

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"Expected {array_name} to be a 2D array with shape "
                         f"(batch_size, {dim}) (i.e. shape "
                         f"(batch_size, {dim_name})) but it had shape "
                         f"{array.shape}.{_format_extra_msg(extra_msg)}")


def check_1d_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (dim,).

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 1 or array.shape[0] != dim:
        raise ValueError(
            f"Expected {array_name} to be a 1D array with shape "
            f"({dim},) (i.e. shape ({dim_name},)) but it had shape "
            f"{array.shape}.{_format_extra_msg(extra_msg)}")


def check_is_1d(array, array_name, extra_msg=""):
    """Checks that an array is 1D."""
    if array.ndim != 1:
        raise ValueError(f"Expected {array_name} to be a 1D array but it had "
                         f"shape {array.shape}.{_format_extra_msg(extra_msg)}")


def check_solution_batch_dim(array, array_name, batch_size, extra_msg=""):
    """Checks the batch dimension of an array with respect to solution_batch."""
    if array.shape[0] != batch_size:
        raise ValueError(f"{array_name} does not match the batch dimension of "
                         "solution_batch -- since solution_batch has shape "
                         f"({batch_size}, ...), {array_name} should have shape "
                         f"({batch_size}, ...), but it has shape "
                         f"{array.shape}.{_format_extra_msg(extra_msg)}")
