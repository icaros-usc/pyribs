"""Miscellaneous internal utilities."""


def check_batch_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (batch_size, dim).

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"Expected {array_name} to be a 2D array with shape "
                         f"(batch_size, {dim}) (i.e. shape "
                         f"(batch_size, {dim_name})) but it had shape "
                         f"{array.shape}{extra_msg}")


def check_1d_shape(array, array_name, dim, dim_name, extra_msg=""):
    """Checks that the array has shape (dim,).

    `array` must be a numpy array, and `dim` must be an int.
    """
    if array.ndim != 1 or array.shape[0] != dim:
        raise ValueError(f"Expected {array_name} to be a 1D array with shape "
                         f"({dim},) (i.e. shape ({dim_name},)) "
                         f"but it had shape {array.shape}{extra_msg}")
