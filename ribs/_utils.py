"""Miscellaneous internal utilities."""


def check_measures_batch_shape(measures_batch,
                               measure_dim,
                               name="measures_batch"):
    """Checks the shape of a batch of measures.

    `measures_batch` must be a numpy array.
    """
    shape = measures_batch.shape
    if len(shape) != 2 or shape[1] != measure_dim:
        raise ValueError(f"Expected {name} to be a 2D array with shape "
                         f"(batch_size, {measure_dim}) (i.e. shape "
                         f"(batch_size, measure_dim)) but it had shape {shape}")


def check_measures_shape(measures, measure_dim):
    """Checks the shape of a 1D vector of measures.

    `measures` must be a numpy array.
    """
    shape = measures.shape
    if len(shape) != 1 or shape[0] != measure_dim:
        raise ValueError("Expected measures to be a 1D array with shape "
                         f"({measure_dim},) (i.e. shape (measure_dim,)) "
                         f"but it had shape {shape}")
