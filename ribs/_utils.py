"""Miscellaneous internal utilities."""


def check_measures_batch_shape(measures_batch, measure_dim):
    """Checks the shape of a batch of measures."""
    shape = measures_batch.shape
    if len(shape) != 2 or shape[1] != measure_dim:
        raise ValueError(
            "Expected measures_batch to be a 2D array with shape "
            f"(batch_size, measure_dim) (i.e. (batch_size, {measure_dim})) "
            f"but it had shape {shape}")


def check_measures_shape(measures, measure_dim):
    pass
