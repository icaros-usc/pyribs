"""Tests for the DensityArchive."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import DensityArchive


def test_add_to_buffer():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
    )

    # All measures should add to the buffer since it is not full yet.
    archive.add(
        None,
        None,
        np.arange(10).reshape((5, 2)),
    )
    assert np.all(archive.buffer == np.arange(10).reshape((5, 2)))


def test_initial_density():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
    )

    measures = np.arange(10).reshape((5, 2))
    density = archive.compute_density(measures)
    add_info = archive.add(None, None, measures)

    assert_allclose(density, np.zeros(5))
    assert_allclose(add_info["density"], np.zeros(5))


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_density_dtype(dtype):
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
        dtype=dtype,
    )

    measures = np.arange(10).reshape((5, 2))
    add_info = archive.add(None, None, measures)
    density = archive.compute_density(measures)

    assert add_info["density"].dtype == dtype
    assert density.dtype == dtype


# TODO: Test density values -- after we add 10 values, the buffer should only
# consider those 10 values during density calculation.
