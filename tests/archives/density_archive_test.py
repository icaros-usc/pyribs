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


def test_dtype():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
        dtype=np.float32,
    )
    assert archive._measures_dtype == np.float32


def test_measures_dtype():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
        measures_dtype=np.float32,
    )
    assert archive._measures_dtype == np.float32


def test_simultaneous_dtypes():
    with pytest.raises(
        ValueError, match=r"dtype cannot be used at the same time as .*"
    ):
        DensityArchive(
            measure_dim=2,
            buffer_size=10000,
            density_method="kde",
            bandwidth=2.0,
            dtype=np.float32,
            measures_dtype=np.float32,
        )


@pytest.mark.parametrize("density_method", ["kde", "kde_sklearn"])
def test_initial_density(density_method):
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method=density_method,
        bandwidth=2.0,
    )

    measures = np.arange(10).reshape((5, 2))
    density = archive.compute_density(measures)
    add_info = archive.add(None, None, measures)

    assert_allclose(density, np.zeros(5))
    assert_allclose(add_info["density"], np.zeros(5))


@pytest.mark.parametrize("density_method", ["kde", "kde_sklearn"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_density_dtype(density_method, dtype):
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method=density_method,
        bandwidth=2.0,
        measures_dtype=dtype,
    )

    measures = np.arange(10).reshape((5, 2))
    add_info = archive.add(None, None, measures)
    density = archive.compute_density(measures)

    assert add_info["density"].dtype == dtype
    assert density.dtype == dtype


# We only test actual density for `kde` since that is the one for which we know
# the true value. `kde_sklearn` is a lot more complicated.
@pytest.mark.parametrize("density_method", ["kde"])
def test_density_after_add(density_method):
    bandwidth = 2.0
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method=density_method,
        bandwidth=bandwidth,
    )

    measures = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    archive.add(None, None, measures)
    density = archive.compute_density([[0, 0]])

    # This is the density computed with just one point like [-1, -1] which is
    # np.sqrt(2) away from [0, 0]. The density for all points is the same for
    # all points since all are equally far from [0, 0], and the final density is
    # the average, so it is the same as just one point.
    expected_density = (
        np.exp(-0.5 * np.square(np.sqrt(2) / bandwidth))
        / np.sqrt(2 * np.pi)
        / bandwidth
    )

    assert_allclose(density, np.array([expected_density]))
