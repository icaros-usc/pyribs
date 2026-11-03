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


def _buffer_after(batch_sizes, seed, buffer_size=100):
    """Buffer after adding measures 0, 1, 2, ... in batches of the given sizes."""
    archive = DensityArchive(
        measure_dim=1,
        buffer_size=buffer_size,
        bandwidth=1.0,
        seed=seed,
    )
    measures = np.arange(sum(batch_sizes), dtype=np.float64)[:, None]
    start = 0
    for size in batch_sizes:
        archive.add(None, None, measures[start : start + size])
        start += size
    return archive.buffer[:, 0]


def test_reservoir_independent_of_batch_size():
    # Test that the buffer is the same even when different batch sizes are used while
    # inserting into the archive. The buffer should always be the same because reservoir
    # sampling always adds solutions one at a time, and it is using the same random seed
    # in all these cases.
    one_at_a_time = _buffer_after([1] * 2_000, seed=42)
    one_batch = _buffer_after([100, 1_900], seed=42)
    uneven = _buffer_after([37, 63, 500, 1, 999, 400], seed=42)

    assert np.all(one_batch == one_at_a_time)
    assert np.all(uneven == one_at_a_time)


def test_reservoir_is_uniform_sample():
    # After 10,100 measures 0, 1, 2, ..., each measure is in the buffer of 100
    # with the same probability, so the buffer holds distinct measures whose mean
    # is close to 5049.5.
    means = []
    for seed in range(50):
        buffer = _buffer_after([100, 10_000], seed=seed)
        assert len(np.unique(buffer)) == 100
        means.append(buffer.mean())

    # The standard error of the mean (SEM) over 50 seeds is about 41. 145 is
    # approximately 3.5 * SEM, which should capture the correct value in 99.95% of the
    # cases.
    assert abs(np.mean(means) - 5049.5) < 145.0


def test_dtype():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
        dtype=np.float32,
    )
    assert archive._measures_dtype == np.float32  # pylint: disable = protected-access


def test_measures_dtype():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=10000,
        density_method="kde",
        bandwidth=2.0,
        measures_dtype=np.float32,
    )
    assert archive._measures_dtype == np.float32  # pylint: disable = protected-access


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
