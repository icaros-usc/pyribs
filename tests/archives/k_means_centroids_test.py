"""Tests for k_means_centroids."""

import numpy as np
import pytest

from ribs.archives import k_means_centroids


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_basic(dtype):
    """The outputs should have the right shape, dtypes, bounds, etc."""
    centroids, samples = k_means_centroids(
        centroids=100,
        ranges=[(-1, 1), (-1, 1)],
        samples=1000,
        dtype=dtype,
        seed=42,
    )

    assert centroids.shape == (100, 2)
    assert centroids.dtype == dtype
    assert np.all(centroids >= [-1, -1])
    assert np.all(centroids <= [1, 1])

    assert samples.shape == (1000, 2)
    assert samples.dtype == dtype
    assert np.all(samples >= [-1, -1])
    assert np.all(samples <= [1, 1])


def test_samples_bad_shape():
    with pytest.raises(
        ValueError, match=r"Expected samples to be an array with shape .*"
    ):
        k_means_centroids(
            centroids=100,
            ranges=[(-1, 1), (-1, 1)],
            # The measure space is 2D but samples is 3D.
            samples=np.zeros((1000, 3)),
            dtype=np.float64,
            seed=42,
        )
