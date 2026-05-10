"""Tests for the DensityArchive that require Zuko.

These tests exercise the DensityArchive's support for DDS-CNF (Lee 2024 --
https://arxiv.org/abs/2312.11331).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import DensityArchive


def _make_cnf_archive(
    *,
    measure_dim: int = 2,
    buffer_size: int = 500,
    train_steps: int = 20,
    batch_size: int = 32,
    min_buffer_size: int = 32,
    seed: int = 0,
    cnf_kwargs: dict | None = None,
):
    """Build a small DDS-CNF archive suited for fast unit tests."""
    return DensityArchive(
        measure_dim=measure_dim,
        buffer_size=buffer_size,
        density_method="cnf",
        cnf_train_steps=train_steps,
        cnf_batch_size=batch_size,
        cnf_min_buffer_size=min_buffer_size,
        cnf_kwargs=cnf_kwargs or {"hidden_features": (16, 16)},
        seed=seed,
    )


def test_cnf_initial_density_is_zero():
    # Before the flow has ever been fit, compute_density must return zeros so
    # that the first scheduler step does not depend on an untrained flow. This
    # matches the KDE behavior on an empty buffer.
    archive = _make_cnf_archive()
    density = archive.compute_density(np.zeros((4, 2)))
    assert_allclose(density, np.zeros(4))
    assert archive._cnf_estimator.fitted is False  # pylint: disable = protected-access


def test_cnf_first_add_returns_zero_density():
    # Algorithm 1 line 10 of the DDS paper computes density BEFORE updating
    # the buffer. On the first call, the flow has not been trained yet, so
    # the returned density must be zero regardless of how many measures are
    # passed in -- matching how KDE behaves on an empty buffer.
    archive = _make_cnf_archive()
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((64, 2))
    add_info = archive.add(None, None, pts)
    assert_allclose(add_info["density"], np.zeros(64))


def test_cnf_trains_after_first_add():
    # After the first add() call, Algorithm 1 line 12 triggers a fit. A
    # subsequent compute_density() call should therefore return non-zero
    # log-density values reflecting the trained flow.
    archive = _make_cnf_archive()
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((64, 2)))

    assert archive._cnf_estimator.fitted is True  # pylint: disable = protected-access
    probe = rng.standard_normal((8, 2))
    density = archive.compute_density(probe)
    assert density.shape == (8,)
    assert np.any(density != 0.0)
    assert np.all(np.isfinite(density))


def test_cnf_density_shape_and_dtype():
    # compute_density must match the measures_dtype of the archive just like
    # the KDE variants do, so that downstream rankers and emitters can rely
    # on a predictable output dtype.
    archive = _make_cnf_archive()
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((64, 2)))
    density = archive.compute_density(rng.standard_normal((5, 2)))
    assert density.shape == (5,)
    assert density.dtype == np.float64


def test_cnf_below_min_buffer_does_not_fit():
    # If the buffer has fewer than cnf_min_buffer_size points, the flow must
    # stay untrained and density queries must stay at zero. This lets users
    # safely run small warm-up batches before enabling real density-based
    # selection.
    archive = _make_cnf_archive(min_buffer_size=128)
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((64, 2)))
    # pylint: disable = protected-access
    assert archive._cnf_estimator.fitted is False
    assert_allclose(archive.compute_density(np.zeros((3, 2))), np.zeros(3))


def test_cnf_differentiates_trained_distribution():
    # A well-trained CNF should assign higher log-density to points near the
    # training distribution than to points far from it. This is a
    # functional/sanity check that the training loop is actually minimizing
    # NLL on the buffer.
    archive = _make_cnf_archive(train_steps=200, min_buffer_size=64)
    rng = np.random.default_rng(0)
    # Train on a tight cluster centered at (5, 5).
    training_points = rng.standard_normal((256, 2)) * 0.3 + np.array([5.0, 5.0])
    archive.add(None, None, training_points)

    in_dist = archive.compute_density(np.array([[5.0, 5.0]]))
    out_of_dist = archive.compute_density(np.array([[-5.0, -5.0]]))
    # in_dist should be strictly larger (higher log-density) than out_of_dist.
    assert in_dist[0] > out_of_dist[0]


def test_cnf_rejects_features_kwarg():
    # `features` would silently break because it needs to match measure_dim.
    # Ensure the estimator rejects it with a clear error.
    with pytest.raises(ValueError, match="features"):
        DensityArchive(
            measure_dim=2,
            density_method="cnf",
            cnf_kwargs={"features": 3, "hidden_features": (8,)},
            cnf_min_buffer_size=32,
            cnf_train_steps=5,
            cnf_batch_size=16,
        )


def test_cnf_seed_is_reproducible():
    # Two archives constructed with the same seed and shown the same data
    # must produce identical log-density queries. This protects users who
    # rely on seeding for experiments in papers and benchmarks.
    rng = np.random.default_rng(42)
    training_points = rng.standard_normal((256, 2)) + 1.0
    probe = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    archive_a = _make_cnf_archive(train_steps=50, seed=7)
    archive_a.add(None, None, training_points)
    d_a = archive_a.compute_density(probe)

    archive_b = _make_cnf_archive(train_steps=50, seed=7)
    archive_b.add(None, None, training_points)
    d_b = archive_b.compute_density(probe)

    assert_allclose(d_a, d_b, rtol=0, atol=0)


def test_cnf_small_buffer_uses_full_batch():
    # If batch_size exceeds the buffer size, the estimator must cap the batch
    # to the buffer rather than attempting to sample out-of-bounds.
    archive = _make_cnf_archive(min_buffer_size=16, batch_size=128, train_steps=5)
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((32, 2)))
    # pylint: disable = protected-access
    assert archive._cnf_estimator.fitted is True
    density = archive.compute_density(rng.standard_normal((4, 2)))
    assert density.shape == (4,)
    assert np.all(np.isfinite(density))


def test_cnf_retrains_on_buffer_update():
    # Second add() call should trigger a second fit() call, updating the flow.
    # We verify this by checking that log-density on an out-of-distribution
    # probe changes after the second add().
    archive = _make_cnf_archive(train_steps=50)
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((64, 2)))
    probe = np.array([[5.0, 5.0]])
    d_before = archive.compute_density(probe)[0]

    # Add a big batch of points centered far away from the origin. This
    # should shift the trained flow noticeably.
    archive.add(None, None, rng.standard_normal((256, 2)) + np.array([5.0, 5.0]))
    d_after = archive.compute_density(probe)[0]

    assert d_after != d_before
    # The new distribution is much closer to the probe, so the log-density at
    # the probe should increase (less negative).
    assert d_after > d_before


def test_cnf_density_float32():
    archive = DensityArchive(
        measure_dim=2,
        buffer_size=500,
        density_method="cnf",
        cnf_train_steps=10,
        cnf_batch_size=32,
        cnf_min_buffer_size=32,
        cnf_kwargs={"hidden_features": (16, 16)},
        measures_dtype=np.float32,
        seed=0,
    )
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((64, 2)).astype(np.float32))
    density = archive.compute_density(rng.standard_normal((5, 2)).astype(np.float32))
    assert density.dtype == np.float32
    assert density.shape == (5,)


def test_cnf_higher_dim():
    archive = _make_cnf_archive(measure_dim=5, train_steps=30, min_buffer_size=64)
    rng = np.random.default_rng(0)
    archive.add(None, None, rng.standard_normal((128, 5)))
    density = archive.compute_density(rng.standard_normal((4, 5)))
    assert density.shape == (4,)
    assert np.all(np.isfinite(density))
    assert np.any(density != 0.0)


def test_cnf_unknown_method_still_raises():
    # Ensure adding the CNF branch didn't break the error path for bad
    # method names.
    with pytest.raises(ValueError, match="Unknown density_method"):
        DensityArchive(measure_dim=2, density_method="not_a_method")
