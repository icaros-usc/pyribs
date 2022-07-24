"""Tests for the GaussianEmitter."""
import numpy as np

from ribs.emitters import GaussianEmitter


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    sigma = 1
    sigma0 = 2
    emitter = GaussianEmitter(archive, x0, sigma, sigma0, batch_size=2)

    assert (emitter.x0 == x0).all()
    assert emitter.sigma == sigma
    assert emitter.sigma0 == sigma0


def test_sigma0_is_correct(archive_fixture):
    archive, x0 = archive_fixture
    sigma = 1
    emitter = GaussianEmitter(archive, x0, sigma)  # sigma0=None

    assert emitter.sigma0 == sigma


def test_upper_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GaussianEmitter(archive, [2, 2], 0, bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GaussianEmitter(archive, [-2, -2], 0, bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 0, batch_size=2)
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    parent_sol = x0 * 5
    archive.add_single(parent_sol, 1, np.array([0, 0]))
    emitter = GaussianEmitter(archive, x0, 0, batch_size=2)

    # All solutions should be generated "around" the single parent solution in
    # the archive.
    solutions = emitter.ask()

    assert (solutions == np.expand_dims(parent_sol, axis=0)).all()
