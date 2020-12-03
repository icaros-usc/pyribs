"""Tests for the GaussianEmitter."""
import numpy as np

from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name


def test_properties_are_correct(_archive_fixture):
    archive, x0 = _archive_fixture
    sigma0 = 1
    emitter = GaussianEmitter(x0, sigma0, archive, batch_size=2)

    assert (emitter.x0 == x0).all()
    assert emitter.sigma0 == sigma0


def test_upper_bounds_enforced(_archive_fixture):
    archive, _ = _archive_fixture
    emitter = GaussianEmitter([2, 2], 0, archive, bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(_archive_fixture):
    archive, _ = _archive_fixture
    emitter = GaussianEmitter([-2, -2], 0, archive, bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_gauss_emits_x0(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = GaussianEmitter(x0, 0, archive, batch_size=2)
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(_archive_fixture):
    archive, x0 = _archive_fixture
    parent_sol = x0 * 5
    archive.add(parent_sol, 1, np.array([0, 0]))
    emitter = GaussianEmitter(x0, 0, archive, batch_size=2)

    # All solutions should be generated "around" the single parent solution in
    # the archive.
    solutions = emitter.ask()

    assert (solutions == np.expand_dims(parent_sol, axis=0)).all()
