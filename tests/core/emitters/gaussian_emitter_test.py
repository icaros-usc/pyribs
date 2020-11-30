"""Tests for the GaussianEmitter."""
import numpy as np
import pytest

from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name


def test_properties_are_correct(_archive_fixture):
    archive, x0 = _archive_fixture
    sigma0 = 1
    emitter = GaussianEmitter(x0, sigma0, archive, batch_size=2)

    assert (emitter.x0 == x0).all()
    assert emitter.sigma0 == sigma0
    assert emitter.lower_bounds == -np.inf
    assert emitter.upper_bounds == np.inf


def test_tuple_bound_correct(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = GaussianEmitter(x0, 1, archive, bounds=(-1, 1))
    assert emitter.lower_bounds == -1
    assert emitter.upper_bounds == 1


def test_bad_tuple_bound_fails(_archive_fixture):
    archive, x0 = _archive_fixture
    with pytest.raises(ValueError):
        GaussianEmitter(x0, 1, archive, bounds=(-1, 0, 1))


def test_array_bound_correct(_archive_fixture):
    archive, x0 = _archive_fixture
    bounds = []
    for i in range(len(x0) - 1):
        bounds.append((-i, i))
    bounds.append(None)
    emitter = GaussianEmitter(x0, 1, archive, bounds=bounds)

    lower_bounds = np.concatenate((-np.arange(len(x0) - 1), [-np.inf]))
    upper_bounds = np.concatenate((np.arange(len(x0) - 1), [np.inf]))

    assert (emitter.lower_bounds == lower_bounds).all()
    assert (emitter.upper_bounds == upper_bounds).all()


def test_long_array_bound_fails(_archive_fixture):
    archive, x0 = _archive_fixture
    bounds = [(-1, 1)] * (len(x0) + 1)  # More bounds than solution dims.
    with pytest.raises(ValueError):
        GaussianEmitter(x0, 1, archive, bounds=bounds)


def test_array_bound_bad_entry_fails(_archive_fixture):
    archive, x0 = _archive_fixture
    bounds = [(-1, 1)] * len(x0)
    bounds[0] = (-1, 0, 1)  # Invalid entry.
    with pytest.raises(ValueError):
        GaussianEmitter(x0, 1, archive, bounds=bounds)


def test_upper_bounds_enforced(_archive_fixture):
    archive, _ = _archive_fixture
    emitter = GaussianEmitter([2, 2], 0, archive, bounds=(-1, 1))
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(_archive_fixture):
    archive, _ = _archive_fixture
    emitter = GaussianEmitter([-2, -2], 0, archive, bounds=(-1, 1))
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
