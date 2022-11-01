"""Tests for the GaussianEmitter."""
import numpy as np
import pytest

from ribs.emitters import GaussianEmitter


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    sigma = 1
    batch_size = 2
    emitter = GaussianEmitter(archive,
                              sigma=sigma,
                              x0=x0,
                              batch_size=batch_size)

    assert np.all(emitter.x0 == x0)
    assert emitter.sigma == sigma
    assert emitter.batch_size == batch_size


def test_initial_solutions_are_correct(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    emitter = GaussianEmitter(archive,
                              sigma=1.0,
                              initial_solutions=initial_solutions)

    assert np.all(emitter.ask() == initial_solutions)
    assert np.all(emitter.initial_solutions == initial_solutions)


def test_initial_solutions_shape(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 0, 0], [1, 1, 1]]

    # archive.solution_dim = 4
    with pytest.raises(ValueError):
        GaussianEmitter(archive, sigma=1.0, initial_solutions=initial_solutions)


def test_neither_x0_nor_initial_solutions_provided(archive_fixture):
    archive, _ = archive_fixture
    with pytest.raises(ValueError):
        GaussianEmitter(archive, sigma=1.0)


def test_both_x0_and_initial_solutions_provided(archive_fixture):
    archive, x0 = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    with pytest.raises(ValueError):
        GaussianEmitter(archive,
                        sigma=1.0,
                        x0=x0,
                        initial_solutions=initial_solutions)


def test_upper_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GaussianEmitter(
        archive,
        sigma=0,
        x0=[2, 2, 2, 2],
        bounds=[(-1, 1)] * 4,
    )
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GaussianEmitter(
        archive,
        sigma=0,
        x0=[-2, -2, -2, -2],
        bounds=[(-1, 1)] * 4,
    )
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, sigma=0, x0=x0, batch_size=2)
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    parent_sol = x0 * 5
    archive.add_single(parent_sol, 1, np.array([0, 0]))
    emitter = GaussianEmitter(archive, sigma=0, x0=x0, batch_size=2)

    # All solutions should be generated "around" the single parent solution in
    # the archive.
    solutions = emitter.ask()

    assert (solutions == np.expand_dims(parent_sol, axis=0)).all()
