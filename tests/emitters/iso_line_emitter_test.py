"""Tests for the IsoLineEmitter."""
import numpy as np
import pytest

from ribs.emitters import IsoLineEmitter


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    iso_sigma = 1
    line_sigma = 2
    batch_size = 2
    emitter = IsoLineEmitter(archive,
                             iso_sigma=iso_sigma,
                             line_sigma=line_sigma,
                             x0=x0,
                             batch_size=batch_size)

    assert np.all(emitter.x0 == x0)
    assert emitter.iso_sigma == iso_sigma
    assert emitter.line_sigma == line_sigma
    assert emitter.batch_size == batch_size


def test_initial_solutions_is_correct(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    emitter = IsoLineEmitter(archive, initial_solutions=initial_solutions)

    assert np.all(emitter.ask() == initial_solutions)
    assert np.all(emitter.initial_solutions == initial_solutions)


def test_initial_solutions_shape(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 0, 0], [1, 1, 1]]

    # archive.solution_dim = 4
    with pytest.raises(ValueError):
        IsoLineEmitter(archive, initial_solutions=initial_solutions)


def test_neither_x0_nor_initial_solutions_provided(archive_fixture):
    archive, _ = archive_fixture
    with pytest.raises(ValueError):
        IsoLineEmitter(archive)


def test_both_x0_and_initial_solutions_provided(archive_fixture):
    archive, x0 = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    with pytest.raises(ValueError):
        IsoLineEmitter(archive, x0=x0, initial_solutions=initial_solutions)


def test_upper_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = IsoLineEmitter(
        archive,
        x0=[2, 2, 2, 2],
        iso_sigma=0,
        line_sigma=0,
        bounds=[(-1, 1)] * 4,
    )
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = IsoLineEmitter(
        archive,
        x0=[-2, -2, -2, -2],
        iso_sigma=0,
        line_sigma=0,
        bounds=[(-1, 1)] * 4,
    )
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0=x0, iso_sigma=0, batch_size=2)
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0=x0, iso_sigma=0, batch_size=2)
    archive.add_single(x0, 1, np.array([0, 0]))

    solutions = emitter.ask()

    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_along_line(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0=x0, iso_sigma=0, batch_size=100)
    archive.add_single(np.array([0, 0, 0, 0]), 1, np.array([0, 0]))
    archive.add_single(np.array([10, 0, 0, 0]), 1, np.array([1, 1]))

    solutions = emitter.ask()

    # All solutions should either come from a degenerate distribution around
    # [0,0,0,0], a degenerate distribution around [10,0,0,0], or the "iso line
    # distribution" between [0,0,0,0] and [10,0,0,0] (i.e. a line between those
    # two points). By having a large batch size, we should be able to cover all
    # cases. In any case, this assertion should hold for all solutions
    # generated.
    assert (solutions[:, 1:] == 0).all()
