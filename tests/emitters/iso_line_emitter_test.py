"""Tests for the IsoLineEmitter."""
import numpy as np

from ribs.emitters import IsoLineEmitter


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    iso_sigma = 1
    line_sigma = 2
    emitter = IsoLineEmitter(archive, x0, iso_sigma, line_sigma, batch_size=2)

    assert (emitter.x0 == x0).all()
    assert emitter.iso_sigma == iso_sigma
    assert emitter.line_sigma == line_sigma


def test_upper_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = IsoLineEmitter(archive, [2, 2],
                             iso_sigma=0,
                             line_sigma=0,
                             bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = IsoLineEmitter(archive, [-2, -2],
                             iso_sigma=0,
                             line_sigma=0,
                             bounds=[(-1, 1)] * 2)
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0, iso_sigma=0, batch_size=2)
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0, iso_sigma=0, batch_size=2)
    archive.add(x0, 1, np.array([0, 0]))

    solutions = emitter.ask()

    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_along_line(archive_fixture):
    archive, x0 = archive_fixture
    emitter = IsoLineEmitter(archive, x0, iso_sigma=0, batch_size=100)
    archive.add(np.array([0, 0, 0, 0]), 1, np.array([0, 0]))
    archive.add(np.array([10, 0, 0, 0]), 1, np.array([1, 1]))

    solutions = emitter.ask()

    # All solutions should either come from a degenerate distribution around
    # [0,0,0,0], a degenerate distribution around [10,0,0,0], or the "iso line
    # distribution" between [0,0,0,0] and [10,0,0,0] (i.e. a line between those
    # two points). By having a large batch size, we should be able to cover all
    # cases. In any case, this assertion should hold for all solutions
    # generated.
    assert (solutions[:, 1:] == 0).all()
