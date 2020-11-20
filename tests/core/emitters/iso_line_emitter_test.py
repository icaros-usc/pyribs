"""Tests for the IsoLineEmitter."""
import numpy as np

from ribs.emitters import IsoLineEmitter

# pylint: disable = invalid-name


def test_degenerate_gauss_emits_x0(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = IsoLineEmitter(x0, archive, iso_sigma=0, config={"batch_size": 2})
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = IsoLineEmitter(x0,
                             archive,
                             iso_sigma=0,
                             config={"batch_size": 100})
    archive.add(x0, 1, np.array([0, 0]))

    solutions = emitter.ask()

    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_along_line(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = IsoLineEmitter(x0,
                             archive,
                             iso_sigma=0,
                             config={"batch_size": 100})
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
