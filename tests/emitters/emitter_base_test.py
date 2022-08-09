"""Tests that should work for all emitters."""
import numpy as np
import pytest

from ribs.emitters import (EvolutionStrategyEmitter, GaussianEmitter,
                           IsoLineEmitter)

# pylint: disable = redefined-outer-name


@pytest.fixture(params=[
    "GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter",
    "RandomDirectionEmitter", "OptimizingEmitter"
])
def emitter_fixture(request, archive_fixture):
    """Creates an archive, emitter, and initial solution.

    Returns:
        Tuple of (archive, emitter, batch_size, x0).
    """
    emitter_type = request.param
    archive, x0 = archive_fixture
    batch_size = 3

    if emitter_type == "GaussianEmitter":
        emitter = GaussianEmitter(archive, x0, 5, batch_size=batch_size)
    elif emitter_type == "IsoLineEmitter":
        emitter = IsoLineEmitter(archive, x0, batch_size=batch_size)
    elif emitter_type == "ImprovementEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0,
                                           5,
                                           "2imp",
                                           batch_size=batch_size)
    elif emitter_type == "RandomDirectionEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0,
                                           5,
                                           "2rd",
                                           batch_size=batch_size)
    elif emitter_type == "OptimizingEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0,
                                           5,
                                           "2obj",
                                           batch_size=batch_size)
    else:
        raise NotImplementedError(f"Unknown emitter type {emitter_type}")

    return archive, emitter, batch_size, x0


#
# ask()
#


def test_ask_emits_correct_num_sols(emitter_fixture):
    _, emitter, batch_size, x0 = emitter_fixture
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_ask_emits_correct_num_sols_on_nonempty_archive(emitter_fixture):
    archive, emitter, batch_size, x0 = emitter_fixture
    archive.add_single(x0, 1, np.array([0, 0]))
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


#
# Bounds handling (only uses GaussianEmitter).
#


def test_array_bound_correct(archive_fixture):
    archive, x0 = archive_fixture
    bounds = []
    for i in range(len(x0) - 1):
        bounds.append((-i, i))
    bounds.append(None)
    emitter = GaussianEmitter(archive, x0, 1, bounds=bounds)

    lower_bounds = np.concatenate((-np.arange(len(x0) - 1), [-np.inf]))
    upper_bounds = np.concatenate((np.arange(len(x0) - 1), [np.inf]))

    assert (emitter.lower_bounds == lower_bounds).all()
    assert (emitter.upper_bounds == upper_bounds).all()


def test_long_array_bound_fails(archive_fixture):
    archive, x0 = archive_fixture
    bounds = [(-1, 1)] * (len(x0) + 1)  # More bounds than solution dims.
    with pytest.raises(ValueError):
        GaussianEmitter(archive, x0, 1, bounds=bounds)


def test_array_bound_bad_entry_fails(archive_fixture):
    archive, x0 = archive_fixture
    bounds = [(-1, 1)] * len(x0)
    bounds[0] = (-1, 0, 1)  # Invalid entry.
    with pytest.raises(ValueError):
        GaussianEmitter(archive, x0, 1, bounds=bounds)


### x0 should be 1-dimensional ###


@pytest.mark.parametrize(
    "emitter_type", ["GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter"])
def test_x0_not_1d(emitter_type, archive_fixture):
    archive, _ = archive_fixture
    x0 = [[1], [1]]

    with pytest.raises(ValueError):
        if emitter_type == "GaussianEmitter":
            _ = GaussianEmitter(archive, x0, 5)
        elif emitter_type == "IsoLineEmitter":
            _ = IsoLineEmitter(archive, x0)
        elif emitter_type == "ImprovementEmitter":
            _ = EvolutionStrategyEmitter(archive, x0, 5, "2imp")
