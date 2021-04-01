"""Tests that should work for all emitters."""
import unittest

import numpy as np
import pytest

from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)


@pytest.fixture(params=[
    "GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter",
    "RandomDirectionEmitter", "OptimizingEmitter"
])
def _emitter_fixture(request, archive_fixture):
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
        emitter = ImprovementEmitter(archive, x0, 5, batch_size=batch_size)
    elif emitter_type == "RandomDirectionEmitter":
        emitter = RandomDirectionEmitter(archive, x0, 5, batch_size=batch_size)
    elif emitter_type == "OptimizingEmitter":
        emitter = OptimizingEmitter(archive, x0, 5, batch_size=batch_size)
    else:
        raise NotImplementedError(f"Unknown emitter type {emitter_type}")

    return archive, emitter, batch_size, x0


#
# ask()
#


def test_ask_emits_correct_num_sols(_emitter_fixture):
    _, emitter, batch_size, x0 = _emitter_fixture
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_ask_emits_correct_num_sols_on_nonempty_archive(_emitter_fixture):
    archive, emitter, batch_size, x0 = _emitter_fixture
    archive.add(x0, 1, np.array([0, 0]))
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


#
# tell()
#


def test_tell_inserts_into_archive(_emitter_fixture):
    archive, emitter, batch_size, _ = _emitter_fixture
    solutions = emitter.ask()
    objective_values = np.full(batch_size, 1.)
    behavior_values = np.array([[-1, -1], [0, 0], [1, 1]])
    metadata = np.full(batch_size, {"metadata_key": 42})
    emitter.tell(solutions, objective_values, behavior_values, metadata)

    # Check all values are inserted. Only behavior values, objectives, and
    # metadata are known; solutions may vary.
    archive_data = archive.as_pandas(include_metadata=True)
    archive_beh = archive_data.loc[:, ["behavior_0", "behavior_1"]].to_numpy()
    unittest.TestCase().assertCountEqual(behavior_values.tolist(),
                                         archive_beh.tolist())
    assert (archive_data["objective"] == objective_values).all()
    assert (archive_data["metadata"] == metadata).all()


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
