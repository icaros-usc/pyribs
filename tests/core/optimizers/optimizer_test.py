"""Tests for the Optimizer."""
import unittest

import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


@pytest.fixture
def _optimizer_fixture():
    """Returns an Optimizer with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=num_solutions)
    ]
    return Optimizer(archive, emitters), solution_dim, num_solutions


def test_init_fails_with_no_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1)] * 5

    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        # Emits 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0], 1),
        # Mismatch -- emits 3D solutions rather than 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0, 0.0], 1),
    ]
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_ask_returns_correct_solution_shape(_optimizer_fixture):
    optimizer, solution_dim, num_solutions = _optimizer_fixture
    solutions = optimizer.ask()
    assert solutions.shape == (num_solutions, solution_dim)


def test_ask_fails_when_called_twice(_optimizer_fixture):
    optimizer, *_ = _optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.ask()
        optimizer.ask()


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_into_archive(_optimizer_fixture, tell_metadata):
    optimizer, _, num_solutions = _optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = [f"metadata_{i}" for i in range(4)] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * num_solutions

    # We pass in 4 solutions with unique behavior values, so all should go into
    # the archive.
    optimizer.tell(
        objective_values=np.ones(num_solutions),
        behavior_values=behavior_values,
        metadata=metadata,
    )

    archive_data = optimizer.archive.as_pandas(include_metadata=True)
    archive_beh = (
        archive_data.loc[:, ["behavior_0", "behavior_1"]].to_numpy().tolist())
    archive_meta = archive_data.loc[:,
                                    "metadata"].to_numpy(dtype=object).tolist()

    # TODO: Use the raw data API described in #118 when it becomes available.
    assert len(archive_data) == num_solutions
    unittest.TestCase().assertCountEqual(behavior_values, archive_beh)
    unittest.TestCase().assertCountEqual(expected_metadata, archive_meta)


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_with_multiple_emitters(tell_metadata):
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1),
        GaussianEmitter(archive, [0.5, 0.5], 1, batch_size=2),
        GaussianEmitter(archive, [-0.5, -0.5], 1, batch_size=3),
    ]
    optimizer = Optimizer(archive, emitters)

    _ = optimizer.ask()
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                       [0.0, 0.0], [0.0, 1.0]]
    metadata = [f"metadata_{i}" for i in range(6)] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * 6

    # The sum of all the emitters' batch sizes is 6.
    optimizer.tell(
        objective_values=np.ones(6),
        behavior_values=behavior_values,
        metadata=metadata,
    )
    assert len(optimizer.archive.as_pandas()) == 6

    archive_data = optimizer.archive.as_pandas(include_metadata=True)
    archive_beh = (
        archive_data.loc[:, ["behavior_0", "behavior_1"]].to_numpy().tolist())
    archive_meta = archive_data.loc[:,
                                    "metadata"].to_numpy(dtype=object).tolist()

    # TODO: Use the raw data API described in #118 when it becomes available.
    assert len(archive_data) == 6
    unittest.TestCase().assertCountEqual(behavior_values, archive_beh)
    unittest.TestCase().assertCountEqual(expected_metadata, archive_meta)


def test_tell_fails_when_ask_not_called(_optimizer_fixture):
    optimizer, *_ = _optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)
