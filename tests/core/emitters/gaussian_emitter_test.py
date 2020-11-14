"""Tests for the GaussianEmitter."""
import unittest

import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name


@pytest.fixture
def _archive_fixture():
    """Provides a simple archive and initial solution."""
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
    x0 = np.array([1, 2, 3, 4])
    archive.initialize(len(x0))
    return archive, x0


def test_ask_emits_correct_num_sols(_archive_fixture):
    archive, x0 = _archive_fixture
    emitter = GaussianEmitter(x0, 5, archive, config={"batch_size": 32})
    solutions = emitter.ask()
    assert solutions.shape == (32, 4)


def test_ask_emits_correct_num_sols_for_non_empty_archive(_archive_fixture):
    archive, x0 = _archive_fixture
    archive.add(x0, 10, np.array([0, 0]))
    emitter = GaussianEmitter(x0 * 2, 10, archive, config={"batch_size": 32})
    solutions = emitter.ask()
    assert solutions.shape == (32, 4)


def test_tell_inserts_into_archive(_archive_fixture):
    archive, x0 = _archive_fixture
    batch_size = 3
    emitter = GaussianEmitter(x0, 5, archive, config={"batch_size": batch_size})
    solutions = emitter.ask()
    objective_values = np.full(batch_size, 1.)
    behavior_values = np.array([[-1, -1], [0, 0], [1, 1]])
    emitter.tell(solutions, objective_values, behavior_values)

    # Check that the archive contains the behavior values inserted above.
    archive_data = archive.as_pandas()
    archive_beh = archive_data.loc[:, ["behavior-0", "behavior-1"]].to_numpy()
    unittest.TestCase().assertCountEqual(behavior_values.tolist(),
                                         archive_beh.tolist())
