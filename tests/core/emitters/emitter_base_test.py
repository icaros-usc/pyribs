"""Tests that should work for all emitters."""
import unittest

import numpy as np
import pytest

from ribs.emitters import GaussianEmitter, IsoLineEmitter

# pylint: disable = invalid-name


@pytest.fixture(params=["GaussianEmitter", "IsoLineEmitter"])
def _emitter_fixture(request, _archive_fixture):
    """Creates an archive, emitter, and initial solution.

    Returns:
        Tuple of (archive, emitter, batch_size, x0).
    """
    emitter_type = request.param
    archive, x0 = _archive_fixture
    batch_size = 3

    if emitter_type == "GaussianEmitter":
        emitter = GaussianEmitter(x0,
                                  5,
                                  archive,
                                  config={"batch_size": batch_size})
    elif emitter_type == "IsoLineEmitter":
        emitter = IsoLineEmitter(x0, archive, config={"batch_size": batch_size})
    else:
        raise NotImplementedError(f"Unknown emitter type {emitter_type}")

    return archive, emitter, batch_size, x0


def test_ask_emits_correct_num_sols(_emitter_fixture):
    _, emitter, batch_size, x0 = _emitter_fixture
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_ask_emits_correct_num_sols_for_non_empty_archive(_emitter_fixture):
    archive, emitter, batch_size, x0 = _emitter_fixture
    archive.add(x0, 1, np.array([0, 0]))
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_tell_inserts_into_archive(_emitter_fixture):
    archive, emitter, batch_size, _ = _emitter_fixture
    solutions = emitter.ask()
    objective_values = np.full(batch_size, 1.)
    behavior_values = np.array([[-1, -1], [0, 0], [1, 1]])
    emitter.tell(solutions, objective_values, behavior_values)

    # Check that the archive contains the behavior values inserted above.
    archive_data = archive.as_pandas()
    archive_beh = archive_data.loc[:, ["behavior-0", "behavior-1"]].to_numpy()
    unittest.TestCase().assertCountEqual(behavior_values.tolist(),
                                         archive_beh.tolist())
