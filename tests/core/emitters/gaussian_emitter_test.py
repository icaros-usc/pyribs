"""Tests for the GaussianEmitter."""
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name


def test_ask_emits_correct_num_sols():
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
    emitter = GaussianEmitter(
        [1, 2, 3, 4],
        5,
        archive,
        config={"batch_size": 32},
    )
    solutions = emitter.ask()
    assert solutions.shape == (32, 4)


def test_ask_emits_correct_num_sols_for_non_empty_archive():
    non_empty_archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
    non_empty_archive.add(np.array([1, 2, 3, 4]), 10, np.array([0, 0]))
    emitter = GaussianEmitter(
        [5, 6, 7, 8],
        10,
        non_empty_archive,
        config={"batch_size": 32},
    )
    solutions = emitter.ask()
    assert solutions.shape == (32, 4)


def test_tell_inserts_into_archive():
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
    batch_size = 3
    emitter = GaussianEmitter(
        [1, 2, 3, 4],
        5,
        archive,
        config={"batch_size": batch_size},
    )
    solutions = emitter.ask()
    objective_values = np.full(batch_size, 1.)
    behavior_values = np.array([[-1, -1], [0, 0], [1, 1]])
    emitter.tell(solutions, objective_values, behavior_values)

    archive_data = archive.as_pandas()
    archive_beh = archive_data.loc[:, ["behavior-0", "behavior-1"]].to_numpy()
    assert len(archive_beh) == 3

    # Convert to set so that we can check that the behavior values in the
    # archive are correct irrespective of order.
    behavior_value_set = set(map(tuple, behavior_values))
    archive_beh_set = set(map(tuple, archive_beh))
    assert behavior_value_set == archive_beh_set
