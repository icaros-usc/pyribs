"""Tests for ArchiveBase."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.archives._archive_base import RandomBuffer

from .conftest import ARCHIVE_NAMES, get_archive_data

# pylint: disable = invalid-name

#
# RandomBuffer tests -- note this is an internal class.
#


def test_random_buffer_in_range():
    buffer = RandomBuffer(buf_size=10)
    for _ in range(10):
        assert buffer.get(1) == 0
    for _ in range(10):
        assert buffer.get(2) in [0, 1]


def test_random_buffer_not_repeating():
    buf_size = 100
    buffer = RandomBuffer(buf_size=buf_size)

    # Both lists contain random integers in the range [0,100), and we triggered
    # a reset by retrieving more than buf_size integers. It should be nearly
    # impossible for the lists to be equal unless something is wrong.
    x1 = [buffer.get(100) for _ in range(buf_size)]
    x2 = [buffer.get(100) for _ in range(buf_size)]

    assert x1 != x2


#
# Tests for the require_init decorator. Just need to make sure it works on one
# method, as it is too much to test on all.
#


def test_method_fails_without_init():
    archive = GridArchive([20, 20], [(-1, 1)] * 2)
    with pytest.raises(RuntimeError):
        archive.add(np.array([1, 2, 3]), 1.0, np.array([1.0, 1.0]))


#
# ArchiveBase tests -- should work for all archive classes.
#

# pylint: disable = redefined-outer-name


@pytest.fixture(params=ARCHIVE_NAMES)
def _archive_data(request):
    """Provides data for testing all kinds of archives."""
    return get_archive_data(request.param)


def test_archive_cannot_reinit(_archive_data):
    with pytest.raises(RuntimeError):
        _archive_data.archive.initialize(len(_archive_data.solution))


def test_archive_is_2d(_archive_data):
    assert _archive_data.archive.is_2d


def test_new_archive_is_empty(_archive_data):
    assert _archive_data.archive.empty


def test_archive_with_entry_is_not_empty(_archive_data):
    assert not _archive_data.archive_with_entry.empty


def test_elite_with_behavior_gets_correct_elite(_archive_data):
    retrieved = _archive_data.archive_with_entry.elite_with_behavior(
        _archive_data.behavior_values)
    assert (retrieved[0] == _archive_data.solution).all()
    assert retrieved[1] == _archive_data.objective_value
    assert (retrieved[2] == _archive_data.behavior_values).all()


def test_elite_with_behavior_returns_none(_archive_data):
    retrieved = _archive_data.archive.elite_with_behavior(
        _archive_data.behavior_values)
    assert (retrieved[0] is None and retrieved[1] is None and
            retrieved[2] is None)


def test_random_elite_gets_single_elite(_archive_data):
    retrieved = _archive_data.archive_with_entry.get_random_elite()
    assert np.all(retrieved[0] == _archive_data.solution)
    assert retrieved[1] == _archive_data.objective_value
    assert np.all(retrieved[2] == _archive_data.behavior_values)


def test_random_elite_fails_when_empty(_archive_data):
    with pytest.raises(IndexError):
        _archive_data.archive.get_random_elite()
