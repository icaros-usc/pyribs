"""Tests for ArchiveBase."""
import numpy as np
import pytest

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
# ArchiveBase tests -- should work for all archive classes.
#

# pylint: disable = redefined-outer-name


@pytest.fixture(params=ARCHIVE_NAMES)
def archive_data(request):
    """Provides data for testing all kinds of archives."""
    return get_archive_data(request.param)


def test_archive_is_2d(archive_data):
    assert archive_data.archive.is_2d()


def test_new_archive_is_empty(archive_data):
    assert archive_data.archive.is_empty()


def test_archive_with_entry_is_not_empty(archive_data):
    assert not archive_data.archive_with_entry.is_empty()


def test_random_elite_gets_single_elite(archive_data):
    retrieved = archive_data.archive_with_entry.get_random_elite()
    print(retrieved)
    print(archive_data.solution)
    assert np.all(retrieved[0] == archive_data.solution)
    assert retrieved[1] == archive_data.objective_value
    assert np.all(retrieved[2] == archive_data.behavior_values)


def test_random_elite_fails_when_empty(archive_data):
    with pytest.raises(IndexError):
        archive_data.archive.get_random_elite()
