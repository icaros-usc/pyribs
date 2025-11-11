"Tests for the DNSArchive."

import pytest
import numpy as np

from ribs.archives import DNSArchive


def test_add_from_empty():
    archive = DNSArchive(
        solution_dim=2,
        measure_dim=2,
        capacity=10,
        k_neighbors=5,
    )
    assert archive.empty, "Archive should be empty upon creation"

    archive.add(
        np.array([[0, 0], [1, 1]]),
        np.array([0, 1]),
        np.array([[0, 0], [1, 1]]),
    )

    #now there should be two solutions in the archive
    assert len(archive) == 2, "Archive should have two solutions"


    # now add 10 more solutions, cap should stay at 10
    archive.add(
        np.random.rand(10, 2),
        np.random.rand(10),
        np.random.rand(10, 2),
    )

    assert len(archive) == 10, "Archive should have 10 solutions"
    assert archive.capacity == 10, "Capacity should stay at 10"

    archive.add(
        np.random.rand(10, 2),
        np.random.rand(10),
        np.random.rand(10, 2),
    )

    assert len(archive) == 10, "Archive should have 10 solutions"
    assert archive.capacity == 10, "Capacity should stay at 10 when adding when full"


#ensure that DNS removes duplicates
def test_remove_duplicates():
    archive = DNSArchive(
        solution_dim=2,
        measure_dim=2,
        capacity=3,
        k_neighbors=5,
    )
    
    archive.add(
        np.array([[0, 0], [1, 1], [0, 0], [2, 2]]),
        np.array([1, 1, 1, 1]),
        np.array([[0, 0], [1, 1], [0, 0], [2, 2]]),
    )

    #even though they all have score 3, we should remove one of the two duplicates
    assert len(archive) == 3, "Archive should have 3 solutions"

    #ensure that we have [0, 0], [1, 1] and [2, 2] in the archive
    got = {tuple(map(float, row)) for row in archive.data("measures")}
    expected = {(0, 0), (1, 1), (2, 2)}
    assert got == expected, "Archive should have [0, 0], [1, 1] and [2, 2]"

