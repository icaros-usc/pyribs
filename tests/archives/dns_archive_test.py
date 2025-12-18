"Tests for the DNSArchive."

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
    rng = np.random.default_rng(42)

    archive.add(
        np.array([[0, 0], [1, 1]]),
        np.array([0, 1]),
        np.array([[0, 0], [1, 1]]),
    )

    # now there should be two solutions in the archive
    assert len(archive) == 2, "Archive should have two solutions"

    # now add 10 more solutions, cap should stay at 10
    archive.add(
        rng.random((10, 2)),
        rng.random(10),
        rng.random((10, 2)),
    )

    assert len(archive) == 10, "Archive should have 10 solutions"
    assert archive.capacity == 10, "Capacity should stay at 10"

    archive.add(
        rng.random((10, 2)),
        rng.random(10),
        rng.random((10, 2)),
    )

    assert len(archive) == 10, "Archive should have 10 solutions"
    assert archive.capacity == 10, "Capacity should stay at 10 when adding when full"


def test_remove_duplicates():
    """Test that DNS removes duplicates."""

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

    # even though they all have score 3, we should remove one of the two duplicates
    assert len(archive) == 3, "Archive should have 3 solutions"

    # ensure that we have [0, 0], [1, 1] and [2, 2] in the archive
    got = {tuple(map(float, row)) for row in archive.data("measures")}
    expected = {(0, 0), (1, 1), (2, 2)}
    assert got == expected, "Archive should have [0, 0], [1, 1] and [2, 2]"


def test_dns_scores_simple_case():
    """Test DNSArchive DNS computation using hard-coded known-correct values."""

    # Measures (1D)
    measures = np.array(
        [
            [0.0],
            [1.0],
            [3.0],
            [10.0],
        ],
        dtype=np.float64,
    )

    # Objectives
    objectives = np.array(
        [
            1.0,
            2.0,
            2.0,
            0.5,
        ],
        dtype=np.float64,
    )

    # Expected DNS values from manual computation
    dns_expected = np.array(
        [
            2.0,
            2.0,
            2.0,
            8.0,
        ],
        dtype=np.float64,
    )

    k = 2
    capacity = 10

    # Initialize archive
    arch = DNSArchive(
        solution_dim=1,
        measure_dim=1,
        capacity=capacity,
        k_neighbors=k,
        seed=0,
    )

    # Solutions can just be equal to measures for testing
    solutions = measures.copy()

    # Add entire batch
    add_info = arch.add(
        solution=solutions,
        objective=objectives,
        measures=measures,
    )

    dns_archive = add_info["dns"]

    # Compare to known-correct values
    assert np.allclose(dns_archive, dns_expected), (
        f"\nDNS mismatch!\nArchive:   {dns_archive}\nExpected: {dns_expected}"
    )


def test_dns_scores_2d_simple():
    measures = np.array(
        [
            [0.0, 0.0],  # idx 0
            [1.0, 0.0],  # idx 1
            [0.0, 2.0],  # idx 2
            [4.0, 0.0],  # idx 3
        ],
        dtype=np.float64,
    )

    objectives = np.array(
        [
            1.0,  # idx 0
            2.0,  # idx 1
            2.0,  # idx 2
            0.5,  # idx 3
        ],
        dtype=np.float64,
    )

    dns_expected = np.array(
        [
            1.5,
            np.sqrt(5),
            np.sqrt(5),
            3.5,
        ]
    )

    arch = DNSArchive(
        solution_dim=2,
        measure_dim=2,
        capacity=10,
        k_neighbors=2,
    )

    sols = measures.copy()

    info = arch.add(
        solution=sols,
        objective=objectives,
        measures=measures,
    )

    dns_archive = info["dns"]

    assert np.allclose(dns_archive, dns_expected), (
        f"\nGot:      {dns_archive}\nExpected: {dns_expected}"
    )
