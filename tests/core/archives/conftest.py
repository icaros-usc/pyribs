"""Useful utilities for all archive tests."""
from collections import namedtuple

import numpy as np
import pytest

from ribs.archives import CVTArchive, GridArchive, SlidingBoundariesArchive


@pytest.fixture
def benchmark_data_10k():
    """Provides a set of standardized benchmark data.

    Includes:
    - The number of values (10k)
    - 10k random solutions in the range (-1,1) in each dim
    - 10k random objective values drawn from the standard normal distribution
    - 10k random behavior values in the range (-1,1) in each dim
    """
    rng = np.random.default_rng(42)
    n_vals = 10_000
    solutions = rng.uniform(-1, 1, (n_vals, 10))
    objective_values = rng.standard_normal(n_vals)
    behavior_values = rng.uniform(-1, 1, (n_vals, 2))
    return n_vals, solutions, objective_values, behavior_values


@pytest.fixture(params=[False, True], ids=["brute_force", "kd_tree"])
def use_kd_tree(request):
    """Whether to use the KD Tree in CVTArchive."""
    return request.param


#
# Helpers for generating archive data.
#

ArchiveFixtureData = namedtuple(
    "ArchiveFixtureData",
    [
        "archive",  # An empty archive with 2D behavior space.
        "archive_with_entry",  # 2D behavior space with one entry.
        "solution",  # A solution.
        "objective_value",  # Float objective value.
        "behavior_values",  # 2D behavior values for the solution.
        "grid_indices",  # Intended indices for GridArchive.
        "centroid",  # Intended centroid coordinates for CVTArchive.
        "bins",  # Total number of bins in the archive.
    ],
)

ARCHIVE_NAMES = [
    "GridArchive",
    "CVTArchive-brute_force",
    "CVTArchive-kd_tree",
]


def get_archive_data(name, dtype=np.float64):
    """Returns ArchiveFixtureData to use for testing each archive.

    The archives vary, but there will always be an empty 2D archive, as well as
    a 2D archive with a single solution added to it. This solution will have a
    value of [1, 2, 3], its objective value will be 1.0, and its behavior values
    will be [0.25, 0.25].

    The name is the name of an archive to create. It should come from
    ARCHIVE_NAMES.
    """
    # Characteristics of a single solution to insert into archive_with_entry.
    solution = np.array([1, 2, 3])
    objective_value = 1.0
    behavior_values = np.array([0.25, 0.25])
    grid_indices = None
    centroid = None

    if name == "GridArchive":
        # Grid archive with 10 bins and range (-1, 1) in first dim, and 20 bins
        # and range (-2, 2) in second dim.
        bins = 10 * 20
        archive = GridArchive([10, 20], [(-1, 1), (-2, 2)], dtype=dtype)
        archive.initialize(len(solution))

        archive_with_entry = GridArchive([10, 20], [(-1, 1), (-2, 2)],
                                         dtype=dtype)
        archive_with_entry.initialize(len(solution))
        grid_indices = (6, 11)
    elif name.startswith("CVTArchive-"):
        # CVT archive with bounds (-1,1) and (-1,1), and 4 centroids at (0.5,
        # 0.5), (-0.5, 0.5), (-0.5, -0.5), and (0.5, -0.5). The entry in
        # archive_with_entry should match with centroid (0.5, 0.5).
        bins = 4
        kd_tree = name == "CVTArchive-kd_tree"
        samples = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
        centroid = [0.5, 0.5]

        archive = CVTArchive(4, [(-1, 1), (-1, 1)],
                             samples=samples,
                             use_kd_tree=kd_tree,
                             dtype=dtype)
        archive.initialize(len(solution))

        archive_with_entry = CVTArchive(4, [(-1, 1), (-1, 1)],
                                        samples=samples,
                                        use_kd_tree=kd_tree,
                                        dtype=dtype)
        archive_with_entry.initialize(len(solution))
    elif name == "SlidingBoundariesArchive":
        # Sliding boundary archive with 10 bins and range (-1, 1) in first dim,
        # and 20 bins and range (-2, 2) in second dim.
        bins = 10 * 20
        archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-2, 2)],
                                           remap_frequency=100,
                                           buffer_capacity=1000,
                                           dtype=dtype)
        archive.initialize(len(solution))

        archive_with_entry = SlidingBoundariesArchive([10, 20], [(-1, 1),
                                                                 (-2, 2)],
                                                      remap_frequency=100,
                                                      buffer_capacity=1000,
                                                      dtype=dtype)
        archive_with_entry.initialize(len(solution))
        grid_indices = (6, 11)

    archive_with_entry.add(solution, objective_value, behavior_values)
    return ArchiveFixtureData(
        archive,
        archive_with_entry,
        solution,
        objective_value,
        behavior_values,
        grid_indices,
        centroid,
        bins,
    )
