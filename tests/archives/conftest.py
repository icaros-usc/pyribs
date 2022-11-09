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
    - 10k random measures in the range (-1,1) in each dim
    """
    rng = np.random.default_rng(42)
    n_vals = 10_000
    solution_batch = rng.uniform(-1, 1, (n_vals, 10))
    objective_batch = rng.standard_normal(n_vals)
    measures_batch = rng.uniform(-1, 1, (n_vals, 2))
    return n_vals, solution_batch, objective_batch, measures_batch


@pytest.fixture(params=[False, True], ids=["brute_force", "kd_tree"])
def use_kd_tree(request):
    """Whether to use the KD Tree in CVTArchive."""
    return request.param


@pytest.fixture(params=["single", "batch"])
def add_mode(request):
    """Single or batch add."""
    return request.param


#
# Helpers for generating archive data.
#

ArchiveFixtureData = namedtuple(
    "ArchiveFixtureData",
    [
        "archive",  # An empty archive with 2D measure space.
        "archive_with_elite",  # 2D measure space with one elite.
        "solution",  # A solution.
        "objective",  # Float objective value.
        "measures",  # 2D measures for the solution.
        "metadata",  # Metadata object for the solution.
        "grid_indices",  # Index for GridArchive and SlidingBoundariesArchive.
        "int_index",  # Integer index corresponding to grid_indices.
        "centroid",  # Centroid coordinates for CVTArchive.
        "cells",  # Total number of cells in the archive.
    ],
)

ARCHIVE_NAMES = [
    "GridArchive",
    "CVTArchive-brute_force",
    "CVTArchive-kd_tree",
    "SlidingBoundariesArchive",
]


def get_archive_data(name, dtype=np.float64):
    """Returns ArchiveFixtureData to use for testing each archive.

    The archives vary, but there will always be an empty 2D archive, as well as
    a 2D archive with a single solution added to it. This solution will have a
    value of [1, 2, 3], its objective value will be 1.0, and its measures will
    be [0.25, 0.25].

    The name is the name of an archive to create. It should come from
    ARCHIVE_NAMES.
    """
    # Characteristics of a single solution to insert into archive_with_elite.
    solution = np.array([1., 2., 3.])
    objective = 1.0
    measures = np.array([0.25, 0.25])
    metadata = {"metadata_key": 42}
    grid_indices = None
    int_index = None
    centroid = None

    if name == "GridArchive":
        # Grid archive with 10 cells and range (-1, 1) in first dim, and 20
        # cells and range (-2, 2) in second dim.
        cells = 10 * 20
        archive = GridArchive(solution_dim=len(solution),
                              dims=[10, 20],
                              ranges=[(-1, 1), (-2, 2)],
                              dtype=dtype)

        archive_with_elite = GridArchive(solution_dim=len(solution),
                                         dims=[10, 20],
                                         ranges=[(-1, 1), (-2, 2)],
                                         dtype=dtype)
        grid_indices = (6, 11)
        int_index = 131
    elif name.startswith("CVTArchive-"):
        # CVT archive with bounds (-1,1) and (-1,1), and 4 centroids at (0.5,
        # 0.5), (-0.5, 0.5), (-0.5, -0.5), and (0.5, -0.5). The elite in
        # archive_with_elite should match with centroid (0.5, 0.5).
        cells = 4
        kd_tree = name == "CVTArchive-kd_tree"
        samples = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
        centroid = [0.5, 0.5]

        archive = CVTArchive(solution_dim=len(solution),
                             cells=4,
                             ranges=[(-1, 1), (-1, 1)],
                             samples=samples,
                             use_kd_tree=kd_tree,
                             dtype=dtype)

        archive_with_elite = CVTArchive(solution_dim=len(solution),
                                        cells=4,
                                        ranges=[(-1, 1), (-1, 1)],
                                        samples=samples,
                                        use_kd_tree=kd_tree,
                                        dtype=dtype)
    elif name == "SlidingBoundariesArchive":
        # Sliding boundary archive with 10 cells and range (-1, 1) in first dim,
        # and 20 cells and range (-2, 2) in second dim.
        cells = 10 * 20
        archive = SlidingBoundariesArchive(solution_dim=len(solution),
                                           dims=[10, 20],
                                           ranges=[(-1, 1), (-2, 2)],
                                           remap_frequency=100,
                                           buffer_capacity=1000,
                                           dtype=dtype)

        archive_with_elite = SlidingBoundariesArchive(
            solution_dim=len(solution),
            dims=[10, 20],
            ranges=[(-1, 1), (-2, 2)],
            remap_frequency=100,
            buffer_capacity=1000,
            dtype=dtype)
        grid_indices = (6, 11)
        int_index = 131

    archive_with_elite.add_single(solution, objective, measures, metadata)

    return ArchiveFixtureData(
        archive,
        archive_with_elite,
        solution,
        objective,
        measures,
        metadata,
        grid_indices,
        int_index,
        centroid,
        cells,
    )
