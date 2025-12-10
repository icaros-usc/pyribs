"""Useful utilities for all archive tests."""

import numpy as np
import pytest
from box import Box

from ribs.archives import (
    CategoricalArchive,
    CVTArchive,
    GridArchive,
    ProximityArchive,
    SlidingBoundariesArchive,
)


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


@pytest.fixture(params=["scipy_kd_tree", "brute_force", "sklearn_nn"])
def nearest_neighbors(request):
    """Nearest neighbors method in CVTArchive."""
    return request.param


@pytest.fixture(params=["single", "batch"])
def add_mode(request):
    """Single or batch add."""
    return request.param


#
# Helpers for generating archive data.
#

ARCHIVE_NAMES = [
    "CategoricalArchive",
    "GridArchive",
    "CVTArchive-scipy_kd_tree",
    "CVTArchive-brute_force",
    "CVTArchive-sklearn_nn",
    "SlidingBoundariesArchive",
    "ProximityArchive",
]


def get_archive_data(name, *, dtype=np.float64):
    """Returns data to use for testing each archive.

    The archives vary, but there will always be an empty 2D archive, as well as
    a 2D archive with a single solution added to it. This solution will have a
    value of [1, 2, 3], its objective value will be 1.0, and its measures will
    be [0.25, 0.25].

    The name is the name of an archive to create. It should come from
    ARCHIVE_NAMES.
    """
    # All local variables are captured at the end.

    # Characteristics of a single solution to insert into archive_with_elite.
    solution = np.array([1.0, 2.0, 3.0])
    objective = 1.0
    if name == "CategoricalArchive":
        measures = np.array(["B", "Two"], dtype=object)
    else:
        measures = np.array([0.25, 0.25])

    if name == "GridArchive":
        # Grid archive with 10 cells and range (-1, 1) in first dim, and 20
        # cells and range (-2, 2) in second dim.
        cells = 10 * 20
        archive = GridArchive(
            solution_dim=len(solution),
            dims=[10, 20],
            ranges=[(-1, 1), (-2, 2)],
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )

        archive_with_elite = GridArchive(
            solution_dim=len(solution),
            dims=[10, 20],
            ranges=[(-1, 1), (-2, 2)],
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )
        grid_indices = (6, 11)
        int_index = 131
    elif name.startswith("CVTArchive-"):
        # CVT archive with bounds (-1,1) and (-1,1), and 4 centroids at (0.5,
        # 0.5), (-0.5, 0.5), (-0.5, -0.5), and (0.5, -0.5). The elite in
        # archive_with_elite should match with centroid (0.5, 0.5).
        cells = 4
        nearest_neighbors = name.removeprefix("CVTArchive-")
        centroids = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
        centroid = [0.5, 0.5]

        archive = CVTArchive(
            solution_dim=len(solution),
            centroids=centroids,
            ranges=[(-1, 1), (-1, 1)],
            nearest_neighbors=nearest_neighbors,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )

        archive_with_elite = CVTArchive(
            solution_dim=len(solution),
            centroids=centroids,
            ranges=[(-1, 1), (-1, 1)],
            nearest_neighbors=nearest_neighbors,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )
    elif name == "SlidingBoundariesArchive":
        # Sliding boundary archive with 10 cells and range (-1, 1) in first dim,
        # and 20 cells and range (-2, 2) in second dim.
        cells = 10 * 20
        archive = SlidingBoundariesArchive(
            solution_dim=len(solution),
            dims=[10, 20],
            ranges=[(-1, 1), (-2, 2)],
            remap_frequency=100,
            buffer_capacity=1000,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )

        archive_with_elite = SlidingBoundariesArchive(
            solution_dim=len(solution),
            dims=[10, 20],
            ranges=[(-1, 1), (-2, 2)],
            remap_frequency=100,
            buffer_capacity=1000,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )
        grid_indices = (6, 11)
        int_index = 131
    elif name == "ProximityArchive":
        cells = 0
        capacity = 1
        k_neighbors = 5
        novelty_threshold = 1.0
        archive = ProximityArchive(
            solution_dim=len(solution),
            measure_dim=2,
            k_neighbors=k_neighbors,
            novelty_threshold=novelty_threshold,
            initial_capacity=capacity,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )

        archive_with_elite = ProximityArchive(
            solution_dim=len(solution),
            measure_dim=2,
            k_neighbors=k_neighbors,
            novelty_threshold=novelty_threshold,
            initial_capacity=capacity,
            solution_dtype=dtype,
            objective_dtype=dtype,
            measures_dtype=dtype,
        )
    elif name == "CategoricalArchive":
        cells = 3 * 4
        grid_indices = (1, 1)
        archive = CategoricalArchive(
            solution_dim=len(solution),
            categories=[
                ["A", "B", "C"],
                ["One", "Two", "Three", "Four"],
            ],
            solution_dtype=dtype,
            objective_dtype=dtype,
        )
        archive_with_elite = CategoricalArchive(
            solution_dim=len(solution),
            categories=[
                ["A", "B", "C"],
                ["One", "Two", "Three", "Four"],
            ],
            solution_dtype=dtype,
            objective_dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown name {name}")

    archive_with_elite.add_single(solution, objective, measures)

    # Captures all the local variables and provides them as data. Box works with
    # dot notation, e.g., data.archive == data["archive"]
    return Box(locals())
