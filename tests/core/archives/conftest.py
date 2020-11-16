"""Useful utilities for all tests."""
import numpy as np
import pytest


@pytest.fixture
def benchmark_data_100k():
    """Provides a set of standardized benchmark data.

    Includes:
    - The number of values (100k)
    - 100k random solutions in the range (-1,1) in each dim
    - 100k random objective values drawn from the standard normal distribution
    - 100k random behavior values in the range (-1,1) in each dim
    """
    rng = np.random.default_rng(42)
    n_vals = int(1e5)
    solutions = rng.uniform(-1, 1, (n_vals, 10))
    objective_values = rng.standard_normal(n_vals)
    behavior_values = rng.uniform(-1, 1, (n_vals, 2))
    return n_vals, solutions, objective_values, behavior_values


@pytest.fixture(params=[False, True], ids=["brute_force", "kd_tree"])
def use_kd_tree(request):
    """Whether to use the KD Tree in CVTArchive."""
    return request.param
