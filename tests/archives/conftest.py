"""Useful utilities for all tests."""
import numpy as np
import pytest

from ribs.archives import GridArchive


@pytest.fixture
def benchmark_data_100k():
    """Provides a set of standardized benchmark data.

    Includes:
    - The number of values (100k)
    - 100k random solutions in the range (-1,1) in each dim
    - 100k random objective values drawn from the standard normal distribution
    - 100k random behavior values in the range (-1,1) in each dim
    """
    n = int(1e5)
    solutions = np.random.uniform(-1, 1, (n, 10))
    objective_values = np.random.randn(n)
    behavior_values = np.random.uniform(-1, 1, (n, 2))
    return n, solutions, objective_values, behavior_values
