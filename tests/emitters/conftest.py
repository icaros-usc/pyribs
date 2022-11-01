"""Useful utilities for all emitter tests."""
import numpy as np
import pytest

from ribs.archives import GridArchive


@pytest.fixture
def archive_fixture():
    """Provides a simple archive and initial solution."""
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(solution_dim=len(x0),
                          dims=[10, 10],
                          ranges=[(-1, 1), (-1, 1)])
    return archive, x0
