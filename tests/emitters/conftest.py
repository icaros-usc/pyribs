"""Useful utilities for all emitter tests."""
import numpy as np
import pytest

from ribs.archives import GridArchive


@pytest.fixture
def archive_fixture():
    """Provides a simple archive and initial solution."""
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10], [(-1, 1), (-1, 1)])
    return archive, x0
