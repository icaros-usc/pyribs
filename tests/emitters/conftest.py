"""Useful utilities for all emitter tests."""
import numpy as np
import pandas as pd
import pytest
from numba import jit

from ribs.archives import ArchiveBase, GridArchive


@pytest.fixture
def archive_fixture():
    """Provides a simple archive and initial solution."""
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
    x0 = np.array([1, 2, 3, 4])
    archive.initialize(len(x0))
    return archive, x0


class FakeArchive(ArchiveBase):
    """Bare-bones archive solely for emitter benchmarking.

    Because this archive is used in emitter benchmarking, we want to
    spend as little time in this archive as possible. Thus, the archive
    functions are optimized for speed and not for any meaningful
    functionality. That's why this is a "fake" archive.

    Note that the get_index() method may not ever actually be called.
    """

    def __init__(self, dims):
        self._dims = np.array(dims)
        behavior_dim = len(self._dims)
        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=behavior_dim,
        )

    def get_random_elite(self):
        index = (0,) * self._behavior_dim
        return (
            self._solutions[index],
            self._objective_values[index],
            self._behavior_values[index],
        )

    def add(self, solution, objective_value, behavior_values, metadata=None):
        return True

    @jit(nopython=True)
    def get_index(self, behavior_values):
        return np.full_like(behavior_values, 0)

    def as_pandas(self):
        return pd.Dataframe()


@pytest.fixture
def fake_archive_fixture():
    """Produces an instance of the FakeArchive."""
    archive = FakeArchive([10, 10])
    x0 = np.array([1, 2, 3, 4])
    archive.initialize(len(x0))
    return archive, x0
