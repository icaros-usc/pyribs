"""Useful utilities for all emitter tests."""
import numpy as np
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

    def __init__(self, dims):
        self._dims = np.array(dims)
        behavior_dim = len(self._dims)
        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=behavior_dim,
        )

    def get_random_elite(self):
        return (
            self._solutions[(0,) * self._behavior_dim],
            self._objective_values[0],
            self._behavior_values[0],
        )

    def add(self, solution, objective_value, behavior_values):
        return True

    # may not ever be called, actually
    @jit(nopython=True)
    def _get_index(self, behavior_values):
        return np.full_like(behavior_values, 0)

    def as_pandas(self):
        return pd.Dataframe()


@pytest.fixture
def fake_archive_fixture():
    archive = FakeArchive([10, 10])
    x0 = np.array([1, 2, 3, 4])
    archive.initialize(len(x0))
    return archive, x0

