"""Useful utilities for all emitter tests."""
import numpy as np
import pytest
from numba import jit
from ribs.archives import ArchiveBase, GridArchive


@pytest.fixture
def archive_fixture():
    """Provides a simple archive and initial solution."""
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10], [(-1, 1), (-1, 1)])
    return archive, x0


class FakeArchive(ArchiveBase):
    """Bare-bones archive solely for emitter benchmarking.

    This archive is used in emitter benchmarking, so each method does as little
    as possible. Since the methods have no meaningful functionality, this
    archive is "fake."
    """

    def __init__(self, dims):
        self._dims = np.array(dims)
        behavior_dim = len(self._dims)
        ArchiveBase.__init__(
            self,
            solution_dim=4,
            cells=np.product(self._dims),
            behavior_dim=behavior_dim,
        )

    def add(self, solution, objective, measures, metadata=None):
        return (0, 0)

    @jit(nopython=True)
    def index_of(self, measures):
        return np.full_like(measures, 0)

    def as_pandas(self):
        return pd.Dataframe()


@pytest.fixture
def fake_archive_fixture():
    """Produces an instance of the FakeArchive."""
    archive = FakeArchive([10, 10])
    x0 = np.array([1, 2, 3, 4])
    return archive, x0
