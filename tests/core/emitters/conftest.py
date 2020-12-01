"""Useful utilities for all emitter tests."""
import numpy as np
import pytest
from numba import jit

from ribs.archives import GridArchive
#from ribs.archives import FakeArchive


@pytest.fixture
def _archive_fixture():
	"""Provides a simple archive and initial solution."""
	archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
	x0 = np.array([1, 2, 3, 4])
	archive.initialize(len(x0))
	return archive, x0


from ribs.archives._archive_base import ArchiveBase

_EPSILON = 1e-9

class FakeArchive(ArchiveBase):

	def __init__(self, dims, ranges, seed=None):
		self._dims = np.array(dims)
		behavior_dim = len(self._dims)
		ArchiveBase.__init__(
			self,
			storage_dims=tuple(self._dims),
			behavior_dim=behavior_dim,
			seed=seed,
		)

		ranges = list(zip(*ranges))
		self._lower_bounds = np.array(ranges[0])
		self._upper_bounds = np.array(ranges[1])
		self._interval_size = self._upper_bounds - self._lower_bounds

	# get_random_elite
	def get_random_elite(self):
		return (
			self._solutions[0],
			self._objective_values[0],
			self._behavior_values[0],
		)

	# add
	def add(self, solution, objective_value, behavior_values):
		return True

	# _get_index
	# may not ever be called, actually
	@jit(nopython=True)
	def _get_index(self, behavior_values):
		return np.full_like(behavior_values, 0)

	# as_pandas
	def as_pandas(self):
		return pd.Dataframe()


@pytest.fixture
def _fake_archive_fixture():
	archive = FakeArchive([10, 10], [(-1, 1), (-1, 1)])
	x0 = np.array([1, 2, 3, 4])
	return archive, x0

