"""Useful utilities for all emitter tests."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.archives import FakeArchive


@pytest.fixture
def _archive_fixture():
	"""Provides a simple archive and initial solution."""
	archive = GridArchive([10, 10], [(-1, 1), (-1, 1)])
	x0 = np.array([1, 2, 3, 4])
	archive.initialize(len(x0))
	return archive, x0

@pytest.fixture
def _fake_archive_fixture():
	archive = FakeArchive([10, 10], [(-1, 1), (-1, 1)])
	x0 = np.array([1, 2, 3, 4])
	return archive, x0