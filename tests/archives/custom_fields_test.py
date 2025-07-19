"""Tests involving custom fields in the archive."""

import numpy as np
import pytest

from ribs.archives import GridArchive

from .conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


@pytest.fixture
def field_archive(data):
    """Archive with extra fields."""
    return GridArchive(
        solution_dim=data.archive.solution_dim,
        dims=data.archive.dims,
        ranges=list(zip(data.archive.lower_bounds, data.archive.upper_bounds)),
        extra_fields={"metadata": ((), object), "square": ((2, 2), np.int32)},
    )


def test_add_without_extra_fields(data, field_archive):
    with pytest.raises(ValueError):
        field_archive.add(data.solution, data.objective, data.measures)


def test_add_retrieve_extra_fields(data, field_archive, add_mode):
    solution = data.solution
    objective = data.objective
    measures = data.measures

    metadata = {"foobar": 42}
    square = [[0, 1], [1, 0]]

    if add_mode == "single":
        field_archive.add_single(
            solution,
            objective,
            measures,
            metadata=metadata,
            square=square,
        )
    else:
        field_archive.add(
            [solution],
            [objective],
            [measures],
            metadata=[metadata],
            square=[square],
        )

    elite = list(field_archive)[0]
    assert elite["metadata"] == metadata
    assert np.all(elite["square"] == square)
