"""Tests for the CategoricalArchive."""

import numpy as np

from ribs.archives import CategoricalArchive

from .grid_archive_test import assert_archive_elites


def test_properties():
    archive = CategoricalArchive(
        solution_dim=(),
        categories=[
            ["A", "B", "C"],
            ["One", "Two", "Three", "Four"],
        ],
    )

    assert (archive.dims == [3, 4]).all()
    assert archive.categories == [
        ["A", "B", "C"],
        ["One", "Two", "Three", "Four"],
    ]


def test_index_of():
    archive = CategoricalArchive(
        solution_dim=(),
        categories=[
            ["A", "B", "C"],
            ["One", "Two", "Three", "Four"],
        ],
    )

    indices = archive.index_of([["A", "One"], ["B", "Two"], ["C", "Four"]])
    expected_indices = [
        0,  # [0, 0]
        5,  # [1, 1]
        11,  # [2, 3]
    ]
    assert (indices == expected_indices).all()


def test_str_solutions():
    archive = CategoricalArchive(
        solution_dim=(),
        categories=[
            ["A", "B", "C"],
            ["One", "Two", "Three", "Four"],
        ],
        solution_dtype=object,
    )
    assert archive.solution_dim == ()
    assert archive.dtypes["solution"] == np.object_
    assert archive.dtypes["measures"] == np.object_

    add_info = archive.add(
        solution=["This is Bob", "Bob says hi", "Good job Bob", "Bob died"],
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 0, 1],
        measures=[["A", "Four"], ["B", "Three"], ["C", "One"], ["C", "One"]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 0, 1]).all()

    assert_archive_elites(
        archive=archive,
        batch_size=3,
        solution_batch=["This is Bob", "Bob says hi", "Bob died"],
        objective_batch=[0, 0, 1],
        measures_batch=[["A", "Four"], ["B", "Three"], ["C", "One"]],
        grid_indices_batch=[[0, 3], [1, 2], [2, 0]],
    )
