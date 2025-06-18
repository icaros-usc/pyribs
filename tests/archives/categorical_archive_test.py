"""Tests for the CategoricalArchive."""
from ribs.archives import CategoricalArchive


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
