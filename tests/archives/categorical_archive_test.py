"""Tests for the CategoricalArchive."""
from ribs.archives import CategoricalArchive


def test_init():
    archive = CategoricalArchive(
        solution_dim=(),
        categories=[
            ["Alpha", "Beta", "Gamma"],
            ["One", "Two", "Three", "Four"],
        ],
    )

    assert (archive.dims == [3, 4]).all()
    assert archive.categories == [
        ["Alpha", "Beta", "Gamma"],
        ["One", "Two", "Three", "Four"],
    ]


def test_index_of():
    archive = CategoricalArchive(
        solution_dim=(),
        categories=[
            ["Alpha", "Beta", "Gamma"],
            ["One", "Two", "Three", "Four"],
        ],
    )

    indices = archive.index_of([["Alpha", "One"], ["Beta", "Two"],
                                ["Gamma", "Four"]])
    expected_indices = [
        0,  # [0, 0]
        5,  # [1, 1]
        11,  # [2, 3]
    ]
    assert (indices == expected_indices).all()
