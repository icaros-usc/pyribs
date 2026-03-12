"""Tests for DiscountArchive.

Requires PyTorch due to the use of discount models.
"""

import numpy as np
import pytest
import torch
from torch import nn

from ribs.archives import CVTArchive, DiscountArchive, GridArchive
from ribs.discount_models import MLP, DiscountModelManager

# pylint: disable = redefined-outer-name


def compute_archive_centers(archive: GridArchive | CVTArchive) -> np.ndarray:
    """Computes the center in measure space of each cell in the archive."""
    if isinstance(archive, GridArchive):
        grid_indices = archive.int_to_grid_index(np.arange(archive.cells))
        return (
            (grid_indices + 0.5) / archive.dims
        ) * archive.interval_size + archive.lower_bounds
    elif isinstance(archive, CVTArchive):
        return archive.centroids
    else:
        raise ValueError("Cannot compute centers for this archive.")


def check_all_measures_in_set(measures: np.ndarray, reference: np.ndarray) -> bool:
    """Checks that every measure in measures exists in reference."""
    for m in measures:
        is_close = np.abs(reference - m[None]) < 1e-5
        # Collapse along axis 1 to require that all components be equal.
        assert np.any(np.all(is_close, axis=1))


@pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request):
    """Fixture for archive dtype."""
    return request.param


@pytest.fixture(params=["GridArchive", "CVTArchive"])
def result_archive(request):
    """Fixture for the result archive."""
    if request.param == "GridArchive":
        return GridArchive(solution_dim=3, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    else:
        rng = np.random.default_rng(42)
        return CVTArchive(
            solution_dim=3,
            centroids=rng.uniform(low=[-1, -1], high=[1, 1], size=(100, 2)),
            ranges=[(-1, 1), (-1, 1)],
        )


@pytest.fixture
def archive(dtype, result_archive):
    """Builds a DiscountArchive with a small MLP."""
    model = MLP(layer_specs=[(2, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")
    discount_model_manager = DiscountModelManager(
        model=model,
        optimizer=optimizer,
        device=device,
        train_epochs=5,
        train_cutoff_loss=0.05,
        train_batch_size=32,
    )
    return DiscountArchive(
        solution_dim=3,
        measure_dim=2,
        learning_rate=0.1,
        threshold_min=0.0,
        discount_model_manager=discount_model_manager,
        result_archive=result_archive,
        init_train_points=100,
        empty_points=10,
        dtype=dtype,
    )


def test_attrs(archive, dtype):
    assert not archive.empty
    assert archive.dtypes == {"solution": dtype, "objective": dtype, "measures": dtype}
    assert archive.learning_rate == 0.1
    assert archive.threshold_min == 0.0
    assert archive.init_train_points == 100
    assert archive.empty_points == 10


def test_init_discount_model(archive, result_archive):
    info = archive.init_discount_model()

    assert info.keys() == {
        "solution_measures",
        "solution_targets",
        "empty_measures",
        "epochs",
        "losses",
    }
    assert info["solution_measures"].shape == (0, 2)
    assert info["solution_targets"].shape == (0,)
    assert info["empty_measures"].shape == (100, 2)
    assert len(info["losses"]) == info["epochs"]

    # All the empty measures should be unique and contained in the archive centers.
    assert np.unique(info["empty_measures"], axis=0).shape == (100, 2)
    check_all_measures_in_set(
        info["empty_measures"], compute_archive_centers(result_archive)
    )


def test_train_discount_model(archive, result_archive):
    archive.init_discount_model()
    archive.add(
        solution=[[1, 2, 3], [4, 5, 6]],
        objective=[1.0, 2.0],
        measures=[[0, 0], [1, 1]],
    )

    info = archive.train_discount_model()

    assert info.keys() == {
        "solution_measures",
        "solution_targets",
        "empty_measures",
        "epochs",
        "losses",
    }
    assert info["solution_measures"].shape == (2, 2)
    assert np.allclose(info["solution_measures"], [[0, 0], [1, 1]])
    assert info["solution_targets"].shape == (2,)
    # It's difficult to know what the value of solution_targets should be since the
    # discount model is not guaranteed to output a discount value of `threshold_min`
    # everywhere.
    assert info["empty_measures"].shape == (10, 2)
    assert len(info["losses"]) == info["epochs"]

    # All the empty measures should be unique.
    assert np.unique(info["empty_measures"], axis=0).shape == (10, 2)
    check_all_measures_in_set(
        info["empty_measures"], compute_archive_centers(result_archive)
    )


def test_add(archive, dtype):
    archive.init_discount_model()
    add_info = archive.add(
        solution=[[1, 2, 3], [4, 5, 6]],
        objective=[
            10.0,  # Should be high enough that the solutions are considered NEW.
            -10.0,  # Should be low enough that the solution is considered NOT_ADDED.
        ],
        measures=[[0, 0], [1, 1]],
    )

    assert add_info["status"].dtype == np.int32
    assert (add_info["status"] == [2, 0]).all()
    assert add_info["value"].dtype == dtype
    assert add_info["value"].shape == (2,)
    assert add_info["discount"].dtype == dtype
    assert add_info["discount"].shape == (2,)
