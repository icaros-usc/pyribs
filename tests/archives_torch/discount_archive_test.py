"""Tests for DiscountArchive.

Requires PyTorch due to the use of discount models.
"""

import numpy as np
import pytest
import torch
from torch import nn

from ribs.archives import DiscountArchive, GridArchive
from ribs.discount_models import MLP, DiscountModelManager

# pylint: disable = redefined-outer-name


@pytest.fixture
def archive():
    """Builds a DiscountArchive with a small MLP."""
    result_archive = GridArchive(
        solution_dim=3, dims=[10, 10], ranges=[(-1, 1), (-1, 1)]
    )
    model = MLP([[2, 16], [16, 1]], nn.ReLU)
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
    )


def test_attrs(archive):
    assert not archive.empty
    assert archive.dtypes == {
        "solution": np.float64,
        "objective": np.float64,
        "measures": np.float64,
    }
    assert archive.learning_rate == 0.1
    assert archive.threshold_min == 0.0
    assert archive.init_train_points == 100
    assert archive.empty_points == 10


def test_init_discount_model(archive):
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


def test_train_discount_model(archive):
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
    assert info["solution_targets"].shape == (2,)
    assert info["empty_measures"].shape == (10, 2)
    assert len(info["losses"]) == info["epochs"]


# TODO: test dtypes
def test_add():
    pass
