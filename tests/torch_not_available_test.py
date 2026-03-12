"""Tests for when PyTorch is not available."""

import importlib.util

import pytest

from ribs.discount_models import MLP, DiscountModelManager

if importlib.util.find_spec("torch") is not None:
    pytest.skip(
        "PyTorch found; skipping torch_not_available_test", allow_module_level=True
    )


def test_mlp_throws_error():
    with pytest.raises(
        ImportError, match=r"PyTorch must be installed to use the MLP\."
    ):
        MLP(layer_specs=[(2, 16), (16, 1)], activation=None)


def test_discount_model_manager_throws_error():
    with pytest.raises(
        ImportError,
        match=r"PyTorch must be installed to use the DiscountModelManager\.",
    ):
        DiscountModelManager(
            model=None,
            optimizer=None,
            device=None,
            train_epochs=1234,
            train_batch_size=1234,
            train_cutoff_loss=1234,
        )
