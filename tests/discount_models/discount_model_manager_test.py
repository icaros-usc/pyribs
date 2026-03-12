"""Tests for the DiscountModelManager.

Requires PyTorch to be installed.
"""

import pytest
import torch
from torch import nn

from ribs.discount_models import MLP, DiscountModelManager


@pytest.mark.parametrize("normalize_measures", [None, "zero_one", "negative_one_one"])
def test_measure_norm_params(normalize_measures):
    model = MLP(layer_specs=[(4, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")

    if normalize_measures is None:
        # No normalization, so nothing happens.
        DiscountModelManager(
            model=model,
            optimizer=optimizer,
            device=device,
            train_epochs=5,
            train_cutoff_loss=0.05,
            train_batch_size=32,
            normalize_measures=normalize_measures,
            measures_low=None,
            measures_high=None,
        )
    else:
        # normalize_measures is passed in but measures_low and measures_high are not,
        # which should result in an error.
        with pytest.raises(
            ValueError,
            match=r"If normalize_measures is not None,.*",
        ):
            DiscountModelManager(
                model=model,
                optimizer=optimizer,
                device=device,
                train_epochs=5,
                train_cutoff_loss=0.05,
                train_batch_size=32,
                normalize_measures=normalize_measures,
                measures_low=None,
                measures_high=None,
            )


@pytest.mark.parametrize("normalize_discount", [None, "zero_one", "negative_one_one"])
def test_discount_norm_params(normalize_discount):
    model = MLP(layer_specs=[(4, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")

    if normalize_discount is None:
        # No normalization, so nothing happens.
        DiscountModelManager(
            model=model,
            optimizer=optimizer,
            device=device,
            train_epochs=5,
            train_cutoff_loss=0.05,
            train_batch_size=32,
            normalize_discount=normalize_discount,
            discount_low=None,
            discount_high=None,
        )
    else:
        # normalize_discount is passed in but discount_low and discount_high are not,
        # which should result in an error.
        with pytest.raises(
            ValueError,
            match=r"If normalize_discount is not None,.*",
        ):
            DiscountModelManager(
                model=model,
                optimizer=optimizer,
                device=device,
                train_epochs=5,
                train_cutoff_loss=0.05,
                train_batch_size=32,
                normalize_discount=normalize_discount,
                discount_low=None,
                discount_high=None,
            )


@pytest.mark.parametrize("normalize_measures", [None, "zero_one", "negative_one_one"])
def test_normalize(normalize_measures):
    model = MLP(layer_specs=[(4, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")
    discount_model_manager = DiscountModelManager(
        model=model,
        optimizer=optimizer,
        device=device,
        train_epochs=5,
        train_cutoff_loss=0.05,
        train_batch_size=32,
        normalize_measures=normalize_measures,
        measures_low=[-2, -5],
        measures_high=[3, 5],
    )

    # pylint: disable-next = protected-access
    normalized = discount_model_manager._normalize(
        [
            [-2.0, -5.0],
            [3.0, 5.0],
            [0.5, 0.0],
            [3.0, -5.0],
            [-4.5, 10.0],
        ],
        discount_model_manager.normalize_measures,
        discount_model_manager.measures_low,
        discount_model_manager.measures_high,
    )

    assert isinstance(normalized, torch.Tensor)
    assert normalized.device == device

    if normalize_measures is None:
        assert torch.allclose(
            normalized,
            torch.asarray(
                [
                    [-2.0, -5.0],
                    [3.0, 5.0],
                    [0.5, 0.0],
                    [3.0, -5.0],
                    [-4.5, 10.0],
                ],
                device=device,
            ),
        )
    elif normalize_measures == "zero_one":
        assert torch.allclose(
            normalized,
            torch.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                    [-0.5, 1.5],
                ],
                device=device,
            ),
        )
    elif normalize_measures == "negative_one_one":
        assert torch.allclose(
            normalized,
            torch.asarray(
                [
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, -1.0],
                    [-2.0, 2.0],
                ],
                device=device,
            ),
        )


@pytest.mark.parametrize("normalize_discount", [None, "zero_one", "negative_one_one"])
def test_unnormalize(normalize_discount):
    model = MLP(layer_specs=[(4, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")
    discount_model_manager = DiscountModelManager(
        model=model,
        optimizer=optimizer,
        device=device,
        train_epochs=5,
        train_cutoff_loss=0.05,
        train_batch_size=32,
        normalize_discount=normalize_discount,
        discount_low=[-2, -5],
        discount_high=[3, 5],
    )

    target = torch.asarray(
        [
            [-2.0, -5.0],
            [3.0, 5.0],
            [0.5, 0.0],
            [3.0, -5.0],
            [-4.5, 10.0],
        ],
        device=device,
    )

    if normalize_discount is None:
        # pylint: disable-next = protected-access
        unnormalized = discount_model_manager._unnormalize(
            target,
            discount_model_manager.normalize_discount,
            discount_model_manager.discount_low,
            discount_model_manager.discount_high,
        )
    elif normalize_discount == "zero_one":
        # pylint: disable-next = protected-access
        unnormalized = discount_model_manager._unnormalize(
            torch.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                    [-0.5, 1.5],
                ],
                device=device,
            ),
            discount_model_manager.normalize_discount,
            discount_model_manager.discount_low,
            discount_model_manager.discount_high,
        )
    elif normalize_discount == "negative_one_one":
        # pylint: disable-next = protected-access
        unnormalized = discount_model_manager._unnormalize(
            torch.asarray(
                [
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, -1.0],
                    [-2.0, 2.0],
                ],
                device=device,
            ),
            discount_model_manager.normalize_discount,
            discount_model_manager.discount_low,
            discount_model_manager.discount_high,
        )
    else:
        raise ValueError("Unknown value for normalize_discount.")

    assert isinstance(unnormalized, torch.Tensor)
    assert unnormalized.device == device
    assert torch.allclose(unnormalized, target)
