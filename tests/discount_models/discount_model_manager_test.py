"""Tests for the DiscountModelManager.

Requires PyTorch to be installed.
"""

import pytest
import torch
from torch import nn

from ribs.discount_models import MLP, DiscountModelManager


@pytest.mark.parametrize("normalize", [None, "zero_one", "negative_one_one"])
def test_normalization_params(normalize):
    model = MLP(layer_specs=[(4, 16), (16, 1)], activation=nn.ReLU)
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device("cpu")

    if normalize is None:
        # No normalization, so nothing happens.
        DiscountModelManager(
            model=model,
            optimizer=optimizer,
            device=device,
            train_epochs=5,
            train_cutoff_loss=0.05,
            train_batch_size=32,
            normalize=normalize,
            norm_low=None,
            norm_high=None,
        )
    else:
        # normalize is passed in but norm_low and norm_high are not, which should result
        # in an error.
        with pytest.raises(
            ValueError,
            match=r"If normalize is not None,.*",
        ):
            DiscountModelManager(
                model=model,
                optimizer=optimizer,
                device=device,
                train_epochs=5,
                train_cutoff_loss=0.05,
                train_batch_size=32,
                normalize=normalize,
                norm_low=None,
                norm_high=None,
            )


def test_normalize_inputs():
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
        normalize="negative_one_one",
        norm_low=[-2, -5],
        norm_high=[3, 5],
    )

    # pylint: disable-next = protected-access
    normalized = discount_model_manager._normalize_inputs(
        [
            [-2.0, -5.0],
            [3.0, 5.0],
            [0.5, 0.0],
            [3.0, -5.0],
            [-4.5, 10.0],
        ]
    )

    assert isinstance(normalized, torch.Tensor)
    assert normalized.device == device
    assert torch.allclose(
        normalized,
        torch.asarray(
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]],
            device=device,
        ),
    )
