"""Tests for the DiscountModelManager.

Requires PyTorch to be installed.
"""

import pytest
import torch
from torch import nn

from ribs.discount_models import MLP, DiscountModelManager


@pytest.mark.parametrize("normalize", [None, "zero_one", "negative_one_one"])
def test_normalize_params(normalize):
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
