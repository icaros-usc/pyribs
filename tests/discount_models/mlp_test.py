"""Tests for the MLP.

Requires PyTorch to be installed.
"""

import numpy as np
import torch
from torch import nn

from ribs.discount_models import MLP


def test_serialization():
    mlp = MLP(
        layer_specs=[
            (4, 16, False),  # No bias.
            (16, 16, True),  # Has bias.
            (16, 1),  # Has bias.
        ],
        activation=nn.ReLU,
    )

    # num_params()
    assert mlp.num_params() == 4 * 16 + 16 * 16 + 16 + 16 * 1 + 1

    # serialize()
    serialized = mlp.serialize()
    assert isinstance(serialized, np.ndarray)
    assert serialized.shape == (4 * 16 + 16 * 16 + 16 + 16 * 1 + 1,)

    # deserialize()
    new_params = np.zeros_like(serialized)
    mlp.deserialize(new_params)
    for p in mlp.parameters():
        assert torch.all(p.data == 0.0)


def test_gradient():
    mlp = MLP(
        layer_specs=[
            (4, 16, False),  # No bias.
            (16, 16, True),  # Has bias.
            (16, 1),  # Has bias.
        ],
        activation=nn.ReLU,
    )

    # Must do a forward and backward pass to populate the gradient.
    mlp(torch.ones((1, 4))).backward()

    assert mlp.gradient().shape == (4 * 16 + 16 * 16 + 16 + 16 * 1 + 1,)
