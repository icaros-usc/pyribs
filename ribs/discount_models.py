"""Discount models and related utilities.

.. autosummary::
    :toctree:

    MLP
    DiscountModelManager
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ribs.typing import Float, Int

__all__ = [
    "MLP",
    "DiscountModelManager",
]

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    IS_TORCH_AVAILABLE = True
except ImportError:
    # pylint: disable = invalid-name, missing-class-docstring
    class nn:  # noqa: N801
        class Module:
            pass

    IS_TORCH_AVAILABLE = False


# Developer Note: The documentation for this class is hacked. To list new methods,
# manually modify the template in docs/_templates/autosummary/class.rst


class MLP(nn.Module):
    """PyTorch multi-layer perceptron model.

    The MLP has identical activations on every layer, and no activation on the last
    layer. Each layer can be configured to have biases.

    .. note::

        This model requires `PyTorch <https://pytorch.org/>`_ to be installed, e.g., by
        running ``pip install torch``.

    Args:
        layer_specs: List of tuples specifying the linear layers. Each tuple can either
            contain ``(in_features, out_features)`` or ``(in_features, out_features,
            bias)``, where ``in_features`` and ``out_features`` are integers specifying
            the input and output shapes of the network, while ``bias`` is a bool
            indicating whether the layer should have a bias.
        activation: Activation layer class, e.g., :class:`torch.nn.Tanh`
    """

    def __init__(
        self,
        layer_specs: Collection[tuple[int, int] | tuple[int, int, bool]],
        activation: Callable,
    ) -> None:
        if not IS_TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use the MLP.")

        super().__init__()

        layers = []
        for i, spec in enumerate(layer_specs):
            layers.append(
                nn.Linear(
                    in_features=spec[0],
                    out_features=spec[1],
                    bias=spec[2] if len(spec) == 3 else True,
                )
            )
            if i != len(layer_specs) - 1:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the inputs through the MLP."""
        return self.model(x)

    def num_params(self) -> int:
        """Counts number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def serialize(self) -> np.ndarray:
        """Returns 1D array with all parameters in the model.

        Essentially, all the parameters of the model are retrieved, flattened, and
        concatenated together.

        Returns:
            1D array whose length corresponds to the number of parameters in the model.
        """
        return nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def deserialize(self, array: np.ndarray) -> MLP:
        """Loads parameters from 1D array.

        For example, given the array output by :meth:`serialize`, this method can be
        used to load that array back into the parameters of this model.

        Returns:
            The model itself, so that it is possible to call ``model =
            MLP(...).deserialize(x)``
        """
        nn.utils.vector_to_parameters(torch.from_numpy(array), self.parameters())
        return self

    def gradient(self) -> np.ndarray:
        """Returns 1D array with gradient of all parameters in the model.

        Essentially, all the gradients of the model's parameters are retrieved,
        flattened, and concatenated together.

        Returns:
            1D array whose length corresponds to the total size of all gradients in the
            model.
        """
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )


class DiscountModelManager:
    """Wraps a PyTorch model so it can be used as a discount model.

    This class handles operations like training the model to match new discount value
    targets (in :meth:`training_loop`) and performing inference (in :meth:`inference`).

    .. note::

        This class assumes all input and output data is of type float32, which is the
        default type in PyTorch. If different data types are needed, one solution may be
        to cast the data before/after calls to this class.

    .. note::

        This class requires `PyTorch <https://pytorch.org/>`_ to be installed, e.g., by
        running ``pip install torch``.

    Args:
        model: A PyTorch model that can take in batches of measures and output batches
            of scalar discount values. We assume this model has already been placed on
            the desired device.
        optimizer: A PyTorch optimizer that is set up to optimize the model's
            parameters. We use this to train the discount model to output new discount
            value targets. The optimizer state is maintained across calls to
            :meth:`training_loop`.
        device: A PyTorch device for placing tensors during training.
        train_epochs: When :meth:`training_loop` is called, the model will train until
            either (1) the total loss on each epoch is less than the
            ``train_cutoff_loss`` described below, or (2) the number of epochs reaches
            ``train_epochs``.
        train_cutoff_loss: See ``train_epochs``.
        train_batch_size: During each epoch of :meth:`training_loop`, the dataset of
            measures and targets will be used to train the model with this batch size.
        normalize: Whether to normalize the inputs. Pass None (default) to indicate no
            normalization. Alternatively, pass "zero_one" to normalize to ``[0, 1]`` or
            "negative_one_one" to normalize to ``[-1, 1]`` (along each dimension). To
            normalize to these values, we linearly transform from the range defined by
            ``norm_low`` and ``norm_high``, described below.
        norm_low: If ``normalize`` is True, this is the lower bound of the inputs for
            normalizing.
        norm_high: If ``normalize`` is True, this is the upper bound of the inputs for
            normalizing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        train_epochs: Int,
        train_cutoff_loss: Float,
        train_batch_size: Int,
        normalize: Literal["zero_one", "negative_one_one"] | None = None,
        norm_low: ArrayLike | None = None,
        norm_high: ArrayLike | None = None,
    ) -> None:
        if not IS_TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch must be installed to use the DiscountModelManager."
            )

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.train_epochs = train_epochs
        self.train_cutoff_loss = train_cutoff_loss
        self.train_batch_size = train_batch_size

        self.normalize = normalize
        self.norm_low = torch.asarray(
            norm_low, device=self.device, dtype=torch.float32
        ).requires_grad_(False)
        self.norm_high = torch.asarray(
            norm_high, device=self.device, dtype=torch.float32
        ).requires_grad_(False)

    def _normalize_inputs(self, x: ArrayLike) -> torch.Tensor:
        """Places x on the manager's device and normalizes it."""
        x = torch.asarray(x, device=self.device, dtype=torch.float32)
        if self.normalize is None:
            return x
        elif self.normalize == "negative_one_one":
            return 2.0 * (x - self.norm_low) / (self.norm_high - self.norm_low) - 1.0
        elif self.normalize == "zero_one":
            return (x - self.norm_low) / (self.norm_high - self.norm_low)
        else:
            raise ValueError(f"Unknown normalization method {self.normalize}.")

    def training_loop(self, measures: ArrayLike, targets: ArrayLike) -> list[float]:
        """Regresses the discount model to match the given targets at the given measures.

        Training proceeds until either (1) the total loss on each epoch is less than the
        ``train_cutoff_loss``, or (2) the number of epochs reaches ``train_epochs``. The
        loss function used during training is :class:`~torch.nn.MSELoss`.

        Args:
            measures: (batch_size, measure_dim) array of measure values.
            targets: (batch_size,) array of target values for the discount function.

        Returns:
            A list with the total MSE loss accumulated on each epoch, normalized/divided
            by the size of the dataset. Strictly speaking, the model is updated after
            every batch is passed through it, so this is not the loss that one would
            obtain if the measures were all passed through the model at once.
        """
        normalized_measures = self._normalize_inputs(measures)
        targets = torch.asarray(targets, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(normalized_measures, targets)
        dataloader = DataLoader(dataset, self.train_batch_size, shuffle=True)

        criterion = nn.MSELoss(reduction="mean")

        all_epoch_loss = []

        for _ in range(1, self.train_epochs + 1):
            epoch_loss = 0.0

            for b_norm_measures, b_targets in dataloader:
                cur = self.model(b_norm_measures).squeeze(dim=1)

                self.optimizer.zero_grad()
                loss = criterion(cur, b_targets)
                loss.backward()
                self.optimizer.step()

                # Multiply so that we track the total loss even if batch size varies.
                epoch_loss += loss.item() * len(b_norm_measures)

            # Divide by total elements in dataset.
            epoch_loss /= len(dataloader.dataset)
            all_epoch_loss.append(epoch_loss)

            if epoch_loss <= self.train_cutoff_loss:
                break

        return all_epoch_loss

    def inference(
        self,
        measures: ArrayLike,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Computes discount values at the given measures using the model.

        This method also puts the model in eval mode and uses :class:`torch.no_grad`.

        Args:
            measures: Inputs to the model of size (n_measures, measure_dim).
            batch_size: If passed in, the model will only be passed ``batch_size``
                inputs at a time. This can be useful if, for instance, the model is very
                large and there is insufficient memory to handle many inputs
                simultaneously.

        Returns:
            The discount values at the input measures.
        """
        if batch_size is None:
            batch_size = len(measures)

        normalized_measures = self._normalize_inputs(measures)
        dataloader = DataLoader(
            dataset=TensorDataset(normalized_measures),
            batch_size=batch_size,
            shuffle=False,
        )

        self.model.eval()
        discounts = []
        with torch.no_grad():
            for (b_norm_measures,) in dataloader:
                b_discounts = self.model(b_norm_measures)
            discounts.append(b_discounts)
        self.model.train()

        # Concatenate all the chunks together.
        discounts = torch.cat(discounts, dim=0)

        # Turn (X, 1) into (X,).
        if discounts.ndim == 2:
            discounts = discounts.squeeze(dim=1)

        return discounts.detach().cpu().numpy()
