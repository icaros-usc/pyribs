"""Discount models and related utilities."""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

# TODO: Add these classes to documentation
# TODO: Get basic version working in sphere.py (replicate results), then mess with the API.
# TODO: What do annotations do? Is it enough for addressing torch types?
# TODO: Try installing in an env without torch

if TYPE_CHECKING:
    import torch

try:
    import torch
    from torch import nn

    IS_TORCH_AVAILABLE = True
except ImportError:

    class nn:  # noqa: N801, D101
        Module: object

    IS_TORCH_AVAILABLE = False


class MLP(nn.Module):
    """MLP archive.

    Feedforward network with identical activations on every layer. There is no
    activation on the last layer.

    Some methods return self so that one can do archive = MLPArchive().method()

    Args:
        layer_specs: List of tuples of (in_shape, out_shape, bias (optional)) for linear
            layers.
        activation: Activation layer class, e.g., nn.Tanh
        normalize: Whether to normalize the inputs. Pass "zero_one" to normalize to [0,
            1] or "negative_one_one" to normalize to [-1, 1]. Or pass False to indicate
            no normalization.
        norm_low: If normalize is True, this is the lower bound of the inputs for
            normalizing.
        norm_high: If normalize is True, this is the upper bound of the inputs for
            normalizing.
    """

    def __init__(
        self,
        layer_specs,
        activation,
        normalize: str | bool = False,
        norm_low: list[int] | None = None,
        norm_high: list[int] | None = None,
    ) -> None:
        super().__init__()

        layers = []
        for i, shape in enumerate(layer_specs):
            layers.append(
                nn.Linear(
                    shape[0], shape[1], bias=shape[2] if len(shape) == 3 else True
                )
            )
            if i != len(layer_specs) - 1:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

        self.normalize = normalize
        self.norm_low = norm_low
        self.norm_high = norm_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to [-1, 1].
        if self.normalize:
            self.norm_low = torch.as_tensor(
                self.norm_low, device=x.device
            ).requires_grad_(False)
            self.norm_high = torch.as_tensor(
                self.norm_high, device=x.device
            ).requires_grad_(False)

            if self.normalize == "negative_one_one":
                x = 2 * (x - self.norm_low) / (self.norm_high - self.norm_low) - 1
            elif self.normalize == "zero_one":
                x = (x - self.norm_low) / (self.norm_high - self.norm_low)
            else:
                raise ValueError("Unknown normalization method.")

        return self.model(x)

    def initialize(self, func, bias_func=nn.init.zeros_):
        """Initializes weights for Linear layers with func.

        Both funcs usually comes from nn.init -- pass func="pytorch_default" to
        use the default pytorch initialization everywhere.
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if func == "pytorch_default":
                    m.reset_parameters()
                else:
                    func(m.weight)
                    if m.bias is not None:
                        bias_func(m.bias)

        self.apply(init_weights)

        return self

    def serialize(self):
        """Returns 1D array with all parameters in the model."""
        return nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        nn.utils.vector_to_parameters(torch.from_numpy(array), self.parameters())
        return self

    def gradient(self):
        """Returns 1D array with gradient of all parameters in the model."""
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )


class DiscountModelBase(ABC):
    """Base class for discount models.

    Each method should try to use settings from the config as much as possible,
    instead of relying on method parameters. This helps keep the API consistent.
    """

    def __init__(
        self,
        cfg: DictConfig,
        seed: int,
        device: torch.device,
    ) -> None:
        """Initializes from a single config."""
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = device

    @staticmethod
    def count_params(model: nn.Module) -> int:
        """Utility for counting parameters in a torch model."""
        return sum(p.numel() for p in model.parameters())

    def num_params(self) -> int | dict[str, int]:
        """Counts number of parameters in this model.

        Returns:
            Number of params, or dict mapping from names of components to number
            of params for each component.
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Set the model into eval mode (like in PyTorch).

        Default is to switch all nn.Module attrs to eval mode.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.eval()

    def train(self) -> None:
        """Set the model into train mode (like in PyTorch).

        Default is to switch all nn.Module attrs to train mode.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.train()

    def training_loop(self, measures: torch.Tensor, targets: torch.Tensor) -> Any:
        """Regresses the discount model to match the given targets at the given measures.

        Args:
            measures: (batch_size, measure_dim) array of measure values.
            targets: (batch_size,) array of target values for the discount function.

        Returns:
            Any data associated with training.
        """
        raise NotImplementedError

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Retrieves discount values from the model for the given inputs (i.e., measures).

        Note that this method does NOT put the model in eval mode or use
        no_grad.

        Args:
            inputs: Inputs to the model, typically of (batch_size, input_dim).

        Returns:
            A (len(inputs),) array of discount values.
        """
        raise NotImplementedError

    def chunked_inference(
        self,
        inputs: np.ndarray | torch.Tensor,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Passes in the given inputs to the model in chunks.

        This method also puts the model in eval mode and uses no_grad when
        running the inference.
        """
        if verbose:
            log.info("Chunked inference")

        if batch_size is None:
            batch_size = len(inputs)

        dataloader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(inputs, dtype=torch.float32, device=self.device)
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        self.eval()
        discounts = []
        for (b_inputs,) in tqdm.tqdm(dataloader) if verbose else dataloader:
            with torch.no_grad():
                b_discounts = self.inference(b_inputs)
            discounts.append(b_discounts)
        self.train()

        # Concatenate all the chunks together.
        discounts = torch.cat(discounts, dim=0)

        # Turn (X, 1) into (X,).
        if discounts.ndim == 2:
            discounts = discounts.squeeze(dim=1)

        return discounts

    def save(self, directory: str | Path) -> None:
        """Saves the model in the given directory.

        Default is to save `self.model` in `directory / model.pth`.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        # pylint: disable-next = no-member
        torch.save(self.model.state_dict(), directory / "model.pth")

    def load(self, directory: str | Path) -> "DiscountModelBase":
        """Loads the model from the given directory.

        Default is to load `self.model` from `directory / model.pth`.
        """
        weights = torch.load(Path(directory) / "model.pth", map_location=self.device)
        # pylint: disable-next = no-member
        self.model.load_state_dict(weights)
        return self


# TODO: Merge with DiscountModelBase
class DiscountModelManager(DiscountModelBase):
    def __init__(self, cfg: DictConfig, seed: int, device: torch.device) -> None:
        super().__init__(cfg, seed, device)

        self.model = hydra.utils.instantiate(self.cfg.model.args)
        self.model.initialize(**self.cfg.init.weights)
        self.model.to(device)

        self.optimizer = hydra.utils.instantiate(
            self.cfg.optimizer.args,
            params=self.model.parameters(),
        )

    def num_params(self) -> int:
        return self.count_params(self.model)

    def training_loop(
        self, measures: torch.Tensor, targets: torch.Tensor
    ) -> list[float]:
        """Regresses the discount model to match the given targets at the given measures.

        Returns the losses from the training epochs.
        """
        dataset = TensorDataset(measures, targets)
        dataloader = DataLoader(
            dataset,
            self.cfg.train.batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss(reduction="mean")

        all_epoch_loss = []

        for _ in range(1, self.cfg.train.epochs + 1):
            epoch_loss = 0.0

            for b_measures, b_targets in dataloader:
                cur = self.model(b_measures).squeeze(dim=1)

                self.optimizer.zero_grad()
                loss = criterion(cur, b_targets)
                loss.backward()
                self.optimizer.step()

                # Multiply so that we track the total loss even if batch size
                # varies.
                epoch_loss += loss.item() * len(b_measures)

            # Divide by total elements in dataset.
            epoch_loss /= len(dataloader.dataset)
            all_epoch_loss.append(epoch_loss)

            if epoch_loss <= self.cfg.train.cutoff_loss:
                break

        return all_epoch_loss

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
