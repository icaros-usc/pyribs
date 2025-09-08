"""Provides GradientOptBase."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ribs.typing import Float


class GradientOptBase(ABC):
    r"""Base class for gradient-based optimizers.

    .. note::
        These optimizers are designed for gradient ascent rather than gradient descent.

    These optimizers maintain a current solution point :math:`\theta`. The solution
    point is obtained with the :attr:`theta` property, and it is updated by passing a
    gradient to :meth:`step`. Finally, the point can be reset to a new value with
    :meth:`reset`.

    Your constructor may take in additional arguments beyond ``theta0`` and ``lr``, but
    expect that these two arguments will always be passed in.

    Args:
        theta0: Initial solution. 1D array.
        lr: Learning rate for the update.
    """

    def __init__(self, theta0: ArrayLike, lr: Float) -> None:
        pass

    @property
    @abstractmethod
    def theta(self) -> np.ndarray:
        """The current solution point."""

    @abstractmethod
    def reset(self, theta0: ArrayLike) -> None:
        """Resets the solution point to a new value.

        Args:
            theta0: The new solution point. 1D array.
        """

    @abstractmethod
    def step(self, gradient: ArrayLike) -> None:
        """Ascends the solution based on the given gradient.

        Args:
            gradient: The (estimated) gradient of the current solution point. 1D array.
        """
