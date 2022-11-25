"""Provides GradientOptBase."""
from abc import ABC, abstractmethod


class GradientOptBase(ABC):
    """Base class for gradient-based optimizers.

    These optimizers maintain a current solution point :math:`\\theta`. The
    solution point may be obtained with the :attr:`theta` property, and it may
    be updated by passing a gradient to :meth:`step`. Finally, the point may be
    reset with :meth:`reset`.
    """

    @property
    @abstractmethod
    def theta(self):
        """The current solution point."""

    @abstractmethod
    def reset(self, theta0):
        """Resets the solution point to a new value.

        Args:
            theta0 (np.ndarray): The new solution point.
        """

    @abstractmethod
    def step(self, gradient):
        """Updates the solution based on the given gradient.

        Args:
            gradient (np.ndarray): The gradient to use for updating the
                solution.
        """
