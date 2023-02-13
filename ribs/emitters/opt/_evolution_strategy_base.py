"""Provides EvolutionStrategyBase."""
from abc import ABC, abstractmethod

import numpy as np


class EvolutionStrategyBase(ABC):
    """Base class for evolution strategy optimizers for use with emitters.

    The basic usage is:

    - Initialize the optimizer and reset it.
    - Repeatedly:

        - Request new solutions with ``ask()``
        - Rank the solutions in the emitter (better solutions come first) and
          pass the indices back with ``tell()``.
        - Use ``check_stop()`` to see if the optimizer has reached a stopping
          condition, and if so, call ``reset()``.

    Args:
        sigma0 (float): Initial step size.
        solution_dim (int): Size of the solution space.
        batch_size (int): Number of solutions to evaluate at a time.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
    """

    def __init__(self,
                 sigma0,
                 solution_dim,
                 batch_size=None,
                 seed=None,
                 dtype=np.float64):
        pass

    @abstractmethod
    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """

    @abstractmethod
    def check_stop(self, ranking_values):
        """Checks if the optimizer should stop and be reset.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to tell().
        Returns:
            True if any of the stopping conditions are satisfied.
        """

    @abstractmethod
    def ask(self, lower_bounds, upper_bounds):
        """Samples new solutions from the Gaussian distribution.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
        """

    @abstractmethod
    def tell(self, ranking_indices, num_parents):
        """Passes the solutions back to the optimizer.

        Args:
            ranking_indices (array-like of int): Indices that indicate the
                ranking of the original solutions returned in ``ask()``.
            num_parents (int): Number of top solutions to select from the
                ranked solutions.
        """
