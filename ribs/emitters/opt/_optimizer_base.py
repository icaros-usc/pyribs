"""Provides OptimizerBase."""
from abc import ABC, abstractmethod

import numpy as np


class OptimizerBase(ABC):
    """Base class for optimizers for use with emitters.

    The basic usage is:
    - Initialize the optimizer and reset it.
    - Repeatedly:
      - Request new solutions with ``ask()``
      - Rank the solutions in the emitter (better solutions come first) and
        pass them back with ``tell()``.
      - Use ``check_stop()`` to see if the optimizer has reached a stopping
        condition, and if so, call ``reset()``.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
    """

    def __init__(self, sigma0, batch_size, solution_dim, seed,
                 dtype):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

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
    def tell(self, solutions, num_parents, ranking_indices=None):
        """Passes the solutions back to the optimizer.

        Args:
            solutions (np.ndarray): Array of ranked solutions. The user should
                have determined some way to rank the solutions, such as by
                objective value. It is important that _all_ of the solutions
                initially given in ask() are returned here.
            num_parents (int): Number of best solutions to select.
            ranking_indices (array-like of int): Indices that were used to
                order solutions from the original solutions returned in ask().
                This argument is not used by all optimizers.
        """
