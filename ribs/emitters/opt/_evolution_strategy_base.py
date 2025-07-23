"""Provides EvolutionStrategyBase."""

from abc import ABC, abstractmethod

import numpy as np

# Number of times solutions can be resampled before triggering a warning.
BOUNDS_SAMPLING_THRESHOLD = 100

# Warning for resampling solutions too many times.
BOUNDS_WARNING = (
    "During bounds handling, this ES resampled at least "
    f"{BOUNDS_SAMPLING_THRESHOLD} times. This may indicate that your solution "
    "space bounds are too tight. When bounds are passed in, the ES resamples "
    "until all solutions are within the bounds -- if the bounds are too tight "
    "or the distribution is too large, the ES will resample forever."
)


class EvolutionStrategyBase(ABC):
    """Base class for evolution strategy optimizers for use with emitters.

    The basic usage is:

    - Initialize the optimizer and reset it.
    - Repeatedly:

        - Request new solutions with ``ask()``
        - Rank the solutions in the emitter (better solutions come first) and pass the
          indices and values back with ``tell()``.
        - Use ``check_stop()`` to see if the optimizer has reached a stopping condition,
          and if so, call ``reset()``.

    Args:
        sigma0 (float): Initial step size.
        solution_dim (int): Size of the solution space.
        batch_size (int): Number of solutions to evaluate at a time.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        lower_bounds (float or numpy.ndarray): scalar or (solution_dim,) array
            indicating lower bounds of the solution space. Scalars specify the same
            bound for the entire space, while arrays specify a bound for each dimension.
            Pass -np.inf in the array or scalar to indicated unbounded space.
        upper_bounds (float or numpy.ndarray): Same as above, but for upper bounds (and
            pass np.inf instead of -np.inf).
    """

    def __init__(
        self,
        sigma0,
        solution_dim,
        batch_size=None,
        seed=None,
        dtype=np.float64,
        lower_bounds=-np.inf,
        upper_bounds=np.inf,
    ):
        pass

    @abstractmethod
    def reset(self, x0):
        """Resets the ES to start at x0.

        Args:
            x0 (numpy.ndarray): Initial mean.
        """

    @abstractmethod
    def check_stop(self, ranking_values):
        """Checks if the ES should stop and be reset.

        Args:
            ranking_values (numpy.ndarray): Array of values that were used to rank the
                solutions. Shape can be either ``(batch_size,)`` or (batch_size,
                n_values)``, where ``batch_size`` is the number of solutions and
                ``n_values`` is the number of values that the ranker used. Note that
                unlike in :meth:`tell`, these values must be sorted according to the
                ``ranking_indices`` passed to :meth:`tell`.
        Returns:
            True if any of the stopping conditions are satisfied.
        """

    @abstractmethod
    def ask(self, batch_size=None):
        """Samples new solutions.

        Args:
            batch_size (int): batch size of the sample. Defaults to ``self.batch_size``.
        """

    @abstractmethod
    def tell(self, ranking_indices, ranking_values, num_parents):
        """Passes the solutions back to the ES.

        Args:
            ranking_indices (numpy.ndarray): Integer indices that are used to rank the
                solutions returned in :meth:`ask`. Note that these are NOT the ranks of
                the solutions. Rather, they are indices such that
                ``solutions[ranking_indices]`` will correctly rank the solutions (think
                of an argsort).
            ranking_values (numpy.ndarray): Array of values that were used to rank the
                solutions. Shape can be either ``(batch_size,)`` or (batch_size,
                n_values)``, where ``batch_size`` is the number of solutions and
                ``n_values`` is the number of values that the ranker used. Note that we
                assume a descending sort, i.e., higher values should come first.
            num_parents (int): Number of top solutions to select from the ranked
                solutions.
        """
