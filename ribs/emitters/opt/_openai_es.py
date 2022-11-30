"""Implementation of OpenAI ES that can be used across various emitters.

See here for more info: https://arxiv.org/abs/1703.03864
"""
import numpy as np
from threadpoolctl import threadpool_limits

from ribs._utils import readonly
from ribs.emitters.opt._adam_opt import AdamOpt
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase


class OpenAIEvolutionStrategy(EvolutionStrategyBase):
    """OpenAI-ES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        mirror_sampling (bool): Whether to use mirror sampling when gathering
            solutions. Defaults to True.
        adam_kwargs (dict): Keyword arguments passed to :class:`AdamOpt`.
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            dtype=np.float64,
            mirror_sampling=True,
            **adam_kwargs):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)
        self._solutions = None

        self.mirror_sampling = mirror_sampling

        # Default batch size should be an even number for mirror sampling.
        if batch_size is None and self.batch_size % 2 != 0:
            self.batch_size += 1

        if self.batch_size <= 1:
            raise ValueError("Batch size of 1 is not supported because rank"
                             " normalization does not work with batch size of"
                             " 1.")

        if self.mirror_sampling and self.batch_size % 2 != 0:
            raise ValueError("If using mirror sampling, batch_size must be an"
                             " even number.")

        # Strategy-specific params -> initialized in reset().
        self.adam_opt = AdamOpt(self.solution_dim, **adam_kwargs)
        self.last_update_ratio = None
        self.noise = None

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        self.adam_opt.reset(x0)
        self.last_update_ratio = np.inf  # Updated at end of tell().
        self.noise = None  # Becomes (batch_size, solution_dim) array in ask().

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to ``tell()``.

        Returns:
            True if any of the stopping conditions are satisfied.
        """
        if self.last_update_ratio < 1e-9:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (len(ranking_values) >= 2 and
                np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Note: Bounds are currently not enforced.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
            batch_size (int): batch size of the sample. Defaults to
                ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        while len(remaining_indices) > 0:
            if self.mirror_sampling:
                noise_half = self._rng.standard_normal(
                    (batch_size // 2, self.solution_dim), dtype=self.dtype)
                self.noise = np.concatenate((noise_half, -noise_half))
            else:
                self.noise = self._rng.standard_normal(
                    (batch_size, self.solution_dim), dtype=self.dtype)

            # TODO Numba
            new_solutions = (self.adam_opt.theta[None] +
                             self.sigma0 * self.noise)
            out_of_bounds = np.logical_or(
                new_solutions < np.expand_dims(lower_bounds, axis=0),
                new_solutions > np.expand_dims(upper_bounds, axis=0),
            )

            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

        return readonly(self._solutions)

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def tell(
            self,
            ranking_indices,
            num_parents,  # pylint: disable = unused-argument
    ):
        """Passes the solutions back to the optimizer.

        Args:
            ranking_indices (array-like of int): Indices that indicate the
                ranking of the original solutions returned in ``ask()``.
            num_parents (int): Number of top solutions to select from the
                ranked solutions.
        """
        # Indices come in decreasing order, so we reverse to get them to
        # increasing order.
        ranks = np.empty(self.batch_size, dtype=np.int32)

        # Assign ranks -- ranks[i] tells the rank of noise[i].
        ranks[ranking_indices[::-1]] = np.arange(self.batch_size)

        # Normalize ranks to [-0.5, 0.5].
        ranks = (ranks / (self.batch_size - 1)) - 0.5

        # Compute the gradient.
        if self.mirror_sampling:
            half_batch = self.batch_size // 2
            gradient = np.sum(
                self.noise[:half_batch] *
                (ranks[:half_batch] - ranks[half_batch:])[:, None],
                axis=0)
            gradient /= half_batch * self.sigma0
        else:
            gradient = np.sum(self.noise * ranks[:, None], axis=0)
            gradient /= self.batch_size * self.sigma0

        # Used to compute last update ratio.
        theta_prev = self.adam_opt.theta.copy()

        self.adam_opt.step(gradient)

        self.last_update_ratio = (np.linalg.norm(self.adam_opt.theta - theta_prev) /
                                  np.linalg.norm(self.adam_opt.theta))
