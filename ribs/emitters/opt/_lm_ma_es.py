"""Implementation of LM-MA-ES that can be used across various emitters.

Adapted from Nikolaus Hansen's pycma:
https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py
"""
import numba as nb
import numpy as np
from threadpoolctl import threadpool_limits

from ribs._utils import readonly
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase


class LMMAEvolutionStrategy(EvolutionStrategyBase):
    """LM-MA-ES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        n_vectors (int): Number of vectors to use in the approximation. If None,
            this defaults to be equal to the batch size.
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            dtype=np.float64,
            n_vectors=None):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)
        self._solutions = None

        if self.batch_size > self.solution_dim:
            raise ValueError(f"batch_size ({self.batch_size}) is greater than"
                             f" solution_dim ({self.solution_dim})")

        self.n_vectors = self.batch_size if n_vectors is None else n_vectors

        # z-vectors for the current solutions - initialized in ask().
        self._solution_z = None

        # Learning rates.
        self.csigma = 2 * self.batch_size / self.solution_dim
        # c_d,i = 1 / (1.5 ** (i - 1) * n) for i in 1..m
        self.cd = 1 / (1.5**np.arange(self.n_vectors) * self.solution_dim)
        # c_c,i = lambda / 4 ** (i - 1) * n for i in 1..m
        self.cc = self.batch_size / (4.0**np.arange(self.n_vectors) *
                                     self.solution_dim)
        assert self.cc.shape == (self.n_vectors,)
        assert self.cd.shape == (self.n_vectors,)

        # Strategy-specific params -> initialized in reset().
        self.current_gens = None
        self.mean = None
        self.sigma = None
        self.ps = None
        self.m = None

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        self.current_gens = 0
        self.sigma = self.sigma0
        self.mean = np.array(x0, self.dtype)

        # Setup evolution path variables.
        self.ps = np.zeros(self.solution_dim, dtype=self.dtype)

        # Setup the matrix vectors.
        self.m = np.zeros((self.n_vectors, self.solution_dim))

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to ``tell()``.

        Returns:
            True if any of the stopping conditions are satisfied.
        """
        # Sigma too small - Note: this was 1e-20 in the reference LM-MA-ES code.
        if self.sigma < 1e-12:
            return True

        # NOTE: We use norm here because we may have multiple ranking values.
        if (len(ranking_values) >= 2 and
                np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    @staticmethod
    @nb.jit(nopython=True)
    def _transform_and_check_sol(z, itrs, cd, m, mean, sigma, lower_bounds,
                                 upper_bounds):
        """Numba helper for transforming parameters to the solution space.

        Numba is important here since we may be resampling multiple times.
        """
        d = z
        for j in range(itrs):
            d = ((1 - cd[j]) * d  # (_, n)
                 + (
                     cd[j] * np.expand_dims(m[j], axis=0) *  # (1, n)
                     (np.expand_dims(m[j], axis=1).T @ d.T).T  # (_, 1)
                 )  # (_, n)
                )
        new_solutions = np.expand_dims(mean, axis=0) + sigma * d

        out_of_bounds = np.logical_or(
            new_solutions < np.expand_dims(lower_bounds, axis=0),
            new_solutions > np.expand_dims(upper_bounds, axis=0),
        )

        return new_solutions, out_of_bounds

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

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
        # NOTE: The LM-MA-ES uses mirror sampling by default, but we do not.
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)
        self._solution_z = np.empty((batch_size, self.solution_dim),
                                    dtype=self.dtype)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        while len(remaining_indices) > 0:
            z = self._rng.standard_normal(
                (len(remaining_indices), self.solution_dim))  # (_, n)
            self._solution_z[remaining_indices] = z

            new_solutions, out_of_bounds = self._transform_and_check_sol(
                z, min(self.current_gens, self.n_vectors), self.cd, self.m,
                self.mean, self.sigma, lower_bounds, upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

        return readonly(self._solutions)

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_strat_params(num_parents):
        """Calculates weights, mueff, and learning rates for LM-MA-ES."""
        # Create fresh weights for the number of parents found.
        weights = (np.log(num_parents + 0.5) -
                   np.log(np.arange(1, num_parents + 1)))
        total_weights = np.sum(weights)
        weights = weights / total_weights
        # The weights should sum to 1 for mueff.
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        return weights, mueff

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def tell(self, ranking_indices, num_parents):
        """Passes the solutions back to the optimizer.

        Args:
            ranking_indices (array-like of int): Indices that indicate the
                ranking of the original solutions returned in ``ask()``.
            num_parents (int): Number of top solutions to select from the
                ranked solutions.
        """
        self.current_gens += 1

        if num_parents == 0:
            return

        weights, mueff = self._calc_strat_params(num_parents)
        parents = self._solutions[ranking_indices][:num_parents]
        z_parents = self._solution_z[ranking_indices][:num_parents]
        z_mean = np.sum(weights[:, None] * z_parents, axis=0)

        # Recombination of the new mean - equivalent to line 12 in Algorithm 1
        # of Loschilov 2018 because weights sum to 1.
        self.mean = np.sum(weights[:, None] * parents, axis=0)

        # Update path for CSA.
        self.ps = ((1 - self.csigma) * self.ps +
                   np.sqrt(mueff * self.csigma * (2 - self.csigma)) * z_mean)

        # Update low-rank matrix representation.
        self.m = ((1 - self.cc[:, None]) * self.m +
                  np.sqrt(mueff * self.cc[:, None] *
                          (2 - self.cc[:, None])) * z_mean[None])

        # Update sigma. Note that CMA-ES offers a more complicated update rule.
        self.sigma *= np.exp(self.csigma / 2 *
                             (np.sum(self.ps**2) / self.solution_dim - 1))
