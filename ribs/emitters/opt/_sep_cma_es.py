"""Implementation of sep-CMA-ES that can be used across various emitters.

Adapted from Nikolaus Hansen's pycma:
https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py
"""
import numba as nb
import numpy as np
from threadpoolctl import threadpool_limits

from ribs._utils import readonly
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase


class DiagonalMatrix:
    """Maintains a diagonal covariance matrix.

    Args:
        dimension (int): Size of the (square) covariance matrix.
        dtype (str or data-type): Data type of the matrix, typically np.float32
            or np.float64.
    """

    def __init__(self, dimension, dtype):
        self.cov = np.ones((dimension,), dtype=dtype)
        self.dtype = dtype

        # The last evaluation on which the eigensystem was updated.
        self.updated_eval = 0

    @property
    def condition_number(self):
        """Condition number of the covariance matrix."""
        return np.max(self.cov) / np.min(self.cov)

    @property
    def eigenvalues(self):
        """Eigenvalues (equal to covariance matrix since it is diagonal)."""
        return self.cov

    @property
    def invsqrt(self):
        """C^-1/2."""
        return 1 / np.sqrt(self.cov)


class SeparableCMAEvolutionStrategy(EvolutionStrategyBase):
    """sep-CMA-ES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        weight_rule (str): Method for generating weights. Either "truncation"
            (positive weights only) or "active" (include negative weights).
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            dtype=np.float64):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)
        self._solutions = None

        # Strategy-specific params -> initialized in reset().
        self.current_eval = None
        self.mean = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.cov = None

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        self.current_eval = 0
        self.sigma = self.sigma0
        self.mean = np.array(x0, self.dtype)

        # Setup evolution path variables.
        self.pc = np.zeros(self.solution_dim, dtype=self.dtype)
        self.ps = np.zeros(self.solution_dim, dtype=self.dtype)

        # Setup the covariance matrix.
        self.cov = DiagonalMatrix(self.solution_dim, self.dtype)

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
        if self.cov.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(max(self.cov.eigenvalues))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (len(ranking_values) >= 2 and
                np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    @staticmethod
    @nb.jit(nopython=True)
    def _transform_and_check_sol(unscaled_params, transform_vec, mean,
                                 lower_bounds, upper_bounds):
        """Numba helper for transforming parameters to the solution space."""
        solutions = (np.expand_dims(transform_vec, axis=0) * unscaled_params +
                     np.expand_dims(mean, axis=0))
        out_of_bounds = np.logical_or(
            solutions < np.expand_dims(lower_bounds, axis=0),
            solutions > np.expand_dims(upper_bounds, axis=0),
        )
        return solutions, out_of_bounds

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
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)
        transform_vec = np.sqrt(self.cov.eigenvalues)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        while len(remaining_indices) > 0:
            unscaled_params = self._rng.normal(
                0.0,
                self.sigma,
                (len(remaining_indices), self.solution_dim),
            ).astype(self.dtype)
            new_solutions, out_of_bounds = self._transform_and_check_sol(
                unscaled_params, transform_vec, self.mean, lower_bounds,
                upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

        return readonly(self._solutions)

    @staticmethod
    def _conedf(df, mu, solution_dim):
        """Used for computing separable learning rate."""
        return 1. / (df + 2. * np.sqrt(df) + float(mu) / solution_dim)

    @staticmethod
    def _cmudf(df, mu, alphamu):
        """Used for computing separable learning rate."""
        return (alphamu + mu + 1. / mu - 2) / (df + 4 * np.sqrt(df) + mu / 2.)

    def _calc_strat_params(self, solution_dim, num_parents):
        """Calculates weights, mueff, and learning rates for sep-CMA-ES.

        Refer here
        https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py#L3767
        for the learning rates.
        """
        # Create fresh weights for the number of parents found.
        weights = (np.log(num_parents + 0.5) -
                   np.log(np.arange(1, num_parents + 1)))
        total_weights = np.sum(weights)
        weights = weights / total_weights
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        # Dynamically update these strategy-specific parameters.
        cc = ((4 + mueff / solution_dim) /  # pylint: disable = unused-variable
              (solution_dim + 4 + 2 * mueff / solution_dim))
        cc_sep = (
            (1 + 1 / solution_dim + mueff / solution_dim) /
            (solution_dim**0.5 + 1 / solution_dim + 2 * mueff / solution_dim))

        cs = (mueff + 2) / (solution_dim + mueff + 5)

        c1 = 2 / ((solution_dim + 1.3)**2 + mueff)
        c1_sep = c1 * self._conedf(solution_dim, mueff, solution_dim)

        cmu = min(  # pylint: disable = unused-variable
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((solution_dim + 2)**2 + mueff),
        )
        # Instead of 0, pycma uses "rankmu_offset" here but also mentions it
        # barely affects performance. The rest of this code also doesn't use
        # rankmu_offset.
        cmu_sep = min(1 - c1_sep, self._cmudf(solution_dim, mueff, 0))

        return weights, mueff, cc_sep, cs, c1_sep, cmu_sep

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_mean(parents, weights):
        """Numba helper for calculating the new mean."""
        return np.sum(parents * np.expand_dims(weights, axis=1), axis=0)

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_weighted_ys(parents, old_mean, weights):
        """Calculates y's for use in rank-mu update."""
        ys = parents - np.expand_dims(old_mean, axis=0)
        return ys * np.expand_dims(weights, axis=1), ys

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_cov_update(cov, c1a, cmu, c1, pc, sigma, rank_mu_update, weights):
        """Calculates covariance matrix update."""
        rank_one_update = c1 * pc**2
        return (cov * (1 - c1a - cmu * np.sum(weights)) + rank_one_update * c1 +
                rank_mu_update * cmu / (sigma**2))

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
        self.current_eval += len(self._solutions[ranking_indices])

        if num_parents == 0:
            return

        parents = self._solutions[ranking_indices][:num_parents]

        weights, mueff, cc, cs, c1, cmu = self._calc_strat_params(
            self.solution_dim, num_parents)

        damps = (1 + 2 * max(
            0,
            np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
        ) + cs)

        # Recombination of the new mean.
        old_mean = self.mean
        self.mean = self._calc_mean(parents, weights)

        # Update the evolution path.
        y = self.mean - old_mean
        z = self.cov.invsqrt * y
        self.ps = ((1 - cs) * self.ps +
                   (np.sqrt(cs * (2 - cs) * mueff) / self.sigma) * z)
        left = (np.sum(np.square(self.ps)) / self.solution_dim /
                (1 - (1 - cs)**(2 * self.current_eval / self.batch_size)))
        right = 2 + 4. / (self.solution_dim + 1)
        hsig = 1 if left < right else 0

        self.pc = ((1 - cc) * self.pc + hsig * np.sqrt(cc *
                                                       (2 - cc) * mueff) * y)

        # Adapt the covariance matrix.
        weighted_ys, ys = self._calc_weighted_ys(parents, old_mean, weights)
        # Equivalent to calculating the outer product of each ys[i] with itself
        # and taking a weighted sum of the outer products.
        rank_mu_update = np.sum(weighted_ys * ys, axis=0)
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        self.cov.cov = self._calc_cov_update(self.cov.cov, c1a, cmu, c1,
                                             self.pc, self.sigma,
                                             rank_mu_update, weights)

        # Update sigma.
        cn, sum_square_ps = cs / damps, np.sum(np.square(self.ps))
        self.sigma *= np.exp(
            min(1,
                cn * (sum_square_ps / self.solution_dim - 1) / 2))
