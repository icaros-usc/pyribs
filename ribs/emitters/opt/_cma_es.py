"""Implementation of CMA-ES that can be used across various emitters.

Adapted from Nikolaus Hansen's pycma:
https://github.com/CMA-ES/pycma/blob/master/cma/purecma.py
"""
import numba as nb
import numpy as np


class DecompMatrix:
    """Maintains a covariance matrix and its eigendecomposition.

    CMA-ES requires the inverse square root of the covariance matrix in order to
    sample new solutions from a multivariate normal distribution. However,
    calculating the inverse square root is an O(n^3) operation because an
    eigendecomposition is involved. (n is the dimensionality of the search
    space). To amortize the operation to O(n^2) and avoid recomputing, this
    class maintains the inverse square root and waits several evals before
    recomputing the inverse square root.

    Args:
        dimension (int): Size of the (square) covariance matrix.
        dtype (str or data-type): Data type of the matrix, typically np.float32
            or np.float64.
    """

    def __init__(self, dimension, dtype):
        self.cov = np.eye(dimension, dtype=dtype)
        self.eigenbasis = np.eye(dimension, dtype=dtype)
        self.eigenvalues = np.ones((dimension,), dtype=dtype)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=dtype)  # C^(-1/2)

        # The last evaluation on which the eigensystem was updated.
        self.updated_eval = 0

    def update_eigensystem(self, current_eval, lazy_gap_evals):
        """Updates the covariance matrix if lazy_gap_evals have passed.

        We have attempted to use numba in this method, but since np.linalg.eigh
        is the bottleneck, and it is already implemented in BLAS or LAPACK,
        numba does not help much (and actually slows things down a bit).

        Args:
            current_eval (int): The number of solutions the optimizer has
                evaluated so far.
            lazy_gap_evals (int): The number of evaluations to wait between
                covariance matrix updates.
        """
        if current_eval <= self.updated_eval + lazy_gap_evals:
            return

        # Force symmetry.
        self.cov = np.maximum(self.cov, self.cov.T)

        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.cov)
        self.eigenvalues = self.eigenvalues.real
        self.eigenbasis = self.eigenbasis.real
        self.condition_number = (np.max(self.eigenvalues) /
                                 np.min(self.eigenvalues))
        self.invsqrt = (self.eigenbasis *
                        (1 / np.sqrt(self.eigenvalues))) @ self.eigenbasis.T

        # Force symmetry.
        self.invsqrt = np.maximum(self.invsqrt, self.invsqrt.T)

        self.updated_eval = current_eval


class CMAEvolutionStrategy:
    """CMA-ES optimizer for use with emitters.

    The basic usage is:
    - Initialize the optimizer and reset it.
    - Repeatedly:
      - Request new solutions with ask()
      - Rank the solutions in the emitter (better solutions come first) and pass
        them back with tell().
      - Use check_stop() to see if the optimizer has reached a stopping
        condition, and if so, call reset().

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        weight_rule (str): Method for generating weights. Either "truncation"
            (positive weights only) or "active" (include negative weights).
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
    """

    def __init__(self, sigma0, batch_size, solution_dim, weight_rule, seed,
                 dtype):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype

        if weight_rule not in ["truncation", "active"]:
            raise ValueError(f"Invalid weight_rule {weight_rule}")
        self.weight_rule = weight_rule

        # Calculate gap between covariance matrix updates.
        num_parents = self.batch_size // 2
        *_, c1, cmu = self._calc_strat_params(self.solution_dim, num_parents,
                                              self.weight_rule)
        self.lazy_gap_evals = (0.5 * self.solution_dim * self.batch_size *
                               (c1 + cmu)**-1 / self.solution_dim**2)

        # Strategy-specific params -> initialized in reset().
        self.current_eval = None
        self.mean = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.cov = None

        self._rng = np.random.default_rng(seed)

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
        self.cov = DecompMatrix(self.solution_dim, self.dtype)

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to tell().
        Returns:
            True if any of the stopping conditions are satisfied.
        """
        if self.cov.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(max(self.cov.eigenvalues))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2
        # parents).
        if (len(ranking_values) >= 2 and
                np.abs(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    @staticmethod
    @nb.jit(nopython=True)
    def _transform_and_check_sol(unscaled_params, transform_mat, mean,
                                 lower_bounds, upper_bounds):
        """Numba helper for transforming parameters to the solution space."""
        solutions = ((transform_mat @ unscaled_params.T).T +
                     np.expand_dims(mean, axis=0))
        out_of_bounds = np.logical_or(
            solutions < np.expand_dims(lower_bounds, axis=0),
            solutions > np.expand_dims(upper_bounds, axis=0),
        )
        return solutions, out_of_bounds

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
        self.cov.update_eigensystem(self.current_eval, self.lazy_gap_evals)
        solutions = np.empty((self.batch_size, self.solution_dim),
                             dtype=self.dtype)
        transform_mat = self.cov.eigenbasis * np.sqrt(self.cov.eigenvalues)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(self.batch_size)
        while len(remaining_indices) > 0:
            unscaled_params = self._rng.normal(
                0.0, self.sigma, (len(remaining_indices), self.solution_dim))
            new_solutions, out_of_bounds = self._transform_and_check_sol(
                unscaled_params, transform_mat, self.mean, lower_bounds,
                upper_bounds)
            solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each entry in each solution is
            # out of bounds).
            out_of_bounds_indices = np.where(np.any(out_of_bounds, axis=1))[0]
            remaining_indices = remaining_indices[out_of_bounds_indices]

        return np.asarray(solutions)

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_strat_params(solution_dim, num_parents, weight_rule):
        """Calculates weights, mueff, and learning rates for CMA-ES."""
        # Create fresh weights for the number of parents found.
        if weight_rule == "truncation":
            weights = (np.log(num_parents + 0.5) -
                       np.log(np.arange(1, num_parents + 1)))
            total_weights = np.sum(weights)
            weights = weights / total_weights
            mueff = np.sum(weights)**2 / np.sum(weights**2)
        elif weight_rule == "active":
            weights = None

        # Dynamically update these strategy-specific parameters.
        cc = ((4 + mueff / solution_dim) /
              (solution_dim + 4 + 2 * mueff / solution_dim))
        cs = (mueff + 2) / (solution_dim + mueff + 5)
        c1 = 2 / ((solution_dim + 1.3)**2 + mueff)
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((solution_dim + 2)**2 + mueff),
        )
        return weights, mueff, cc, cs, c1, cmu

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
    def _calc_cov_update(cov, c1a, cmu, c1, pc, sigma, rank_mu_update):
        """Calculates covariance matrix update."""
        rank_one_update = c1 * np.outer(pc, pc)
        return (cov * (1 - c1a - cmu) + rank_one_update * c1 +
                rank_mu_update * cmu / (sigma**2))

    def tell(self, solutions, num_parents):
        """Passes the solutions back to the optimizer.

        Note that while we use numba to optimize certain parts of this function
        (in particular the covariance update), we are more cautious about other
        parts because the code that uses numba is significantly harder to read
        and maintain.

        Args:
            solutions (np.ndarray): Array of ranked solutions. The user should
                have determined some way to rank the solutions, such as by
                objective value. It is important that _all_ of the solutions
                initially given in ask() are returned here.
            num_parents (int): Number of best solutions to select.
        """
        self.current_eval += len(solutions)

        if num_parents == 0:
            return

        parents = solutions[:num_parents]

        weights, mueff, cc, cs, c1, cmu = self._calc_strat_params(
            self.solution_dim, num_parents, self.weight_rule)

        damps = (1 + 2 * max(
            0,
            np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
        ) + cs)

        # Recombination of the new mean.
        old_mean = self.mean
        self.mean = self._calc_mean(parents, weights)

        # Update the evolution path.
        y = self.mean - old_mean
        z = np.matmul(self.cov.invsqrt, y)
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
        # and taking a weighted sum of the outer products. Unfortunately, numba
        # does not support einsum.
        rank_mu_update = np.einsum("ki,kj", weighted_ys, ys)
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        self.cov.cov = self._calc_cov_update(self.cov.cov, c1a, cmu, c1,
                                             self.pc, self.sigma,
                                             rank_mu_update)

        # Update sigma.
        cn, sum_square_ps = cs / damps, np.sum(np.square(self.ps))
        self.sigma *= np.exp(
            min(1,
                cn * (sum_square_ps / self.solution_dim - 1) / 2))
