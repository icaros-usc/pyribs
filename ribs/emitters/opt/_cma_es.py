"""TODO."""
import numpy as np


class DecompMatrix:
    """Maintains a covariance matrix and its eigendecomposition.

    Args:
        dimension (int): Size of the (square) covariance matrix.
        dtype (str or data-type): Data type of the matrix, typically np.float32
            or np.float64.
    """

    def __init__(self, dimension, dtype):
        self.C = np.eye(dimension, dtype=dtype)
        self.eigenbasis = np.eye(dimension, dtype=dtype)
        self.eigenvalues = np.ones((dimension,), dtype=dtype)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=dtype)  # C^(-1/2)
        self.updated_eval = 0

    def update_eigensystem(self, current_eval, lazy_gap_evals):
        # TODO: uncomment this
        #  if current_eval <= self.updated_eval + lazy_gap_evals:
        #      return

        # Force symmetry.
        self.C = np.maximum(self.C, self.C.T)

        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.C)
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = (np.max(self.eigenvalues) /
                                 np.min(self.eigenvalues))

        self.invsqrt = (self.eigenbasis *
                        (1 / np.sqrt(self.eigenvalues))) @ self.eigenbasis.T
        # Force symmetry.
        self.invsqrt = np.maximum(self.invsqrt, self.invsqrt.T)

        self.updated_eval = current_eval


class CMAEvolutionStrategy:
    """TODO.

    Args:
        batch_size (int): If None, we calculate a default batch size based on
            solution_dim.
        weight_rule (str): "truncation" (positive weights only) or "active"
            (include negative weights)
    """

    def __init__(self, sigma0, batch_size, solution_dim, weight_rule, dtype):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype

        if weight_rule not in ["truncation", "active"]:
            raise ValueError(f"Invalid weight_rule {weight_rule}")
        self.weight_rule = weight_rule

        num_parents = batch_size // 2
        weights, mueff, cc, cs, c1, cmu = self._calc_strat_params(num_parents)
        self.lazy_gap_evals = (0.5 * self.solution_dim * self.batch_size *
                               (c1 + cmu)**-1 / self.solution_dim**2)

        # Strategy-specific params -> initialized in reset().
        self.current_eval = None
        self.mean = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.C = None

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
        self.C = DecompMatrix(self.solution_dim, self.dtype)

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            TODO
        Returns:
            TODO
        """
        if self.C.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2
        # parents).
        if (len(ranking_values) >= 2 and
                np.abs(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

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
        self.C.update_eigensystem(self.current_eval, self.lazy_gap_evals)
        solutions = np.empty((self.batch_size, self.solution_dim),
                             dtype=self.dtype)
        transform_mat = self.C.eigenbasis * np.sqrt(self.C.eigenvalues)

        for i in range(self.batch_size):
            # Resampling method for bound constraints -> break when solutions
            # are within bounds.
            while True:
                solutions[i] = (transform_mat @ np.random.normal(
                    0.0, self.sigma, self.solution_dim) + self.mean)

                if np.all(
                        np.logical_and(solutions[i] >= lower_bounds,
                                       solutions[i] <= upper_bounds)):
                    break

        return np.asarray(solutions)

    def _calc_strat_params(self, num_parents):
        # Create fresh weights for the number of parents found.
        if self.weight_rule == "truncation":
            weights = (np.log(num_parents + 0.5) -
                       np.log(np.arange(1, num_parents + 1)))
            total_weights = np.sum(weights)
            weights = weights / total_weights
            mueff = np.sum(weights)**2 / np.sum(weights**2)
        elif self.weight_rule == "active":
            weights = None

        # Dynamically update these strategy-specific parameters.
        cc = ((4 + mueff / self.solution_dim) /
              (self.solution_dim + 4 + 2 * mueff / self.solution_dim))
        cs = (mueff + 2) / (self.solution_dim + mueff + 5)
        c1 = 2 / ((self.solution_dim + 1.3)**2 + mueff)
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((self.solution_dim + 2)**2 + mueff),
        )
        return weights, mueff, cc, cs, c1, cmu

    def tell(self, solutions, num_parents):
        """Passes the solutions back to the optimizer.

        Args:
            solutions (np.ndarray): TODO
            num_parents (int): TODO
        """
        self.current_eval += len(solutions)

        if num_parents == 0:
            return

        parents = solutions[:num_parents]

        weights, mueff, cc, cs, c1, cmu = self._calc_strat_params(num_parents)

        damps = (1 + 2 * max(
            0,
            np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
        ) + cs)
        chiN = self.solution_dim**0.5 * (1 - 1 / (4 * self.solution_dim) + 1. /
                                         (21 * self.solution_dim**2))

        # Recombination of the new mean.
        old_mean = self.mean
        self.mean = np.sum(parents * weights[:, None], axis=0)

        # Update the evolution path.
        y = self.mean - old_mean
        z = np.matmul(self.C.invsqrt, y)
        self.ps = ((1 - cs) * self.ps +
                   (np.sqrt(cs * (2 - cs) * mueff) / self.sigma) * z)
        left = (np.sum(np.square(self.ps)) / self.solution_dim /
                (1 - (1 - cs)**(2 * self.current_eval / self.batch_size)))
        right = 2 + 4. / (self.solution_dim + 1)
        hsig = 1 if left < right else 0

        self.pc = ((1 - cc) * self.pc + hsig * np.sqrt(cc *
                                                       (2 - cc) * mueff) * y)

        # Adapt the covariance matrix
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        self.C.C *= (1 - c1a - cmu)
        self.C.C += c1 * np.outer(self.pc, self.pc)
        # TODO: batch this calculation.
        for k, w in enumerate(weights):
            dv = parents[k] - old_mean
            self.C.C += w * cmu * np.outer(dv, dv) / (self.sigma**2)

        # Update sigma.
        cn, sum_square_ps = cs / damps, np.sum(np.square(self.ps))
        self.sigma *= np.exp(
            min(1,
                cn * (sum_square_ps / self.solution_dim - 1) / 2))
