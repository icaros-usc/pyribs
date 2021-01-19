"""TODO."""

import numpy as np


class DecompMatrix:

    def __init__(self, dimension):
        self.C = np.eye(dimension, dtype=np.float_)
        self.eigenbasis = np.eye(dimension, dtype=np.float_)
        self.eigenvalues = np.ones((dimension,), dtype=np.float_)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=np.float_)

    def update_eigensystem(self):
        for i in range(len(self.C)):
            for j in range(i):
                self.C[i, j] = self.C[j, i]

        self.eigenvalues, self.eigenbasis = np.linalg.eig(self.C)
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        for i in range(len(self.C)):
            for j in range(i + 1):
                self.invsqrt[i, j] = self.invsqrt[j, i] = sum(
                    self.eigenbasis[i, k] * self.eigenbasis[j, k] /
                    self.eigenvalues[k]**0.5 for k in range(len(self.C)))


class CMAEvolutionStrategy:
    """TODO."""

    def __init__(self, sigma0, batch_size, solution_dim):
        self.batch_size = batch_size
        self.sigma0 = self.sigma0
        self.solution_dim = solution_dim
        self.dtype = np.float32  # TODO

        # Strategy-specific params -> initialized in reset().
        self.mean = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.C = None

        self.reset()

    def reset(self, x0):
        """TODO."""
        self.sigma = self.sigma0
        self.mean = x0

        #  if len(self.feature_map.elite_map) == 0:
        #      self.mean = np.zeros(self.solution_dim)
        #  else:
        #      self.mean = self.feature_map.get_random_elite().param_vector

        # Setup evolution path variables
        self.pc = np.zeros((self.solution_dim,), dtype=self.dtype)
        self.ps = np.zeros((self.solution_dim,), dtype=self.dtype)

        # Setup the covariance matrix
        self.C = DecompMatrix(self.solution_dim)

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * np.sqrt(max(self.C.eigenvalues))

        if area < 1e-11:
            return True
        if len(parents[0].fitness) > 1:
            if abs(parents[0].fitness[-1] - parents[-1].fitness[-1]) < 1e-12:
                return True
        else:
            if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
                return True

        return False

    #  def is_blocking(self):
    #      return self.individuals_disbatched > self.batch_size * 1.1

    def ask(self, lower_bounds, upper_bounds):
        """TODO."""
        solutions = np.empty((self.batch_size, self.solution_dim),
                             dtype=self.dtype)
        transform_mat = self.C.eigenbasis * np.sqrt(self.C.eigenvalues)
        for i in range(self.batch_size):
            # Resampling method for bound constraints
            while True:
                solutions[i] = (transform_mat @ np.random.normal(
                    0.0, self.sigma, self.solution_dim) + self.mean)

                if np.all(
                        np.logical_and(solutions[i] >= lower_bounds,
                                       solutions[i] <= upper_bounds)):
                    break

        return np.asarray(solutions)

    def tell(self, solutions, num_parents):
        """TODO.

        solutions -> (batch_size, solution_dim)
        """
        parents = solutions[:num_parents]

        # Create fresh weights for the number of parents found.
        weights = (np.log(num_parents + 0.5) -
                   np.log(np.arange(1, num_parents + 1)))
        total_weights = np.sum(weights)
        weights = weights / total_weights

        self.mean = np.sum(parents * weights[:, None], axis=0)  # solution_dim

        #  self.population.append(ind)
        #  self.individuals_evaluated += 1
        #  if self.feature_map.add(ind):
        #      self.parents.append(ind)
        #  if ind.generation != self.generation:
        #      return

        #  if len(self.population) < self.batch_size:
        #      return

        #  # Only filter by this generation
        #  num_parents = len(self.parents)
        #  needs_restart = num_parents == 0

        #  # Only update if there are parents
        #  if num_parents > 0:
        #      parents = sorted(self.parents, key=lambda x: x.delta)[::-1]

        #      # Create fresh weights for the number of parents found.
        #      weights = [math.log(num_parents + 0.5) \
        #              - math.log(i+1) for i in range(num_parents)]
        #      total_weights = sum(weights)
        #      weights = np.array([w / total_weights for w in weights])

        #      # Dynamically update these parameters
        #      mueff = sum(weights)**2 / sum(weights**2)
        #      cc = (4 + mueff / self.solution_dim) / (self.solution_dim + 4 +
        #                                            2 * mueff / self.solution_dim)
        #      cs = (mueff + 2) / (self.solution_dim + mueff + 5)
        #      c1 = 2 / ((self.solution_dim + 1.3)**2 + mueff)
        #      cmu = min(
        #          1 - c1, 2 * (mueff - 2 + 1 / mueff) /
        #          ((self.solution_dim + 2)**2 + mueff))
        #      damps = 1 + 2 * max(
        #          0,
        #          math.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1) + cs
        #      chiN = self.solution_dim**0.5 * (1 - 1 / (4 * self.solution_dim) + 1. /
        #                                     (21 * self.solution_dim**2))

        #      # Recombination of the new mean
        #      old_mean = self.mean
        #      self.mean = sum(
        #          ind.param_vector * w for ind, w in zip(parents, weights))

        #      # Update the evolution path
        #      y = self.mean - old_mean
        #      z = np.matmul(self.C.invsqrt, y)
        #      self.ps = (1-cs) * self.ps +\
        #          (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
        #      left = sum(x**2 for x in self.ps) / self.solution_dim \
        #          / (1-(1-cs)**(2*self.individuals_evaluated / self.batch_size))
        #      right = 2 + 4. / (self.solution_dim + 1)
        #      hsig = 1 if left < right else 0

        #      self.pc = (1-cc) * self.pc + \
        #          hsig * math.sqrt(cc*(2-cc)*mueff) * y

        #      # Adapt the covariance matrix
        #      c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        #      self.C.C *= (1 - c1a - cmu)
        #      self.C.C += c1 * np.outer(self.pc, self.pc)
        #      for k, w in enumerate(weights):
        #          dv = parents[k].param_vector - old_mean
        #          self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power**
        #                                                    2)

        #      # Update the covariance matrix decomposition and inverse
        #      if self.check_stop(parents):
        #          needs_restart = True
        #      else:
        #          self.C.update_eigensystem()

        #      # Update sigma
        #      cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
        #      self.mutation_power *= math.exp(
        #          min(1,
        #              cn * (sum_square_ps / self.solution_dim - 1) / 2))

        #  if needs_restart:
        #      self.reset()

        #  # Reset the population
        #  self.individuals_disbatched = 0
        #  self.generation += 1
        #  self.population.clear()
        #  self.parents.clear()
