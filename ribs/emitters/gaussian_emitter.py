import numpy as np


class GaussianEmitter:

    def __init__(self, sigma0, batch_size, archive):
        self.sigma0 = sigma0
        self.archive = archive
        self.batch_size = batch_size

    def ask(self):

        solution, objective_value = self.archive.get_random_elite()

        return solution + np.random.normal(
            scale=self.sigma0, size=(self.batch_size, len(solution)))

    def tell(self, solutions, objective_values, behavior_values):

        for sol, obj, beh in zip(solutions, objective_values, behavior_values):
            self.archive.add(sol, obj, beh)
