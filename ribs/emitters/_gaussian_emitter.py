import numpy as np

from ribs.config import DEFAULT_CONFIG


class GaussianEmitter:

    def __init__(self, x0, sigma0, archive, config=DEFAULT_CONFIG):
        self.x0 = x0
        self.sigma0 = sigma0
        self.archive = archive
        self.batch_size = config["batch_size"]

        self._rng = np.random.default_rng(config["seed"])

    def ask(self):

        if self.archive.is_empty():
            # Use x0 only on the first iteration in the Gaussian Emitter.
            parents = np.expand_dims(self.x0, axis=0)
        else:
            parents = [
                self.archive.get_random_elite().solution
                for _ in range(self.batch_size)
            ]

        return parents + self._rng.normal(scale=self.sigma0,
                                          size=(self.batch_size, len(self.x0)))

    def tell(self, solutions, objective_values, behavior_values):

        for sol, obj, beh in zip(solutions, objective_values, behavior_values):
            self.archive.add(sol, obj, beh)
