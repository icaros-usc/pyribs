from ribs.archives import GridArchive
import numpy as np


class Optimizer:

    def __init__(self, x0, sigma0, batch_size):
        self.archive = GridArchive((100, 100), [(-4, 4), (-4, 4)])
        self.x0 = np.array(x0)
        self.sigma0 = sigma0
        self.batch_size = batch_size
        self.num_iters = 0
        self.last_batch = None

    def ask(self):
        self.num_iters += 1

        if self.num_iters == 1:
            return np.random.normal(loc=self.x0,
                                    scale=self.sigma0,
                                    size=(self.batch_size, len(self.x0)))
        else:
            return np.random.normal(loc=self.x0,
                                    scale=self.sigma0,
                                    size=(self.batch_size, len(self.x0)))

    def tell(self, solutions, objective_values, behavior_values):

        # Convert user input into numpy arrays
        objective_values = np.array(objective_values)
        behavior_values = np.array(behavior_values)

        for sol, obj, beh in zip(solutions, objective_values, behavior_values):
            self.archive.add(sol, obj, beh)
