import numpy as np

from ribs.config import DEFAULT_CONFIG
from ribs.emitters import GaussianEmitter


class Optimizer:

    def __init__(self, x0, sigma0, archive, config=DEFAULT_CONFIG):
        self.archive = archive
        self.x0 = np.array(x0)
        self.sigma0 = sigma0
        self.num_iters = 0
        self.last_batch = None

        self.emitters = [
            GaussianEmitter(self.x0, self.sigma0, self.archive, config)
            for _ in range(1)
        ]

    def ask(self):

        solutions = []
        for emitter in self.emitters:
            solutions.append(emitter.ask())
        return np.concatenate(solutions)

    def tell(self, solutions, objective_values, behavior_values):

        # Convert user input into numpy arrays
        objective_values = np.array(objective_values)
        behavior_values = np.array(behavior_values)

        pos = 0
        for emitter in self.emitters:
            end = pos + emitter.batch_size
            emitter.tell(solutions[pos:end], objective_values[pos:end],
                         behavior_values[pos:end])
            pos = end
