"""Provides the Optimizer and corresponding OptimizerConfig."""
from collections import namedtuple

import numpy as np

from ribs.config import create_config
from ribs.emitters import GaussianEmitter

#: Configuration for the Optimizer.
#:
#: Attributes:
OptimizerConfig = namedtuple("OptimizerConfig", [])
OptimizerConfig.__new__.__defaults__ = ()


class Optimizer:

    def __init__(self, x0, sigma0, archive, emitters=None, config=None):
        self.config = create_config(config, OptimizerConfig)

        self.archive = archive
        self.x0 = np.array(x0)
        self.sigma0 = sigma0
        self.num_iters = 0
        self.last_batch = None

        if emitters is None:
            self.emitters = [
                GaussianEmitter(self.x0, self.sigma0, self.archive)
                for _ in range(1)
            ]
        else:
            self.emitters = emitters

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
