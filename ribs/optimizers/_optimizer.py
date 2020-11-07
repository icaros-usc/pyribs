"""Provides the Optimizer and corresponding OptimizerConfig."""
import numpy as np

from ribs.config import create_config


class OptimizerConfig:
    """Configuration for the Optimizer.

    Args:
        (none yet)
    """

    def __init__(self):
        pass


class Optimizer:

    def __init__(self, archive, emitters, config=None):
        self.config = create_config(config, OptimizerConfig)

        self.archive = archive
        self.num_iters = 0
        self.last_batch = None
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
