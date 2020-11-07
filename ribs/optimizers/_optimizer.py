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

        self.asked = False
        self.solutions = []

    def ask(self):
        if self.asked:
            raise RuntimeError("You have called ask() twice in a row.")

        self.asked = True
        self.solutions = []
        for emitter in self.emitters:
            self.solutions.append(emitter.ask())
        self.solutions = np.concatenate(self.solutions)
        return self.solutions

    def tell(self, objective_values, behavior_values):
        if not self.asked:
            raise RuntimeError("You have called tell() without ask().")

        self.asked = False

        # Convert user input into numpy arrays
        objective_values = np.array(objective_values)
        behavior_values = np.array(behavior_values)

        pos = 0
        for emitter in self.emitters:
            end = pos + emitter.batch_size
            emitter.tell(self.solutions[pos:end], objective_values[pos:end],
                         behavior_values[pos:end])
            pos = end
