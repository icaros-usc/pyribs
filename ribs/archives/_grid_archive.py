from random import choice

import numpy as np


class GridArchive:

    def __init__(self, dims, ranges):
        self.dims = np.array(dims)
        self.grid = dict()

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])
        self.interval_size = self.upper_bounds - self.lower_bounds

    def _get_index(self, behavior_values):
        # epsilon = 1e-9 accounts for floating point precision errors that
        # happen from transforming the behavior values into grid coordinates.
        behavior_values = np.clip(behavior_values + 1e-9, self.lower_bounds,
                                  self.upper_bounds)
        index = ((behavior_values - self.lower_bounds) \
                / self.interval_size) * self.dims
        return tuple(index.astype(int))

    def is_empty(self):
        return not self.grid

    def add(self, solution, objective_value, behavior_values):
        index = self._get_index(behavior_values)
        print(behavior_values, index)

        if index not in self.grid or self.grid[index][1] < objective_value:
            self.grid[index] = (solution, objective_value)
            return True
        return False

    def get_random_elite(self):
        index = choice(list(self.grid))
        return self.grid[index]

    # index0, index1, ..., indexd-1, bv0, bv1, ..., bvd-1, f, solution
    def as_pandas(self):
        pass
