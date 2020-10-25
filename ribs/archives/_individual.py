"""Provides an Individual class that represents archive entries."""

from dataclasses import dataclass

import numpy as np


def _ndarray_eq(a1, a2):
    """Performs a stricter check of whether two numpy arrays are equal."""
    return (isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray) and
            a1.shape == a2.shape and np.all(a1 == a2))


@dataclass(eq=False)
class Individual:
    """Holds data of a single archive entry.

    Attributes:
        objective_value (float): The evaluated performance of this individual's
            solution.
        behavior_values (np.ndarray): An array with the individual's coordinates
            in behavior space.
        solution (np.ndarray): An array with the individual's solution.
    """
    objective_value: float
    behavior_values: np.ndarray
    solution: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, Individual):
            return NotImplemented
        return (self.objective_value == other.objective_value and
                _ndarray_eq(self.behavior_values, other.behavior_values) and
                _ndarray_eq(self.solution, other.solution))
