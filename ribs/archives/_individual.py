"""Provides an Individual class that represents archive entries."""

from collections import namedtuple

#: Holds data of a single archive entry.
#:
#: Attributes:
#:     objective_value (float): The evaluated performance of this individual's
#:         solution.
#:     behavior_values (np.ndarray): An array with the individual's coordinates
#:         in behavior space.
#:     solution (np.ndarray): An array with the individual's solution.
Individual = namedtuple("Individual", [
    "objective_value",
    "behavior_values",
    "solution",
])
