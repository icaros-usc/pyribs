"""Provides OperatorBase."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class OperatorBase(ABC):
    """Base class for operators.

    Operators output new solutions when passed parents.
    """

    @abstractmethod
    def ask(self, parents: ArrayLike) -> np.ndarray:
        """Operates on parents to generate new solutions.

        Args:
            parents: Array of solutions to be mutated. Typically passed in by an emitter
                after selection from an archive.

        Returns:
            ``(batch_size, solution_dim)`` array that contains ``batch_size`` mutated
            solutions.
        """
