"""Provides OperatorBase."""
from abc import ABC, abstractmethod


class OperatorBase(ABC):
    """Base class for operators.

    Operators take in parents and output new solutions when their ask method
    is called. They can also be instantiated with any arguments.
    """

    @abstractmethod
    def ask(self, parents):
        """Operates on parents to generate new solutions.

        Args:
            parents (array-like): Array of solutions to be mutated. Typically
                passed in by an emitter after selection from an archive.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """
