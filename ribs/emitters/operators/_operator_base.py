"""Provides OperatorBase."""

from abc import ABC, abstractmethod


class OperatorBase(ABC):
    """Base class for operators.

    Operators output new solutions when passed parents.
    """

    @abstractmethod
    def ask(self, parents):
        """Operates on parents to generate new solutions.

        Args:
            parents (array-like): Array of solutions to be mutated. Typically passed in
                by an emitter after selection from an archive.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """
