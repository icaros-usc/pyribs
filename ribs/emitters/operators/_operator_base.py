"""Operator Base Class"""
from abc import ABC, abstractmethod


class OperatorBase(ABC):
    """Operator interface for GeneticAlgorithmEmitter (GAE). User passes
        operator string to GAE which maps to appropriate operator class.

        Provides ask function which mutates emitter-passed solutions.
        """

    @abstractmethod
    def __init__(self):
        """Init"""

    @abstractmethod
    def ask(self):
        """Mutates solutions provided with class-defined mutation

         Args:
            parents (np-array): (batch_size, :attr:`solution_dim`)
                array of solutions selected by emitter

        Returns:
            ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` mutated solutions
        """
