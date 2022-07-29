"""Provides DQDEmitterBase."""
from abc import abstractmethod

import numpy as np

from ribs.emitters import EmitterBase


class DQDEmitterBase(EmitterBase):
    """Base class for DQD emitters.

    In addition to being an instance of EmitterBase, DQD emitters also implement
    :meth:`ask_dqd` and :meth:`tell_dqd`. These two functions should be used to
    communicate gradient information to the emitters. Generally, such as
    GradientAborescenceEmitter, these functions should be called, in a similar
    fashion to :meth:`ask` and :meth:`tell`, before calling :meth:`ask`.
    """

    @abstractmethod
    def ask_dqd(self):
        """Samples a new solution from the gradient optimizer.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            a new solution to evalute.
        """

    @abstractmethod
    def tell_dqd(self, jacobian):
        """Gives the emitter results from evaluating the gradient of
        the solutions.

        Args:
            jacobian (numpy.ndarray): Jacobian matrix of the solutions
                obtained from :meth:`ask_dqd`.
        """
