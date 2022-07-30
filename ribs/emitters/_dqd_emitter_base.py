"""Provides DQDEmitterBase."""
from abc import abstractmethod

from ribs.emitters._emitter_base import EmitterBase


class DQDEmitterBase(EmitterBase):
    """Base class for differentiable quality diversity (DQD) emitters.

    This class is a special instance of :class:`EmitterBase` which implements
    :meth:`ask_dqd` and :meth:`tell_dqd`. These two functions should be used to
    communicate gradient information to the emitters. The ask and tell functions
    should now be called in this order: :meth:`ask_dqd`, :meth:`tell_dqd`,
    :meth:`ask`, :meth:`tell`.
    """

    @abstractmethod
    def ask_dqd(self):
        """Generates a ``(batch_size, solution_dim)`` array of solutions for
        which gradient information must be computed."""

    @abstractmethod
    def tell_dqd(self, jacobian):
        """Gives the emitter results from evaluating the gradient of the
        solutions.

        See :meth:`tell` for the definitions of the remaining arguments.

        Args:
            jacobian_batch (numpy.ndarray): ``(batch_size, 1 + measure_dim,
                solution_dim)`` array consisting of Jacobian matrices of the
                solutions obtained from :meth:`ask_dqd`. Each matrix should
                consist of the objective gradient of the solution followed by
                the measure gradients.
        """
