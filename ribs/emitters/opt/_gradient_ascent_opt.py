"""Provides GradientAscentOpt.

Adapted from:
https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/opt/_adam.py
https://github.com/hardmaru/estool/blob/master/es.py
"""
import numpy as np

from ribs.emitters.opt._gradient_opt_base import GradientOptBase


class GradientAscentOpt(GradientOptBase):
    """Vanilla gradient ascent.

    Args:
        theta0: Initial solution point.
        stepsize: Used to scale the gradient during the update.
    """

    def __init__(self, theta0, stepsize):
        self.dim = len(theta0)
        self.stepsize = stepsize

        self._theta = None

        self.reset(theta0)

    @property
    def theta(self):
        return self._theta

    def reset(self, theta0):
        self._theta = np.copy(theta0)

    def step(self, grad):
        step = self.stepsize * grad
        self._theta += step
