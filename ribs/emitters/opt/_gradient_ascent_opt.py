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
        step_size: Used to scale the gradient during the update.
    """

    def __init__(self, theta0, step_size):
        self.dim = len(theta0)
        self.step_size = step_size

        self._theta = None

        self.reset(theta0)

    @property
    def theta(self):
        return self._theta

    def reset(self, theta0):
        self._theta = np.copy(theta0)

    def step(self, gradient):
        step = self.step_size * gradient
        self._theta += step
