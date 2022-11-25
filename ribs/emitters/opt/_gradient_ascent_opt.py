"""Provides GradientAscentOpt.

Adapted from:
https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/opt/_adam.py
https://github.com/hardmaru/estool/blob/master/es.py
"""
import numpy as np

from ribs.emitters.opt._gradient_opt_base import GradientOptBase


class GradientAscentOpt(GradientOptBase):
    """Vanilla gradient ascent."""

    def __init__(self, theta0, stepsize, epsilon=1e-8):
        self.epsilon = epsilon

        self.dim = len(theta0)
        self.stepsize = stepsize
        self.reset(theta0)

    def reset(self, theta0):
        self.theta = np.copy(theta0)

    def _compute_step(self, grad):
        return self.stepsize * grad

    def step(self, grad):
        step = self._compute_step(grad)
        self.theta += step
