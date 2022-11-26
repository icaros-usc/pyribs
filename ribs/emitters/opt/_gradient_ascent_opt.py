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
        theta0 (array-like): Initial solution. 1D array.
        lr (float): Learning rate for the update.
    """

    def reset(self, theta0):
        self._theta = np.copy(theta0)

    def step(self, gradient):
        step = self._lr * np.asarray(gradient)
        self._theta += step
