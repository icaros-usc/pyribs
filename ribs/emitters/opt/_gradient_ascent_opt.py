"""Provides GradientAscentOpt.

Adapted from:
https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/opt/_adam.py
https://github.com/hardmaru/estool/blob/master/es.py
"""

import numpy as np
from numpy.typing import ArrayLike

from ribs.emitters.opt._gradient_opt_base import GradientOptBase
from ribs.typing import Float


class GradientAscentOpt(GradientOptBase):
    """Vanilla gradient ascent.

    Args:
        theta0: Initial solution. 1D array.
        lr: Learning rate for the update.
    """

    def __init__(self, theta0: ArrayLike, lr: Float) -> None:
        self._lr = lr
        self._theta = None
        self.reset(theta0)

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    def reset(self, theta0: ArrayLike) -> None:
        self._theta = np.asarray(theta0, copy=True)

    def step(self, gradient: ArrayLike) -> None:
        step = self._lr * np.asarray(gradient)
        self._theta += step
