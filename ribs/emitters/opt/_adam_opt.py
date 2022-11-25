"""Provides AdamOpt.

Adapted from:
https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/opt/_adam.py
https://github.com/hardmaru/estool/blob/master/es.py
https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
"""
import numpy as np

from ribs.emitters.opt._gradient_opt_base import GradientOptBase


class AdamOpt(GradientOptBase):
    """Adam optimizer.

    Refer to `Kingma and Ba 2014 <https://arxiv.org/pdf/1412.6980.pdf>`_ for
    more information on hyperparameters.

    Args:
        theta0 (array-like): 1D initial solution.
        step_size (float): Scale for the gradient. Also known as alpha.
        beta1 (float): Exponential decay rate for the moment estimates.
        beta2 (float): Another exponential decay rate for the moment estimates.
        epsilon (float): Hyperparameter for numerical stability.
    """

    def __init__(self,
                 theta0,
                 step_size=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        self._epsilon = epsilon
        self._step_size = step_size
        self._beta1 = beta1
        self._beta2 = beta2

        self._theta = None
        self._m = None
        self._v = None
        self._t = None

        self.reset(theta0)

    @property
    def theta(self):
        return self._theta

    def reset(self, theta0):
        self._theta = np.copy(theta0)
        self._m = np.zeros_like(self._theta)
        self._v = np.zeros_like(self._theta)
        self._t = 0

    def step(self, gradient):
        gradient = np.asarray(gradient)
        self._t += 1
        a = (self._step_size * np.sqrt(1 - self._beta2**self._t) /
             (1 - self._beta1**self._t))
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = (self._beta2 * self._v + (1 - self._beta2) *
                   (gradient * gradient))
        step = a * self._m / (np.sqrt(self._v) + self._epsilon)
        self._theta += step
