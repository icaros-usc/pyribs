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

    def __init__(self, theta0, step_size, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.epsilon = epsilon

        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2

        self.t = 0
        self.m = None
        self.v = None
        self._theta = None

        self.reset(theta0)

    @property
    def theta(self):
        return self._theta

    def reset(self, theta0):
        self._theta = np.copy(theta0)
        self.m = np.zeros_like(self._theta)
        self.v = np.zeros_like(self._theta)
        self.t = 0

    def step(self, gradient):
        gradient = np.asarray(gradient)
        self.t += 1
        a = self.step_size * np.sqrt(1 - self.beta2**self.t) / (
            1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        step = a * self.m / (np.sqrt(self.v) + self.epsilon)
        self._theta += step
