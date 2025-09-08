"""Provides AdamOpt.

Adapted from:
https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/opt/_adam.py
https://github.com/hardmaru/estool/blob/master/es.py
https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
"""

import numpy as np
from numpy.typing import ArrayLike

from ribs.emitters.opt._gradient_opt_base import GradientOptBase
from ribs.typing import Float


class AdamOpt(GradientOptBase):
    """Adam optimizer.

    Refer to `Kingma and Ba 2014 <https://arxiv.org/pdf/1412.6980.pdf>`_ for more
    information on hyperparameters.

    Args:
        theta0: Initial solution. 1D array.
        lr: Learning rate for the update.
        beta1: Exponential decay rate for the moment estimates.
        beta2: Another exponential decay rate for the moment estimates.
        epsilon: Hyperparameter for numerical stability.
        l2_coeff: Coefficient for L2 regularization. Note this is **not** the same as
            "weight decay" -- see `this blog post
            <https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html>_` and
            `Loshchilov and Hutler 2019 <https://arxiv.org/abs/1711.05101>_` for more
            info.
    """

    def __init__(
        self,
        theta0: ArrayLike,
        lr: Float = 0.001,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        l2_coeff: Float = 0.0,
    ):
        self._m = None
        self._v = None
        self._t = None

        self._epsilon = epsilon
        self._beta1 = beta1
        self._beta2 = beta2
        self._l2_coeff = l2_coeff

        self._lr = lr
        self._theta = None
        self.reset(theta0)

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    def reset(self, theta0: ArrayLike) -> None:
        self._theta = np.asarray(theta0, copy=True)
        self._m = np.zeros_like(self._theta)
        self._v = np.zeros_like(self._theta)
        self._t = 0

    def step(self, gradient: ArrayLike) -> None:
        # Invert gradient since we seek to maximize -- see pseudocode here:
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        gradient = -np.asarray(gradient)

        # L2 regularization (not weight decay).
        gradient += self._l2_coeff * self._theta

        self._t += 1

        a = self._lr * np.sqrt(1 - self._beta2**self._t) / (1 - self._beta1**self._t)
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = self._beta2 * self._v + (1 - self._beta2) * (gradient * gradient)
        step = -a * self._m / (np.sqrt(self._v) + self._epsilon)
        self._theta += step
