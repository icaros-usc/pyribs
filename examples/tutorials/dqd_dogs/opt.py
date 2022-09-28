# Adapted from: https://github.com/hardmaru/estool/blob/master/es.py

import numpy as np

class AdamOpt:
    def __init__(self, theta0, stepsize, betas=(0.99, 0.999), epsilon=1e-8):
        self.epsilon = epsilon

        self.t = 0

        self.dim = len(theta0)
        self.stepsize = stepsize
        self.beta1 = betas[0]
        self.beta2 = betas[1]
    
        self.reset(theta0)

    def reset(self, theta0):
        self.theta = theta0
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, grad):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def step(self, grad):
        self.t += 1
        step = self._compute_step(grad)
        # ratio = np.linalg.norm(step) / (np.linalg.norm(self.theta) + self.epsilon)
        self.theta += step
