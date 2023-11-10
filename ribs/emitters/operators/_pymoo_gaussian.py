"""Pymoo Guassian Mutator"""
import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from ribs.emitters.operators._operator_base import OperatorBase


class PymooGaussianOperator(OperatorBase):
    """Applies Pymoo's GaussianMutation Operator to Emitted Solutions"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        class_args = ['operator', 'lower_bounds', 'upper_bounds', 'bounds']

        if not all(arg in kwargs for arg in class_args):
            raise ValueError(
                "PymooGaussian Operator initialization arguments must be \
                provided.")

    def ask(self, **kwargs):
        if 'parents' not in kwargs:
            raise ValueError("Parents must be provided.")

        parents = kwargs['parents']

        problem = Problem(len(parents[0]),
                          xl=self.bounds[0],
                          xu=self.bounds[0],
                          vtype=float)

        pop = Population.new(X=parents)
        solution = self.operator(problem, pop).get("X")
        return np.clip(solution, self.lower_bounds, self.upper_bounds)
