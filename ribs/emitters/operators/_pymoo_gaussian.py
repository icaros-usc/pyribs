"""Pymoo Guassian Mutator"""
import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from ribs.emitters.operators._operator_base import OperatorBase


class PymooGaussianOperator(OperatorBase):
    """Applies Pymoo's GaussianMutation Operator to Emitted Solutions"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        class_args = ['operator', 'lower_bounds', 'upper_bounds']

        if not all(arg in kwargs for arg in class_args):
            raise ValueError(
                "PymooGaussian Operator initialization arguments must be \
                provided.")

    def operate(self, kwargs):
        if ('parents' not in kwargs or 'bounds' not in kwargs or
                'n_var' not in kwargs):
            raise ValueError("Parents, bounds and n_var must be provided.")

        parents = kwargs['parents']
        n_var = kwargs['n_var']
        bounds = kwargs['bounds']

        problem = Problem(n_var, xl=bounds[0], xu=bounds[0], vtype=float)

        print("Parents ", parents)
        pop = Population.new(X=parents)
        solution = self.operator(problem, pop).get("X")
        return np.clip(solution, self.lower_bounds, self.upper_bounds)
