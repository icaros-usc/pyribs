"""Pymoo Guassian Mutator"""
import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from ribs.emitters.operators._operator_base import OperatorBase


class PymooGaussianOperator(OperatorBase):
    """Applies Pymoo's GaussianMutation Operator to Emitted Solutions"""

    def __init__(self, operator, lower_bounds, upper_bounds):
        self._operator = operator
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def operate(self, parents, bounds, n_var=4):
        problem = Problem(n_var, xl=bounds[0], xu=bounds[0], vtype=float)

        print("Parents ", parents)
        pop = Population.new(X=parents)
        solution = self._operator(problem, pop).get("X")
        return np.clip(solution, self._lower_bounds, self._upper_bounds)
