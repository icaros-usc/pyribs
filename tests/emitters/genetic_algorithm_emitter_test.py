"""Tests for EvolutionStrategyEmitter."""
import numpy as np
from pymoo.operators.mutation.gauss import GaussianMutation

from ribs.archives import GridArchive
from ribs.emitters import GeneticAlgorithmEmitter


def test_pymoo_gaussian_operator():
    x0 = np.array([1, 2, 3, 4])
    bounds = [(1, 4), (1, 4), (1, 4), (1, 4)]
    operator = GaussianMutation(sigma=0.1)
    archive = GridArchive(solution_dim=4,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GeneticAlgorithmEmitter(archive,
                                      operator=operator,
                                      x0=x0,
                                      batch_size=1,
                                      bounds=bounds,
                                      os="pymooGaussian")
    solution = emitter.ask()
    print("solution-------- ", solution)


def test_pymoo_iso_line_operator():
    pass


def test_pymoo_gradient_operator():
    pass
