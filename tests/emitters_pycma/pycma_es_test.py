"""Tests for EvolutionStrategyEmitter."""
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.emitters.opt import PyCMAEvolutionStrategy

# pylint: disable = redefined-outer-name


def test_randomness():
    """If you create two separate ES's, they should not interfere with each
    other, and thus they should produce the same results. Good test to have
    since pycma by default uses the global np.random, so we should make sure
    that we are correctly injecting a generator."""
    es1 = PyCMAEvolutionStrategy(sigma0=0.1, solution_dim=10, seed=42)
    es2 = PyCMAEvolutionStrategy(sigma0=0.1, solution_dim=10, seed=42)
    es1.reset(np.zeros(10))
    es2.reset(np.zeros(10))
    assert np.all(es1.ask() == es2.ask())


def test_sphere():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive,
                                       x0=np.zeros(10),
                                       sigma0=1.0,
                                       es="pycma_es")

    # Try running with the negative sphere function for a few iterations.
    for _ in range(5):
        solution_batch = emitter.ask()
        objective_batch = -np.sum(np.square(solution_batch), axis=1)
        measures_batch = solution_batch[:, :2]
        add_info = archive.add(solution_batch, objective_batch, measures_batch)
        emitter.tell(solution_batch, objective_batch, measures_batch, add_info)
