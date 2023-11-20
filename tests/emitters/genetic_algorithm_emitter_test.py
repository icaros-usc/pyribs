"""Tests for EvolutionStrategyEmitter."""
import numpy as np
import pytest

from ribs.emitters import GeneticAlgorithmEmitter

# from pymoo.operators.mutation.gauss import GaussianMutation
# from ribs.archives import GridArchive


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    iso_sigma = 1
    line_sigma = 2
    batch_size = 2
    emitter = GeneticAlgorithmEmitter(archive,
                                      iso_sigma=iso_sigma,
                                      line_sigma=line_sigma,
                                      x0=x0,
                                      batch_size=batch_size,
                                      os="isoline")

    assert np.all(emitter.x0 == x0)
    assert emitter.iso_sigma == iso_sigma
    assert emitter.line_sigma == line_sigma
    assert emitter.batch_size == batch_size


def test_initial_solutions_is_correct(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    emitter = GeneticAlgorithmEmitter(archive,
                                      initial_solutions=initial_solutions,
                                      os="isoline")

    assert np.all(emitter.ask() == initial_solutions)
    assert np.all(emitter.initial_solutions == initial_solutions)


def test_initial_solutions_shape(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 0, 0], [1, 1, 1]]

    # archive.solution_dim = 4
    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(archive,
                                initial_solutions=initial_solutions,
                                os="isoline")


def test_neither_x0_nor_initial_solutions_provided(archive_fixture):
    archive, _ = archive_fixture
    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(archive)


def test_both_x0_and_initial_solutions_provided(archive_fixture):
    archive, x0 = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(archive,
                                x0=x0,
                                initial_solutions=initial_solutions,
                                os="isoline")


def test_upper_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      x0=[2, 2, 2, 2],
                                      iso_sigma=0,
                                      line_sigma=0,
                                      bounds=[(-1, 1)] * 4,
                                      os="isoline")
    sols = emitter.ask()
    assert np.all(sols <= 1)


def test_lower_bounds_enforced(archive_fixture):
    archive, _ = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      x0=[-2, -2, -2, -2],
                                      iso_sigma=0,
                                      line_sigma=0,
                                      bounds=[(-1, 1)] * 4,
                                      os="isoline")
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_iso_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      x0=x0,
                                      iso_sigma=0,
                                      batch_size=2,
                                      os="isoline")
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_iso_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      x0=x0,
                                      iso_sigma=0,
                                      batch_size=2,
                                      os="isoline")
    archive.add_single(x0, 1, np.array([0, 0]))

    solutions = emitter.ask()

    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_iso_gauss_emits_along_line(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      x0=x0,
                                      iso_sigma=0,
                                      batch_size=100,
                                      os="isoline")
    archive.add_single(np.array([0, 0, 0, 0]), 1, np.array([0, 0]))
    archive.add_single(np.array([10, 0, 0, 0]), 1, np.array([1, 1]))

    solutions = emitter.ask()

    # All solutions should either come from a degenerate distribution around
    # [0,0,0,0], a degenerate distribution around [10,0,0,0], or the "iso line
    # distribution" between [0,0,0,0] and [10,0,0,0] (i.e. a line between those
    # two points). By having a large batch size, we should be able to cover all
    # cases. In any case, this assertion should hold for all solutions
    # generated.
    assert (solutions[:, 1:] == 0).all()


def test_degenerate_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(archive,
                                      sigma=0,
                                      x0=x0,
                                      batch_size=2,
                                      os="gaussian")
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    parent_sol = x0 * 5
    archive.add_single(parent_sol, 1, np.array([0, 0]))
    emitter = GeneticAlgorithmEmitter(archive,
                                      sigma=0,
                                      x0=x0,
                                      batch_size=2,
                                      os="gaussian")

    # All solutions should be generated "around" the single parent solution in
    # the archive.
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(parent_sol, axis=0)).all()


# def test_pymoo_gaussian_operator_error_free():
#     x0 = np.array([1, 2, 3, 4])
#     bounds = [(1, 4), (1, 4), (1, 4), (1, 4)]
#     operator = GaussianMutation(sigma=0.1)
#     archive = GridArchive(solution_dim=4,
#                           dims=[20, 20],
#                           ranges=[(-1.0, 1.0)] * 2)
#     emitter = GeneticAlgorithmEmitter(archive,
#                                       operator=operator,
#                                       x0=x0,
#                                       batch_size=1,
#                                       bounds=bounds,
#                                       os="pymooGaussian")
#     solution = emitter.ask()
#     assert len(solution[0]) == len(x0)
