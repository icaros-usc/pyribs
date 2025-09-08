"""Tests for EvolutionStrategyEmitter."""

import numpy as np
import pytest

from ribs.emitters import GeneticAlgorithmEmitter


def test_properties_are_correct(archive_fixture):
    archive, x0 = archive_fixture
    iso_sigma = 1
    line_sigma = 2
    batch_size = 2
    operator_kwargs = {"iso_sigma": iso_sigma, "line_sigma": line_sigma}

    emitter = GeneticAlgorithmEmitter(
        archive,
        x0=x0,
        batch_size=batch_size,
        operator="isoline",
        operator_kwargs=operator_kwargs,
    )

    assert np.all(emitter.x0 == x0)
    assert emitter.batch_size == batch_size


def test_initial_solutions_is_correct(archive_fixture):
    archive, _ = archive_fixture
    initial_solutions = [[0, 1, 2, 3], [-1, -2, -3, -4]]
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        operator="isoline",
        operator_kwargs={
            "iso_sigma": 0.1,
            "line_sigma": 0.2,
        },
        initial_solutions=initial_solutions,
    )

    assert np.all(emitter.ask() == initial_solutions)
    assert np.all(emitter.initial_solutions == initial_solutions)


def test_initial_solutions_shape(archive_fixture):
    archive, _ = archive_fixture

    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            # archive.solution_dim = 4, but these are 3D.
            initial_solutions=[[0, 0, 0], [1, 1, 1]],
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )


def test_neither_x0_nor_initial_solutions_provided(archive_fixture):
    archive, _ = archive_fixture
    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(archive, batch_size=36, operator="gaussian")


def test_both_x0_and_initial_solutions_provided(archive_fixture):
    archive, x0 = archive_fixture
    with pytest.raises(ValueError):
        GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            x0=x0,
            initial_solutions=[[0, 1, 2, 3], [-1, -2, -3, -4]],
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )


@pytest.mark.parametrize("bound_type", ["bounds", "lower_upper"])
def test_upper_bounds_enforced(archive_fixture, bound_type):
    archive, _ = archive_fixture
    if bound_type == "bounds":
        emitter = GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            x0=[2, 2, 2, 2],
            bounds=[(-1, 1)] * 4,
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )
    else:
        emitter = GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            x0=[2, 2, 2, 2],
            lower_bounds=[-1, -1, -1, -1],
            upper_bounds=[1, 1, 1, 1],
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )
    sols = emitter.ask()
    assert np.all(sols <= 1)


@pytest.mark.parametrize("bound_type", ["bounds", "lower_upper"])
def test_lower_bounds_enforced(archive_fixture, bound_type):
    archive, _ = archive_fixture
    if bound_type == "bounds":
        emitter = GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            x0=[2, 2, 2, 2],
            bounds=[(-1, 1)] * 4,
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )
    else:
        emitter = GeneticAlgorithmEmitter(
            archive,
            batch_size=36,
            x0=[2, 2, 2, 2],
            lower_bounds=[-1, -1, -1, -1],
            upper_bounds=[1, 1, 1, 1],
            operator="isoline",
            operator_kwargs={
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
            },
        )
    sols = emitter.ask()
    assert np.all(sols >= -1)


def test_degenerate_iso_gauss_emits_x0(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        x0=x0,
        operator="isoline",
        operator_kwargs={
            # Degenerate.
            "iso_sigma": 0.0,
            "line_sigma": 0.2,
        },
    )
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_iso_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        x0=x0,
        operator="isoline",
        operator_kwargs={
            # Degenerate.
            "iso_sigma": 0.0,
            "line_sigma": 0.2,
        },
    )
    archive.add_single(x0, 1, np.array([0, 0]))

    solutions = emitter.ask()

    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_iso_gauss_emits_along_line(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        x0=x0,
        operator="isoline",
        operator_kwargs={
            "iso_sigma": 0.0,
            "line_sigma": 0.2,
        },
    )
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
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        x0=x0,
        operator="gaussian",
        operator_kwargs={"sigma": 0.0},
    )
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(x0, axis=0)).all()


def test_degenerate_gauss_emits_parent(archive_fixture):
    archive, x0 = archive_fixture
    parent_sol = x0 * 5
    archive.add_single(parent_sol, 1, np.array([0, 0]))
    emitter = GeneticAlgorithmEmitter(
        archive,
        batch_size=36,
        x0=x0,
        operator="gaussian",
        operator_kwargs={"sigma": 0.0},
    )

    # All solutions should be generated "around" the single parent solution in
    # the archive.
    solutions = emitter.ask()
    assert (solutions == np.expand_dims(parent_sol, axis=0)).all()
