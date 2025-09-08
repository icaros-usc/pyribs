"""Tests for the BayesianOptimizationEmitter."""

import numpy as np
import pytest

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import BayesianOptimizationEmitter


@pytest.fixture
def archive_fixture():
    """A low resolution archive to speed up BayesianOptimizationEmitter tests."""
    archive = GridArchive(solution_dim=4, dims=[2, 2], ranges=[(-1, 1), (-1, 1)])
    return archive


@pytest.fixture
def full_archive_emitter_fixture(archive_fixture):
    """Returns a BayesianOptimizationEmitter that has filled its archive to
    100% coverage."""
    rng = np.random.default_rng()

    emitter = BayesianOptimizationEmitter(
        archive=archive_fixture,
        lower_bounds=[-1, -1, -1, -1],
        upper_bounds=[1, 1, 1, 1],
        upscale_schedule=[[2, 2], [4, 4]],
        num_initial_samples=1,
        seed=0,
        batch_size=1,
    )

    md1, md2 = archive_fixture.dims
    all_measures = np.array(
        np.meshgrid(np.linspace(-1, 1, md1), np.linspace(-1, 1, md2))
    ).T.reshape(-1, 2)
    for solution, objective, measures in zip(
        rng.uniform(-1, 1, (archive_fixture.cells, 4)),
        np.full((100,), archive_fixture.cells),
        all_measures,
    ):
        emitter.ask()
        add_info = archive_fixture.add(
            solution[None, :], [objective], measures[None, :]
        )
        emitter.tell(solution[None, :], [objective], measures[None, :], add_info)

    assert len(emitter.archive) == archive_fixture.cells, (
        "BayesianOptimizationEmitter should have filled all "
        f"{archive_fixture.cells} archive cells by this point; actually filled "
        f"{len(emitter.archive)}."
    )

    return emitter


def test_wrong_archive_type():
    """BayesianOptimizationEmitter is currently only compatible with
    GridArchive. If it is initialized with a different archive type, it should
    raise NotImplementedError."""
    archive = CVTArchive(solution_dim=1, cells=100, ranges=[(-1, 1)])
    with pytest.raises(
        NotImplementedError,
        match="archive type CVTArchive not implemented for"
        " BayesianOptimizationEmitter. Expected GridArchive.",
    ):
        BayesianOptimizationEmitter(
            archive,  # ty: ignore[invalid-argument-type]
            lower_bounds=[-1],
            upper_bounds=[1],
            num_initial_samples=1,
        )


@pytest.mark.parametrize(
    "wrong_upscale_schedule",
    [[[3, 3], [4, 4]], [[2, 2], [1, 1]], [[2, 2, 2], [4, 4]]],
    ids=["starting_res_mismatch", "decrease_res", "wrong_dim"],
)
def test_invalid_upscale_schedule(archive_fixture, wrong_upscale_schedule):
    """Should throw a ValueError if an invalid upscale_schedule is used to
    initialize BayesianOptimizationEmitter. A valid upscale_schedule must
    satisfy the following conditions:
        1. Must be a 2D array where the second dim equals `archive.measure_dim`.
        2. The resolutions corresponding to each measure must be non-decreasing
            along axis 0.
        3. The first resolution within the schedule must equal
            :attr:`archive.dims`.

        Example of valid upscale_schedule:
        [
            [5, 5],
            [5, 10],
            [10, 10]
        ]

        Example of invalid upscale_schedule:
        [
            [5, 5],
            [5, 10],
            [10, 5]  <-  resolution for measure 2 decreases
        ]"""
    with pytest.raises(ValueError):
        BayesianOptimizationEmitter(
            archive=archive_fixture,
            lower_bounds=[-1, -1, -1, -1],
            upper_bounds=[1, 1, 1, 1],
            upscale_schedule=wrong_upscale_schedule,
            num_initial_samples=1,
            seed=0,
        )


def test_upscale(full_archive_emitter_fixture):
    """If an upcale_schedule is provided during initialization,
    BayesianOptimizationEmitter should return the next resolution on the
    upscale schedule after failing to improve archive coverage for multiple
    iterations. Note that the actual upscaling is handled by
    BayesianOptimizationScheduler to ensure all emitters upscale at the same
    time, so we only test that BayesianOptimizationEmitter returns the next
    resolution correctly in this test."""
    rng = np.random.default_rng()
    # With starting resolution [md1, md2], full_archive_emitter_fixture should
    # tolerate sqrt(md1*md2) tell() calls that do not improve archive coverage
    # before upscaling the archive.
    no_improve_tolerance = (
        np.ceil(np.sqrt(full_archive_emitter_fixture.archive.cells)).astype(np.int32)
        + 1
    )
    for solution, objective, measures in zip(
        rng.uniform(-1, 1, (no_improve_tolerance, 4)),
        np.full((no_improve_tolerance,), 100),
        np.zeros((no_improve_tolerance, 2)),
    ):
        full_archive_emitter_fixture.ask()
        add_info = full_archive_emitter_fixture.archive.add(
            solution[None, :], [objective], measures[None, :]
        )
        next_res = full_archive_emitter_fixture.tell(
            solution[None, :], [objective], measures[None, :], add_info
        )

    assert np.all(next_res == [4, 4]), (
        "Expected BayesianOptimizationEmitter to return the next resolution "
        f"{[4, 4]} on the upscale schedule; actually got {next_res}"
    )
