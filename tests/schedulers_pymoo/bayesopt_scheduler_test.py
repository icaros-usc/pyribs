"""Tests for BayesianOptimizationScheduler."""

import pytest

from ribs.archives import GridArchive
from ribs.emitters import BayesianOptimizationEmitter, GaussianEmitter
from ribs.schedulers import BayesianOptimizationScheduler


def test_bayesopt_wrong_emitter_type():
    """When BayesianOptimizationScheduler is initialized with emitters that
    are not of type BayesianOptimizationEmitter, it should raise a TypeError.
    """
    archive = GridArchive(solution_dim=2, dims=[100, 100], ranges=[(-1, 1), (-1, 1)])
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]

    with pytest.raises(TypeError):
        BayesianOptimizationScheduler(archive, emitters)


@pytest.mark.parametrize(
    "wrong_upscale_schedules",
    [
        [None, [[2, 2]]],
        [[[2, 2], [4, 4]], [[2, 2], [4, 4], [8, 8]]],
        [[[2, 2], [4, 4]], [[2, 2], [8, 8]]],
    ],
    ids=["one_None", "num_res_mismatch", "res_mismatch"],
)
def test_bayesopt_mismatched_upscale_schedules(wrong_upscale_schedules):
    """When BayesianOptimizationScheduler is initialized with multiple
    emitters, all emitters must have exactly the same upscale schedule.
    If not, it should raise a ValueError.
    """
    archive = GridArchive(solution_dim=4, dims=[2, 2], ranges=[(-1, 1), (-1, 1)])
    (
        emitter1_upscale_schedule,
        emitter2_upscale_schedule,
    ) = wrong_upscale_schedules
    emitters = [
        BayesianOptimizationEmitter(
            archive=archive,
            bounds=[[-1, 1]] * 4,
            upscale_schedule=emitter1_upscale_schedule,
            num_initial_samples=1,
            seed=0,
        ),
        BayesianOptimizationEmitter(
            archive=archive,
            bounds=[[-1, 1]] * 4,
            upscale_schedule=emitter2_upscale_schedule,
            num_initial_samples=1,
            seed=1,
        ),
    ]

    with pytest.raises(ValueError):
        BayesianOptimizationScheduler(archive, emitters)
