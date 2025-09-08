"""Tests that should work for all emitters."""

import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import (
    EvolutionStrategyEmitter,
    GaussianEmitter,
    GradientArborescenceEmitter,
    IsoLineEmitter,
)


@pytest.fixture(
    params=[
        "GaussianEmitter",
        "IsoLineEmitter",
        "ImprovementEmitter",
        "RandomDirectionEmitter",
        "OptimizingEmitter",
    ]
)
def emitter_fixture(request, archive_fixture):
    """Creates an archive, emitter, and initial solution.

    Returns:
        Tuple of (archive, emitter, batch_size, x0).
    """
    emitter_type = request.param
    archive, x0 = archive_fixture
    batch_size = 3

    if emitter_type == "GaussianEmitter":
        emitter = GaussianEmitter(archive, sigma=5, x0=x0, batch_size=batch_size)
    elif emitter_type == "IsoLineEmitter":
        emitter = IsoLineEmitter(archive, x0=x0, batch_size=batch_size)
    elif emitter_type == "ImprovementEmitter":
        emitter = EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=5, ranker="2imp", batch_size=batch_size
        )
    elif emitter_type == "RandomDirectionEmitter":
        emitter = EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=5, ranker="2rd", batch_size=batch_size
        )
    elif emitter_type == "OptimizingEmitter":
        emitter = EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=5, ranker="2obj", batch_size=batch_size
        )
    else:
        raise NotImplementedError(f"Unknown emitter type {emitter_type}")

    return archive, emitter, batch_size, x0


#
# ask()
#


def test_ask_emits_correct_num_sols(emitter_fixture):
    _, emitter, batch_size, x0 = emitter_fixture
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_ask_emits_correct_num_sols_on_nonempty_archive(emitter_fixture):
    archive, emitter, batch_size, x0 = emitter_fixture
    archive.add_single(x0, 1, np.array([0, 0]))
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


@pytest.mark.parametrize("shape", [(), 4, (5,), (3, 3)])
def test_default_ask_dqd_has_correct_sol_shape(shape):
    archive = GridArchive(solution_dim=shape, dims=[10, 20], ranges=[(-1, 1), (-2, 2)])
    emitter = GaussianEmitter(archive, sigma=0.1, x0=np.ones(shape))

    expected_shape = (0, 4) if shape == 4 else (0, *shape)

    assert emitter.ask_dqd().shape == expected_shape


#
# tell()
#


@pytest.mark.parametrize(
    "emitter_type",
    ["GradientArborescenceEmitter", "EvolutionStrategyEmitter"],
    ids=["GAEmitter", "ESEmitter"],
)
@pytest.mark.parametrize(
    ("wrong_array", "offsets"),
    [
        ("solution_batch", [(0, 1), (1, 0)]),
        ("objective_batch", [(1,)]),
        ("measures_batch", [(0, 1), (1, 0)]),
        ("jacobian_batch", [(0, 0, 1), (0, 1, 0), (1, 0, 0)]),
        ("status_batch", [(1,)]),
        ("value_batch", [(1,)]),
    ],
)
def test_tell_arguments_incorrect_shape(emitter_type, wrong_array, offsets):
    """Test for incorrect shape of arguments in tell and tell_dqd.

    This test assumes that testing for every dimension of the input shapes
    individually covers the cases where some combination of the dimensions are
    wrong. For instance, if we have solution_batch.shape = (x, y), then testing
    for incorrect x and incorrect y individually is enough. We don't have to
    test for when both x and y are incorrect.
    """
    batch_size = 3
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])
    if emitter_type == "GradientArborescenceEmitter":
        emitter = GradientArborescenceEmitter(
            archive, x0=np.array([0]), sigma0=1.0, lr=0.1, batch_size=batch_size
        )
    elif emitter_type == "EvolutionStrategyEmitter":
        emitter = EvolutionStrategyEmitter(
            archive, x0=np.array([0]), sigma0=1.0, ranker="imp", batch_size=batch_size
        )
    else:
        raise RuntimeError

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim)
    )
    status_batch = np.ones(batch_size)
    value_batch = np.ones(batch_size)

    for offset in offsets:
        if wrong_array == "solution_batch":
            solution_batch = np.ones(
                (
                    batch_size + offset[0],
                    archive.solution_dim + offset[1],
                )
            )
        elif wrong_array == "objective_batch":
            objective_batch = np.ones(batch_size + offset[0])
        elif wrong_array == "measures_batch":
            measures_batch = np.ones(
                (
                    batch_size + offset[0],
                    archive.measure_dim + offset[1],
                )
            )
        elif wrong_array == "jacobian_batch":
            jacobian_batch = np.ones(
                (
                    batch_size + offset[0],
                    archive.measure_dim + 1 + offset[1],
                    archive.solution_dim + offset[2],
                )
            )
        elif wrong_array == "status_batch":
            status_batch = np.ones(batch_size + offset[0])
        elif wrong_array == "value_batch":
            value_batch = np.ones(batch_size + offset[0])

        # Only GradientArborescenceEmitter has tell_dqd method.
        if isinstance(emitter, GradientArborescenceEmitter):
            with pytest.raises(ValueError):
                emitter.tell_dqd(
                    solution_batch,
                    objective_batch,
                    measures_batch,
                    jacobian_batch,
                    {
                        "status": status_batch,
                        "value": value_batch,
                    },
                )

        if wrong_array == "jacobian_batch":
            # tell() does not use jacobian_batch paramter, so we skip calling
            # it when we are testing for incorrect jacobian_batch shape.
            return

        with pytest.raises(ValueError):
            # For GradientArborescenceEmitter, tell is called before tell_dqd,
            # but shape check exception should be thrown before tell complains
            # that tell_dqd is not called.
            emitter.tell(
                solution_batch,
                objective_batch,
                measures_batch,
                {"status": status_batch, "value": value_batch},
            )


#
# Bounds handling (only uses GaussianEmitter).
#


def test_default_bounds_correct(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, sigma=1, x0=x0)

    assert (emitter.lower_bounds == np.full(archive.solution_dim, -np.inf)).all()
    assert (emitter.upper_bounds == np.full(archive.solution_dim, np.inf)).all()


@pytest.mark.parametrize("bound_type", ["bounds", "lower_upper"])
def test_array_bound_correct(archive_fixture, bound_type):
    archive, x0 = archive_fixture
    lower_bounds = np.concatenate((-np.arange(len(x0) - 1), [-np.inf]))
    upper_bounds = np.concatenate((np.arange(len(x0) - 1), [np.inf]))

    if bound_type == "bounds":
        emitter = GaussianEmitter(
            archive,
            sigma=1,
            x0=x0,
            bounds=list(zip(lower_bounds, upper_bounds)),
        )
    else:
        emitter = GaussianEmitter(
            archive,
            sigma=1,
            x0=x0,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    assert (emitter.lower_bounds == lower_bounds).all()
    assert (emitter.upper_bounds == upper_bounds).all()


def test_multidim_array_bound_correct():
    archive = GridArchive(solution_dim=(5, 5), dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    lower_bounds = np.full((5, 5), -1)
    upper_bounds = np.full((5, 5), 1)
    emitter = GaussianEmitter(
        archive,
        sigma=1,
        x0=np.zeros((5, 5)),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    assert (emitter.lower_bounds == lower_bounds).all()
    assert (emitter.upper_bounds == upper_bounds).all()


@pytest.mark.parametrize("bound_type", ["bounds", "lower_upper"])
def test_long_array_bound_fails(archive_fixture, bound_type):
    archive, x0 = archive_fixture

    if bound_type == "bounds":
        with pytest.raises(
            ValueError,
            match="If it is an array-like, bounds must have length solution_dim",
        ):
            GaussianEmitter(
                archive,
                sigma=1,
                x0=x0,
                # More bounds than solution dims.
                bounds=[(-1, 1)] * (len(x0) + 1),
            )
    else:
        with pytest.raises(
            ValueError, match="Expected lower_bounds to be an array with shape .*"
        ):
            GaussianEmitter(
                archive,
                sigma=1,
                x0=x0,
                # More bounds than solution dims.
                lower_bounds=[-1] * (len(x0) + 1),
                upper_bounds=[1] * len(x0),
            )
        with pytest.raises(
            ValueError, match="Expected upper_bounds to be an array with shape .*"
        ):
            GaussianEmitter(
                archive,
                sigma=1,
                x0=x0,
                lower_bounds=[1] * len(x0),
                upper_bounds=[-1] * (len(x0) + 1),
            )


def test_wrong_bound_shape(archive_fixture):
    archive, x0 = archive_fixture

    with pytest.raises(
        ValueError, match="Expected .*_bounds to be an array with shape .*"
    ):
        GaussianEmitter(
            archive,
            sigma=1,
            x0=x0,
            lower_bounds=np.ones((10, 10)),
            upper_bounds=np.ones((10, 10)),
        )


@pytest.mark.parametrize(
    "emitter_type", ["GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter"]
)
def test_emitters_fail_when_x0_not_1d(emitter_type, archive_fixture):
    archive, _ = archive_fixture
    x0 = [[1], [1]]

    if emitter_type == "GaussianEmitter":
        with pytest.raises(ValueError):
            _ = GaussianEmitter(archive, sigma=5, x0=x0)
    elif emitter_type == "IsoLineEmitter":
        with pytest.raises(ValueError):
            _ = IsoLineEmitter(archive, x0=x0)
    elif emitter_type == "ImprovementEmitter":
        with pytest.raises(ValueError):
            _ = EvolutionStrategyEmitter(archive, x0=x0, sigma0=5, ranker="2imp")
