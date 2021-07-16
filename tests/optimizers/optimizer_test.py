"""Tests for the Optimizer."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import EmitterBase, GaussianEmitter
from ribs.optimizers import Optimizer

# pylint: disable = redefined-outer-name


@pytest.fixture
def optimizer_fixture():
    """Returns an Optimizer with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=num_solutions)
    ]
    return Optimizer(archive, emitters), solution_dim, num_solutions


def test_init_fails_with_no_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1)] * 5

    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        # Emits 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0], 1),
        # Mismatch -- emits 3D solutions rather than 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0, 0.0], 1),
    ]
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_ask_returns_correct_solution_shape(optimizer_fixture):
    optimizer, solution_dim, num_solutions = optimizer_fixture
    solutions = optimizer.ask()
    assert solutions.shape == (num_solutions, solution_dim)


def test_ask_fails_when_called_twice(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.ask()
        optimizer.ask()


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_into_archive(optimizer_fixture, tell_metadata):
    optimizer, _, num_solutions = optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = ([f"metadata_{i}" for i in range(num_solutions)]
                if tell_metadata else None)
    expected_metadata = metadata if tell_metadata else [None] * num_solutions

    # We pass in 4 solutions with unique behavior values, so all should go into
    # the archive.
    optimizer.tell(
        objective_values=np.ones(num_solutions),
        behavior_values=behavior_values,
        metadata=metadata,
    )

    # Note: This assumes elite data is in order of insertion, which may change
    # in the future.
    assert len(optimizer.archive) == num_solutions
    assert (optimizer.archive.behavior_values == behavior_values).all()
    assert (optimizer.archive.objective_values == np.ones(num_solutions)).all()
    assert (optimizer.archive.metadata == expected_metadata).all()


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_with_multiple_emitters(tell_metadata):
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1),
        GaussianEmitter(archive, [0.5, 0.5], 1, batch_size=2),
        GaussianEmitter(archive, [-0.5, -0.5], 1, batch_size=3),
    ]
    optimizer = Optimizer(archive, emitters)

    _ = optimizer.ask()
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                       [0.0, 0.0], [0.0, 1.0]]
    metadata = [f"metadata_{i}" for i in range(6)] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * 6

    # The sum of all the emitters' batch sizes is 6.
    optimizer.tell(
        objective_values=np.ones(6),
        behavior_values=behavior_values,
        metadata=metadata,
    )

    # Note: This assumes elite data is in order of insertion, which may change
    # in the future.
    assert len(optimizer.archive) == 6
    assert (optimizer.archive.behavior_values == behavior_values).all()
    assert (optimizer.archive.objective_values == np.ones(6)).all()
    assert (optimizer.archive.metadata == expected_metadata).all()


def test_tell_fails_when_ask_not_called(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)


@pytest.fixture
def kwargs_fixture():
    """Fixture for testing emitter_kwargs in the optimizer."""

    class KwargsEmitter(EmitterBase):
        """Emitter which takes in kwargs in its ask() and tell() methods.

        ask() and tell() simply set self.arg to be the value of arg.
        """

        def __init__(self, archive):
            EmitterBase.__init__(self, archive, 3, None)
            self.arg = None

        def ask(self, arg=None):
            self.arg = arg
            return []

        def tell(self,
                 solutions,
                 objective_values,
                 behavior_values,
                 metadata=None,
                 arg=None):
            self.arg = arg

    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [KwargsEmitter(archive) for _ in range(3)]
    return emitters, Optimizer(archive, emitters)


def test_ask_with_no_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask(emitter_kwargs=None)
    for e in emitters:
        assert e.arg is None


def test_ask_with_dict_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask(emitter_kwargs={"arg": 42})
    for e in emitters:
        assert e.arg == 42


def test_ask_with_list_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask(emitter_kwargs=[{"arg": 1}, {"arg": 2}, {"arg": 3}])
    for e, val in zip(emitters, [1, 2, 3]):
        assert e.arg == val


def test_ask_with_wrong_num_emitter_kwargs(kwargs_fixture):
    _, optimizer = kwargs_fixture
    with pytest.raises(ValueError):
        # There are 3 emitters but only 2 dicts of kwargs here.
        optimizer.ask(emitter_kwargs=[{"arg": 1}, {"arg": 2}])


def test_tell_with_no_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask()
    optimizer.tell([], [], [], emitter_kwargs=None)
    for e in emitters:
        assert e.arg is None


def test_tell_with_dict_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask()
    optimizer.tell([], [], [], emitter_kwargs={"arg": 42})
    for e in emitters:
        assert e.arg == 42


def test_tell_with_list_emitter_kwargs(kwargs_fixture):
    emitters, optimizer = kwargs_fixture
    optimizer.ask()
    optimizer.tell(
        [],
        [],
        [],
        emitter_kwargs=[{
            "arg": 1
        }, {
            "arg": 2
        }, {
            "arg": 3
        }],
    )
    for e, val in zip(emitters, [1, 2, 3]):
        assert e.arg == val
