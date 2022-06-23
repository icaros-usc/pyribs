"""Tests for the rankers."""

from ribs.emitters import GaussianEmitter
from ribs.emitters.selectors import FilterSelector, MuSelector


def test_filter_selector(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    solutions = emitter.ask()
    objective_values = [0, 3, 2, 1]
    behavior_values = [[0], [0], [1], [1]]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    selector = FilterSelector()

    num_parents = selector.select(emitter, archive, None, solutions,
                                  objective_values, behavior_values, metadata,
                                  statuses, values)
    print(statuses)
    assert num_parents == 3


def test_mu_selector(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    solutions = emitter.ask()
    objective_values = [0, 1, 2, 3]
    behavior_values = [[0], [0], [1], [1]]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    selector = MuSelector()

    num_parents = selector.select(emitter, archive, None, solutions,
                                objective_values, behavior_values, metadata,
                                statuses, values)

    assert num_parents == 2
