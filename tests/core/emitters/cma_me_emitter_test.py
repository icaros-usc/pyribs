"""Tests for CMA-ME emitters.

We group these emitters together because they are very similar.

Currently:
- ImprovementEmitter
- RandomDirectionEmitter
- OptimizingEmitter.
"""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import (ImprovementEmitter, OptimizingEmitter,
                           RandomDirectionEmitter)


@pytest.mark.parametrize(
    "emitter_name",
    ["ImprovementEmitter", "RandomDirectionEmitter", "OptimizingEmitter"])
def test_auto_batch_size(emitter_name):
    archive = GridArchive([20, 20], [(-1.0, 1.0)] * 2)

    # Batch size is not provided, so it should be auto-generated.
    emitter = {
        "ImprovementEmitter":
            lambda: ImprovementEmitter(archive, np.zeros(20), 1.0),
        "RandomDirectionEmitter":
            lambda: RandomDirectionEmitter(archive, np.zeros(20), 1.0),
        "OptimizingEmitter":
            lambda: OptimizingEmitter(archive, np.zeros(20), 1.0),
    }[emitter_name]()

    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


@pytest.mark.parametrize(
    "emitter_name",
    ["ImprovementEmitter", "RandomDirectionEmitter", "OptimizingEmitter"])
def test_list_as_initial_solution(emitter_name):
    archive = GridArchive([20, 20], [(-1.0, 1.0)] * 2)

    emitter = {
        "ImprovementEmitter":
            lambda: ImprovementEmitter(archive, [0.0] * 20, 1.0),
        "RandomDirectionEmitter":
            lambda: RandomDirectionEmitter(archive, [0.0] * 20, 1.0),
        "OptimizingEmitter":
            lambda: OptimizingEmitter(archive, [0.0] * 20, 1.0),
    }[emitter_name]()

    # The list was passed in but should be converted to a numpy array.
    assert (emitter.x0 == np.zeros(20)).all()
