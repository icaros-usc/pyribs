"""Tests for ribs.factory."""
import numpy as np
import pytest
import toml

import ribs.factory
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


@pytest.mark.parametrize(
    "registration_func",
    [
        ribs.factory.register_archive,
        ribs.factory.register_emitter,
        ribs.factory.register_optimizer,
    ],
    ids=[
        "archive",
        "emitter",
        "optimizer",
    ],
)
def test_registering_again_fails(registration_func):

    class NewClass:
        """Arbitrary class for registration."""

    with pytest.raises(ribs.factory.RegistrationError):
        registration_func("NewClass", NewClass)
        # The second registration should fail.
        registration_func("NewClass", NewClass)


@pytest.mark.parametrize("use_toml", [False, True], ids=["dict", "toml"])
def test_factory_from_config(use_toml, tmp_path):
    seed = 42
    batch_size = 4

    archive = GridArchive([64, 64], [(-1, 1), (-1, 1)], seed=seed)
    emitters = [
        GaussianEmitter([0.0, 0.0],
                        0.1,
                        archive,
                        batch_size=batch_size,
                        seed=seed)
    ]
    optimizer = Optimizer(archive, emitters)

    config_dict = {
        "archive": {
            "type": "GridArchive",
            "dims": [64, 64],
            "ranges": [(-1, 1), (-1, 1)],
            "seed": seed,
        },
        "emitters": [{
            "type": "GaussianEmitter",
            "x0": [0.0, 0.0],
            "sigma0": 0.1,
            "batch_size": batch_size,
            "seed": seed,
        }],
        "optimizer": {
            "type": "Optimizer",
        },
    }
    if use_toml:
        config_path = tmp_path / "config.toml"
        with config_path.open("w") as file:
            toml.dump(config_dict, file)
        created_optimizer = ribs.factory.from_config(config_path)
    else:
        created_optimizer = ribs.factory.from_config(config_dict)

    # Check types.
    assert isinstance(created_optimizer, Optimizer)
    assert isinstance(created_optimizer.archive, GridArchive)
    assert len(created_optimizer.emitters) == 1
    assert isinstance(created_optimizer.emitters[0], GaussianEmitter)

    # Check results from ask() and tell() -- since seeds are the same, all
    # results should be the same.
    optimizer_sols = optimizer.ask()
    created_optimizer_sols = created_optimizer.ask()
    assert len(optimizer_sols) == batch_size
    assert (optimizer_sols == created_optimizer_sols).all()

    objective_values = [0.0] * batch_size
    behavior_values = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    optimizer.tell(objective_values, behavior_values)
    created_optimizer.tell(objective_values, behavior_values)
    assert (optimizer.archive.as_pandas() ==
            created_optimizer.archive.as_pandas()).all(None)
