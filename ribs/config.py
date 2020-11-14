"""Functions for dealing with configs."""
import pathlib

import toml

import ribs.archives
import ribs.emitters
import ribs.optimizers

__all__ = [
    "create_optimizer",
]

_ARCHIVE_TYPES = {
    "GridArchive": ribs.archives.GridArchive,
    "CVTArchive": ribs.archives.CVTArchive,
}

_EMITTER_TYPES = {
    "GaussianEmitter": ribs.emitters.GaussianEmitter,
    "IsoLineEmitter": ribs.emitters.IsoLineEmitter,
}

_OPTIMIZER_TYPES = {
    "Optimizer": ribs.optimizers.Optimizer,
}


def _remove_type_key(config):
    """Returns a shallow copy of the config, with the "type" key removed."""
    config = config.copy()
    config.pop("type")
    return config


def create_optimizer(config):
    """Creates an optimizer and its archive and emitters from a single config.

    The config must be either a dict, or the name of a toml file (str or
    pathlib.Path are okay). In any case, the config must be structured as
    follows::

        {
            "archive": {
                # This class must be under `ribs.archives`.
                "type": "GridArchive",
                # Args for the archive.
                ...
            },
            "emitters": [
                # Each item in this list configures an emitter.
                {
                    # This class must be under `ribs.emitters`.
                    "type": "GaussianEmitter",
                    # Args for the emitter. Exclude the `archive` param, as we
                    # will automatically add it for you.
                    ...
                }
                # More emitters.
                ...
            ],
            "optimizer": {
                # This class must be under `ribs.optimizers`.
                "type": "Optimizer",
                # Args for the optimizer.
                ...
            },
        }

    Args:
        config (dict or str or pathlib.Path): Dict of configuration options
            described as above, or the name of a toml file with the structure
            shown above.
    Returns:
        ribs.optimizers.Optimizer: An optimizer created according to the
        options specified in the config.
    """
    if isinstance(config, (str, pathlib.Path)):
        with open(str(config), "r") as file:
            config = toml.load(file)

    archive_class = _ARCHIVE_TYPES[config["archive"]["type"]]
    archive = archive_class(**_remove_type_key(config["archive"]))

    emitters = []
    for emitter_config in config["emitters"]:
        emitter_class = _EMITTER_TYPES[emitter_config["type"]]
        emitters.append(
            emitter_class(
                **_remove_type_key(emitter_config),
                archive=archive,
            ))

    optimizer_class = _OPTIMIZER_TYPES[config["optimizer"]["type"]]
    return optimizer_class(
        archive=archive,
        emitters=emitters,
        **_remove_type_key(config["optimizer"]),
    )
