"""Functions for factory-style creation of ribs.

.. autosummary::
    :toctree:

    ribs.factory.from_config
    ribs.factory.register_archive
    ribs.factory.register_emitter
    ribs.factory.register_optimizer
    ribs.factory.RegistrationError
"""
import pathlib

import toml

from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._improvement_emitter import ImprovementEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter
from ribs.emitters._optimizing_emitter import OptimizingEmitter
from ribs.emitters._random_direction_emitter import RandomDirectionEmitter
from ribs.optimizers._optimizer import Optimizer

__all__ = [
    "from_config",
    "register_archive",
    "register_emitter",
    "register_optimizer",
    "RegistrationError",
]

#
# Factory registration.
#

_ARCHIVE_TYPES = {}

_EMITTER_TYPES = {}

_OPTIMIZER_TYPES = {}


class RegistrationError(Exception):
    """Raised when there is an issue with factory registration."""


def register_archive(name, archive_class):
    """Registers a new archive with the ribs factory.

    After registration, pass ``name`` in the ``config["archive"]["type"]`` field
    when using :meth:`from_config`.

    As an example, :class:`ribs.archives.GridArchive` is registered with::

        register_archive("GridArchive", ribs.archives.GridArchive)

    Args:
        name (str): the name of the archive.
        archive_class (type): The archive's class.
    Raises:
        RegistrationError: The archive is already registered.
    """
    if name in _ARCHIVE_TYPES:
        raise RegistrationError(f"Archive '{name}' is already registered")
    _ARCHIVE_TYPES[name] = archive_class


def register_emitter(name, emitter_class):
    """Registers a new emitter with the ribs factory.

    After registration, pass ``name`` in the ``config["emitters"][i]["type"]``
    fields (where ``i`` is a list index) when using :meth:`from_config`.

    As an example, :class:`ribs.emitters.GaussianEmitter` is registered with::

        register_emitter("GaussianEmitter", ribs.emitters.GaussianEmitter)

    Args:
        name (str): the name of the emitter.
        emitter_class (type): The emitter's class.
    Raises:
        RegistrationError: The emitter is already registered.
    """
    if name in _EMITTER_TYPES:
        raise RegistrationError(f"Emitter '{name}' is already registered")
    _EMITTER_TYPES[name] = emitter_class


def register_optimizer(name, optimizer_class):
    """Registers a new optimizer with the ribs factory.

    After registration, pass ``name`` in the ``config["optimizer"]["type"]``
    field when using :meth:`from_config`.

    As an example, :class:`ribs.optimizers.Optimizer` is registered with::

        register_optimizer("Optimizer", ribs.optimizers.Optimizer)

    Args:
        name (str): the name of the optimizer.
        optimizer_class (type): The optimizer's class.
    Raises:
        RegistrationError: The optimizer is already registered.
    """
    if name in _OPTIMIZER_TYPES:
        raise RegistrationError(f"Optimizer '{name}' is already registered")
    _OPTIMIZER_TYPES[name] = optimizer_class


register_archive("CVTArchive", CVTArchive)
register_archive("GridArchive", GridArchive)
register_archive("SlidingBoundariesArchive", SlidingBoundariesArchive)
register_emitter("GaussianEmitter", GaussianEmitter)
register_emitter("ImprovementEmitter", ImprovementEmitter)
register_emitter("IsoLineEmitter", IsoLineEmitter)
register_emitter("OptimizingEmitter", OptimizingEmitter)
register_emitter("RandomDirectionEmitter", RandomDirectionEmitter)
register_optimizer("Optimizer", Optimizer)

#
# Factory creation.
#


def _remove_type_key(config):
    """Returns a shallow copy of the config, with the "type" key removed."""
    config = config.copy()
    config.pop("type")
    return config


def _attempt_creation(entity_name, type_name, type_dict, provided_kwargs,
                      additional_kwargs):
    """Tries to create an archive, emitter, or optimizer (i.e. an "entity").

    See ``from_config`` for example usage.
    """
    if type_name not in type_dict:
        raise KeyError(f"{entity_name.title()} '{type_name}' is not registered")
    entity_type = type_dict[type_name]
    return entity_type(**additional_kwargs, **_remove_type_key(provided_kwargs))


def from_config(config):
    """Creates an optimizer and its archive and emitters from a single config.

    The config must be either a dict, or the name of a toml file (str or
    pathlib.Path are okay). In any case, the config must be structured as
    follows::

        {
            "archive": {
                # This class must be under `ribs.archives`, or it must have been
                # registered with register_archive().
                "type": "GridArchive",
                # Args for the archive.
                ...
            },
            "emitters": [
                # Each item in this list configures an emitter.
                {
                    # This class must be under `ribs.emitters`, or it must have
                    # been registered with register_emitter().
                    "type": "GaussianEmitter",
                    # Args for the emitter. Exclude the `archive` param, as it
                    # is automatically added.
                    ...
                }
                # More emitters.
                ...
            ],
            "optimizer": {
                # This class must be under `ribs.optimizers`, or it must have
                # been registered with register_optimizer().
                "type": "Optimizer",
                # Args for the optimizer.
                ...
            },
        }

    In TOML format, the config would look like this::

        [archive]
        type = "GridArchive"
        ... # Other archive args.

        [[emitters]]
        type = "GaussianEmitter"
        ... # Other emitter args.

        [[emitters]]
        ... # Another emitter

        [optimizer]
        type = "Optimizer"
        ... # Other optimizer args.

    .. note:: If the config has custom archives, emitters, or optimizers,
        register them with the appropriate method *before* running this method.

    Args:
        config (dict or str or pathlib.Path): Dict of configuration options
            described as above, or the name of a toml file with the structure
            shown above.
    Returns:
        ribs optimizer: An optimizer created according to the options specified
        in the config.
    Raises:
        KeyError: An archive, emitter, or optimizer specified in the config is
            not registered.
    """
    if isinstance(config, (str, pathlib.Path)):
        with open(str(config), "r") as file:
            config = toml.load(file)

    archive = _attempt_creation("Archive", config["archive"]["type"],
                                _ARCHIVE_TYPES, config["archive"], {})

    emitters = []
    for emitter_config in config["emitters"]:
        emitters.append(
            _attempt_creation(
                "Emitter",
                emitter_config["type"],
                _EMITTER_TYPES,
                emitter_config,
                {"archive": archive},
            ))

    return _attempt_creation("Optimizer", config["optimizer"]["type"],
                             _OPTIMIZER_TYPES, config["optimizer"], {
                                 "archive": archive,
                                 "emitters": emitters
                             })
