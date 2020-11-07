"""Functions for dealing with configs."""
import toml

__all__ = [
    "create_config",
    "save_configs",
    "load_configs",
]


def create_config(config, config_class):
    """Creates an instance of ``config_class`` given the user's ``config``.

    - If ``config`` is None, a default instance of ``config_class`` is created.
    - If ``config`` is a dict, the dict is passed into ``config_class`` as
      keyword args.
    - Otherwise, ``config`` is simply returned (we assume it is an instance of
      ``config_class``).

    Args:
        config (None, dict, or config_class): User-provided configuration.
    Returns:
        An instance of ``config_class``, described as above.
    """
    if config is None:
        return config_class()
    if isinstance(config, dict):
        return config_class(**config)
    return config


def _single_config_data(config):
    """Returns a dict with the data to save for a single config."""
    return {
        "type": config.__class__.__name__,
        "data": config.__dict__,
    }


def save_configs(optimizer_config, archive_config, emitter_configs, filename):
    """Saves all configs to a TOML file.

    Args:
        optimizer_config: Configuration object for an optimizer, such as
            :class:`ribs.optimizers.OptimizerConfig`.
        archive_config: Configuration object for an archive, such as
            :class:`ribs.archives.GridArchiveConfig`.
        emitter_configs (list): List of configuration objects for emitters, such
            as :class:`ribs.emitters.GaussianEmitterConfig`.
        filename (str): Path to save the TOML file.
    """
    with open(filename, "w") as file:
        toml.dump(
            {
                "optimizer":
                    _single_config_data(optimizer_config),
                "archive":
                    _single_config_data(archive_config),
                "emitters":
                    [_single_config_data(config) for config in emitter_configs],
            },
            file,
        )


def _load_single_config(data, name_to_config_class):
    """Creates one config from data returned by :meth:`_single_config_data`."""
    config_class = name_to_config_class[data["type"]]
    return config_class(**data["data"])


def load_configs(filename):
    """Loads configs from a TOML file to reconstruct an optimizer.

    Args:
        filename (str): Path from which to load the TOML file.
    Returns:
        tuple: 3-element tuple containing:

            **optimizer_config**: Configuration object for an optimizer.

            **archive_config**: Configuration object for an archive.

            **emitter_configs**: List of configuration objects for emitters.
    """
    # We cannot import these at top-level because all these modules import this
    # one (config), so we would have a circular dependency.
    # pylint: disable = import-outside-toplevel, cyclic-import
    from ribs.archives._cvt_archive import CVTArchiveConfig
    from ribs.archives._grid_archive import GridArchiveConfig
    from ribs.emitters._gaussian_emitter import GaussianEmitterConfig
    from ribs.optimizers._optimizer import OptimizerConfig

    name_to_config_class = {
        "GridArchiveConfig": GridArchiveConfig,
        "CVTArchiveConfig": CVTArchiveConfig,
        "GaussianEmitterConfig": GaussianEmitterConfig,
        "OptimizerConfig": OptimizerConfig,
    }

    with open(filename, "r") as file:
        data = toml.load(file)
        optimizer_config = _load_single_config(data["optimizer"],
                                               name_to_config_class)
        archive_config = _load_single_config(data["archive"],
                                             name_to_config_class)
        emitter_configs = [
            _load_single_config(d, name_to_config_class)
            for d in data["emitters"]
        ]
        return optimizer_config, archive_config, emitter_configs
