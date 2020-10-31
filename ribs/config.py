"""Functions for dealing with configs."""


def create_config(config, config_class):
    """Creates an instance of `config_class` given the user's `config`.

    - If `config` is None, a default instance of `config_class` is created.
    - If `config` is a dict, the dict is passed into `config_class` as
      keyword args.
    - Otherwise, `config` is simply returned (we assume it is an instance of
      `config_class`).

    Args:
        config (None, dict, or config_class): User-provided configuration.
    Returns:
        (config_class): An instance of `config_class`, as described above.
    """
    if config is None:
        return config_class()
    if isinstance(config, dict):
        return config_class(**config)
    return config


def save_configs(optimizer_config, archive_config, emitter_configs, filename):
    """Saves all configs to a ___ file."""

    # TODO


def load_configs(filename):
    """Loads configs from a file to reconstruct an optimizer."""

    # TODO
