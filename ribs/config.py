"""Ribs config options and functions for dealing with configs."""

__all__ = [
    "DEFAULT_CONFIG",
    "update",
    "merge_with_default_config",
]

#: Default configuration.
DEFAULT_CONFIG = {
    # (int) The size of each batch.
    "batch_size": 64,

    # Seed for random number generators. Leave None to avoid seeding.
    "seed": None,
}


def update(default, new):
    """Merges two configuration dicts, overriding ``default`` with ``new``.

    ``default`` is modified in-place with the new options.

    Handles nested dict options as well.

    Note that ``new`` is allowed to provide new keys to ``default``.

    If we have:

        ::

            default = {
                "a": 1,
                "b": {
                    "c": 2,
                    "d": 3,
                },
            }

            new = {
                "a": 4,
                "b": {
                    "c": 5,
                },
                "e": 6,
            }

    Then ``update(default, new)`` modifies ``default`` to be:

        ::

            default = {
                "a": 4,
                "b": {
                    "c": 5,
                    "d": 3,
                }
                "e": 6,
            }

    Args:
        default (dict): Default configuration options.
        new (dict): New configuration options.
    Raises:
        TypeError: When one attempts to override a non-dict value with a dict
        value.
    """
    for key in new:
        if isinstance(new[key], dict):
            if key not in default:
                default[key] = {}
            if not isinstance(default[key], dict):
                raise TypeError(f"The value at key `{key}` in new is a dict, "
                                f"but in default, it is {default[key]}")
            update(default[key], new[key])
        else:
            default[key] = new[key]


def merge_with_default_config(config):
    """Creates a configuration that updates ``DEFAULT_CONFIG`` with ``config``.

    Values in ``DEFAULT_CONFIG`` are overridden by those in ``config``,
    including those in nested dicts.

    Args:
        config (dict): A configuration dict to use to update ``DEFAULT_CONFIG``.
    Returns:
        dict: An updated version of ``DEFAULT_CONFIG``. The update is made with
        ``update``.
    """
    new_config = {}
    update(new_config, DEFAULT_CONFIG)
    update(new_config, config)
    return new_config
