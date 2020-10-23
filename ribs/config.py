"""Ribs config options and functions for dealing with configs."""

__all__ = [
    "DEFAULT_CONFIG",
    "update",
]

DEFAULT_CONFIG = {
    # This is a comment.
    "option1": 2.0,

    # This is another comment.
    "option2": 3.0,
}


def update(new, default=DEFAULT_CONFIG):
    """Merges the two configuration dicts, overriding ``default`` with ``new``

    Args:
        new (dict): New configuration options.
        default (dict): Default configuration options. Passing in this parameter
            is not recommended unless you have your own default config.
    Returns:
        A dict where the options from ``new`` have overridden those in
        ``default``.
    """

    # TODO: Examples in docstring

    return default
