"""Provides ArchiveDataFrame."""
import pandas as pd

from ribs.archives._elite import Elite

# Developer Note: The documentation for this class is hacked -- to add new
# methods, manually modify the template in docs/_templates/autosummary/class.rst


# TODO: docstring
class ArchiveDataFrame(pd.DataFrame):
    """A modified :class:`~pandas.DataFrame` for archive data.

    As this class inherits from :class:`~pandas.DataFrame`, it has all of the
    same methods and attributes, but it adds several more that make it
    convenient to work with elites. This documentation only lists the additional
    methods and attributes. Note that the `__init__` takes in the exact same
    arguments as :class:`~pandas.DataFrame`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterelites(self):
        """Iterator which outputs :class:`Elite`'s in the DataFrame."""
