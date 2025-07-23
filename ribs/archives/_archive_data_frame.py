"""Provides ArchiveDataFrame."""

import re

import pandas as pd

# Developer Notes:
# - The documentation for this class is hacked -- to add new methods, manually modify
#   the template in docs/_templates/autosummary/class.rst
# - See here for info on extending DataFrame:
#   https://pandas.pydata.org/pandas-docs/stable/development/extending.html


class ArchiveDataFrame(pd.DataFrame):
    """A modified :class:`~pandas.DataFrame` for archive data.

    As this class inherits from :class:`~pandas.DataFrame`, it has the same methods,
    attributes, and arguments (even though the arguments shown here are ``*args`` and
    ``**kwargs``). However, this class adds methods that make it convenient to work with
    elites. This documentation only lists these additional methods and attributes.

    Example:

        This object is created by :meth:`~ArchiveBase.data` (i.e. users typically do not
        create it on their own)::

            df = archive.data(..., return_type="pandas")

        To iterate through every elite as a dict, use::

            for elite in df.iterelites():
                elite["solution"]  # Shape: (solution_dim,)
                elite["objective"]
                ...

        Arrays corresponding to individual fields can be accessed with
        :meth:`get_field`. For instance, the following is an array where entry ``i``
        contains the measures of the ``i``'th elite in the DataFrame::

            df.get_field("measures")

    .. warning::

        Calling :meth:`get_field` always creates a copy, so the following will copy the
        measures 3 times::

            df.get_field("measures")[0]
            df.get_field("measures").mean()
            df.get_field("measures").median()

        **Thus, if you need to use the method several times, we recommend storing it
        first, like so**::

            measures = df.get_field("measures")
            measures[0]
            measures.mean()
            measures.median()

    .. note::

        After saving an ArchiveDataFrame to a CSV, loading it with
        :func:`pandas.read_csv` will load a :class:`~pandas.DataFrame`. To load a CSV as
        an ArchiveDataFrame, pass the ``DataFrame`` from ``read_csv`` to
        ArchiveDataFrame::

            df = ArchiveDataFrame(pd.read_csv("file.csv"))

    .. note::

        Results of :meth:`get_field` "align" with each other -- e.g.
        ``get_field("measures")[i]`` corresponds to ``get_field("index")[i]``,
        ``get_field("objective")[i]``, and ``get_field("solution")[i]``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ArchiveDataFrame

    def iterelites(self):
        """Iterator that outputs every elite in the ArchiveDataFrame as a dict."""
        # Identify fields in the data frame. There are some edge cases here, such as if
        # someone purposely names their field with an underscore and a number at the end
        # like "foobar_0", but it covers most cases.
        fields = {}
        for col in self:
            split = col.split("_")
            if len(split) == 1:
                # Single column.
                fields[col] = None
            elif split[-1].isdigit():
                # If the last item in the split is numerical, this should match vector
                # fields like "measures_0".

                # Exclude last val and underscore - note negative sign.
                field_name = col[: -(len(split[-1]) + 1)]

                fields[field_name] = None
            else:
                fields[col] = None

        # Retrieve field data.
        for name in fields:
            fields[name] = self.get_field(name)

        n_elites = len(self)

        return map(
            lambda i: {name: arr[i] for name, arr in fields.items()},
            range(n_elites),
        )

    def get_field(self, field):
        """Array holding the data for the given field.

        None if there is no data for the field.
        """
        # Note: The column names cannot be pre-computed because the DataFrame columns
        # might change in-place, e.g., when a column is deleted.

        if field in self:
            # Scalar field -- e.g., "objective"
            return self[field].to_numpy(copy=True)
        else:
            # Vector field -- e.g., field="measures" and we want columns like
            # "measures_0" and "measures_1".
            field_re = f"{field}_\\d+"
            cols = [c for c in self if re.fullmatch(field_re, c)]
            return self[cols].to_numpy(copy=True) if cols else None
