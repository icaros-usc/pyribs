"""Provides ArrayStore."""


class ArrayStore:
    """Maintains a set of arrays that share a common dimension.

    The ArrayStore consists of several *fields* of data that are manipulated
    simultaneously via batch operations. Each field is a NumPy array with a
    dimension of ``(capacity, ...)`` and can be of any type.

    Since the arrays all share a common first dimension, they also share a
    common index. For instance, if we :meth:`retrieve` the entries at indices
    ``[0, 2, 1]``, we would get a dict that contains the objective and measures
    at indices 0, 2, and 1::

        {
            "objective": [-1, 3, -5],
            "measures": [[0, 0], [2, 1], [3, 5]],
        }

    The ArrayStore supports several further operations, in particular a flexible
    :meth:`add` method that inserts entries into the ArrayStore.

    Args:
        field_desc: TODO
        capacity: TODO

    Attributes:
        _props: TODO
        _fields: TODO
    """

    def __init__(self):
        pass
