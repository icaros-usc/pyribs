"""Provides ArrayStore."""
import numpy as np

from ribs._utils import readonly


class ArrayStore:
    """Maintains a set of arrays that share a common dimension.

    The ArrayStore consists of several *fields* of data that are manipulated
    simultaneously via batch operations. Each field is a NumPy array with a
    dimension of ``(capacity, ...)`` and can be of any type.

    Since the arrays all share a common first dimension, they also share a
    common index. For instance, if we :meth:`retrieve` the entries at indices
    ``[0, 2, 1]``, we would get a dict that contains the objective and measures
    at indices 0, 2, and 1, e.g.::

        {
            "objective": [-1, 3, -5],
            "measures": [[0, 0], [2, 1], [3, 5]],
        }

    The ArrayStore supports several further operations, in particular a flexible
    :meth:`add` method that inserts entries into the ArrayStore.

    Args:
        field_desc (dict): Description of fields in the array store. The
            description is a dict mapping from str to tuple of ``(shape,
            dtype)``. For instance, ``{"objective": ((), np.float32),
            "measures": ((10,), np.float32)}`` will create an "objective" field
            with shape ``(capacity,)`` and a "measures" field with shape
            ``(capacity, 10)``.
        capacity (int): Total possible cells in the store.

    Attributes:
        _props: Dict with properties that are common to every ArrayStore.

            * "capacity": Maximum number of entries in the store.
            * "occupied": Boolean array of size ``(capacity,)`` indicating
              whether each index has an entry.
            * "n_occupied": Number of entries currently in the store.
            * "occupied_list": Array of size ``(capacity,)`` listing all
              occupied indices in the store. Only the first ``n_occupied``
              entries will be valid.

        _fields: Dict holding all the arrays with their data.
    """

    def __init__(self, field_desc, capacity):
        self._props = {
            "capacity": capacity,
            "occupied": np.zeros(capacity, dtype=bool),
            "n_occupied": 0,
            "occupied_list": np.empty(capacity, dtype=int),
        }

        self._fields = {}
        for name, (field_shape, dtype) in field_desc.items():
            array_shape = (capacity,) + tuple(field_shape)
            self._fields[name] = np.empty(array_shape, dtype)

    def __len__(self):
        """Number of occupied indices in the store, i.e.g, number of indices
        that have a corresponding entry."""
        return self._props["n_occupied"]

    @property
    def capacity(self):
        """int: Maximum number of entries in the store."""
        return self._props["capacity"]

    @property
    def occupied(self):
        """numpy.ndarray: Boolean array of size ``(capacity,)`` indicating
        whether each index has an entry."""
        return self._props["occupied"]

    @property
    def occupied_list(self):
        """numpy.ndarray: Integer array listing all occupied indices in the
        store."""
        return readonly(
            self._props["occupied_list"][:self._props["n_occupied"]])
