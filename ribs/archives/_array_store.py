"""Provides ArrayStore."""
import numpy as np
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import readonly


class ArrayStore:
    """Maintains a set of arrays that share a common dimension.

    The ArrayStore consists of several *fields* of data that are manipulated
    simultaneously via batch operations. Each field is a NumPy array with a
    dimension of ``(capacity, ...)`` and can be of any type.

    Since the arrays all share a common first dimension, they also share a
    common index. For instance, if we :meth:`retrieve` the data at indices ``[0,
    2, 1]``, we would get a dict that contains the objective and measures at
    indices 0, 2, and 1, e.g.::

        {
            "objective": [-1, 3, -5],
            "measures": [[0, 0], [2, 1], [3, 5]],
        }

    The ArrayStore supports several further operations, in particular a flexible
    :meth:`add` method that inserts data into the ArrayStore.

    Args:
        field_desc (dict): Description of fields in the array store. The
            description is a dict mapping from str to tuple of ``(shape,
            dtype)``. For instance, ``{"objective": ((), np.float32),
            "measures": ((10,), np.float32)}`` will create an "objective" field
            with shape ``(capacity,)`` and a "measures" field with shape
            ``(capacity, 10)``.
        capacity (int): Total possible entries in the store.

    Attributes:
        _props: Dict with properties that are common to every ArrayStore.

            * "capacity": Maximum number of data entries in the store.
            * "occupied": Boolean array of size ``(capacity,)`` indicating
              whether each index has data associated with it.
            * "n_occupied": Number of data entries currently in the store.
            * "occupied_list": Array of size ``(capacity,)`` listing all
              occupied indices in the store. Only the first ``n_occupied``
              elements will be valid.

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
        that have a corresponding data entry."""
        return self._props["n_occupied"]

    @property
    def capacity(self):
        """int: Maximum number of data entries in the store."""
        return self._props["capacity"]

    @property
    def occupied(self):
        """numpy.ndarray: Boolean array of size ``(capacity,)`` indicating
        whether each index has an data entry."""
        return self._props["occupied"]

    @property
    def occupied_list(self):
        """numpy.ndarray: Integer array listing all occupied indices in the
        store."""
        return readonly(
            self._props["occupied_list"][:self._props["n_occupied"]])

    def retrieve(self, indices):
        """Collects the data at the given indices.

        Args:
            indices (array-like): List of indices at which to collect data.
        Returns:
            - **occupied**: Array indicating which indices, among those passed,
              in have an associated data entry. For instance, if ``indices`` is
              ``[0, 1, 2]`` and only index 2 has data, then ``occupied`` will be
              ``[False, False, True]``.
            - **data**: Dict mapping from the field name to the field data at
              the given indices. For instance, if we have an ``objective`` field
              and request data at indices ``[4, 1, 0]``, we might get ``data``
              that looks like ``{"objective": [1.5, 6.0, 2.3]}``. Note that if a
              given index is not marked as occupied, it can have any data value
              associated with it. For instance, if index 1 was not occupied,
              then the 6.0 returned above should be ignored.
        """
        occupied = readonly(self._props["occupied"][indices])
        data = {
            name: readonly(arr[indices]) for name, arr in self._fields.items()
        }
        return occupied, data

    def add(self, indices, new_data, transforms):
        """Adds new data to the archive at the given indices.

        Raise:
            ValueError: The final version of ``new_data`` does not have the same
                keys as the fields of this store.
            ValueError: The final version of ``new_data`` has fields that have a
                different length than ``indices``.
        """
        add_info = {}
        for transform in transforms:
            occupied, cur_data = self.retrieve(indices)
            indices, new_data, add_info = transform(indices, occupied, cur_data,
                                                    new_data, add_info)

        # Verify that new_data ends up with the correct fields after the
        # transforms.
        if new_data.keys() != self._fields.keys():
            raise ValueError(
                f"`new_data` had keys {new_data.keys()} but should have the "
                f"same keys as this ArrayStore, i.e., {self._fields.keys()}")

        # Verify that the array shapes match the indices.
        for name, arr in new_data.items():
            if len(arr) != len(indices):
                raise ValueError(
                    f"In `new_data`, the array for `{name}` has length "
                    f"{len(arr)} but should be the same length as indices "
                    f"({len(indices)})")

        # Update occupancy data.
        unique_indices = np.where(aggregate(indices, 1, func="len") != 0)[0]
        cur_occupied = self._props["occupied"][unique_indices]
        new_indices = unique_indices[~cur_occupied]
        n_occupied = self._props["n_occupied"]
        self._props["occupied"][new_indices] = True
        self._props["occupied_list"][n_occupied:n_occupied +
                                     len(new_indices)] = new_indices
        self._props["n_occupied"] = n_occupied + len(new_indices)

        # Insert into the ArrayStore. Note that we do not assume indices are
        # unique. Hence, when updating occupancy data above, we computed the
        # unique indices. In contrast, here we let NumPy's default behavior
        # handle duplicate indices.
        for name, arr in self._fields.items():
            arr[indices] = new_data[name]

        return add_info

    def clear(self):
        """Removes all entries from the store."""
        self._props["n_occupied"] = 0  # Effectively clears occupied_list too.
        self._props["occupied"].fill(False)
