"""Provides ArrayStore."""
import itertools
import numbers
from enum import IntEnum
from functools import cached_property

import numpy as np
from numpy_groupies import aggregate_nb as aggregate
from pandas import DataFrame

from ribs._utils import readonly


class Update(IntEnum):
    """Indices into the updates array in ArrayStore."""
    ADD = 0
    CLEAR = 1


class ArrayStoreIterator:
    """An iterator for an ArrayStore's entries."""

    # pylint: disable = protected-access

    def __init__(self, store):
        self.store = store
        self.iter_idx = 0
        self.state = store._props["updates"].copy()

    def __iter__(self):
        """This is the iterator, so it returns itself."""
        return self

    def __next__(self):
        """Returns dicts with each entry's data.

        Raises RuntimeError if the store was modified.
        """
        if not np.all(self.state == self.store._props["updates"]):
            # This check should go before the StopIteration check because a call
            # to clear() would cause the len(self.store) to be 0 and thus
            # trigger StopIteration.
            raise RuntimeError(
                "ArrayStore was modified with add() or clear() during "
                "iteration.")

        if self.iter_idx >= len(self.store):
            raise StopIteration

        idx = self.store._props["occupied_list"][self.iter_idx]
        self.iter_idx += 1

        d = {"index": idx}
        for name, arr in self.store._fields.items():
            d[name] = arr[idx]

        return d


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
            description is a dict mapping from a str to a tuple of ``(shape,
            dtype)``. For instance, ``{"objective": ((), np.float32),
            "measures": ((10,), np.float32)}`` will create an "objective" field
            with shape ``(capacity,)`` and a "measures" field with shape
            ``(capacity, 10)``. Note that field names must be valid Python
            identifiers.
        capacity (int): Total possible entries in the store.

    Attributes:
        _props (dict): Properties that are common to every ArrayStore.

            * "capacity": Maximum number of data entries in the store.
            * "occupied": Boolean array of size ``(capacity,)`` indicating
              whether each index has data associated with it.
            * "n_occupied": Number of data entries currently in the store.
            * "occupied_list": Array of size ``(capacity,)`` listing all
              occupied indices in the store. Only the first ``n_occupied``
              elements will be valid.
            * "updates": Int array recording number of calls to functions that
              modified the store.

        _fields (dict): Holds all the arrays with their data.

    Raises:
        ValueError: One of the fields in ``field_desc`` has a reserved name
            (currently, "index" is the only reserved name).
        ValueError: One of the fields in ``field_desc`` has a name that is not a
            valid Python identifier.
    """

    def __init__(self, field_desc, capacity):
        self._props = {
            "capacity": capacity,
            "occupied": np.zeros(capacity, dtype=bool),
            "n_occupied": 0,
            "occupied_list": np.empty(capacity, dtype=np.int32),
            "updates": np.array([0, 0]),
        }

        self._fields = {}
        for name, (field_shape, dtype) in field_desc.items():
            if name == "index":
                raise ValueError(f"`{name}` is a reserved field name.")
            if not name.isidentifier():
                raise ValueError(
                    f"Field names must be valid identifiers: `{name}`")

            if isinstance(field_shape, numbers.Integral):
                field_shape = (field_shape,)

            array_shape = (capacity,) + tuple(field_shape)
            self._fields[name] = np.empty(array_shape, dtype)

    def __len__(self):
        """Number of occupied indices in the store, i.e., number of indices that
        have a corresponding data entry."""
        return self._props["n_occupied"]

    def __iter__(self):
        """Iterates over entries in the store.

        When iterated over, this iterator yields dicts mapping from the fields
        to the individual entries. For instance, if we had an "objective" field,
        one entry might look like ``{"index": 1, "objective": 6.0}`` (similar to
        :meth:`retrieve`, the index is included in the output).

        Example:

            ::

                for entry in store:
                    entry["index"]
                    entry["objective"]
                    ...
        """
        return ArrayStoreIterator(self)

    @property
    def capacity(self):
        """int: Maximum number of data entries in the store."""
        return self._props["capacity"]

    @property
    def occupied(self):
        """numpy.ndarray: Boolean array of size ``(capacity,)`` indicating
        whether each index has a data entry."""
        return readonly(self._props["occupied"].view())

    @property
    def occupied_list(self):
        """numpy.ndarray: int32 array listing all occupied indices in the
        store."""
        return readonly(
            self._props["occupied_list"][:self._props["n_occupied"]])

    @cached_property
    def field_desc(self):
        """dict: Description of fields in the store.

        Example:

            ::

                store.field_desc == {
                    "objective": ((), np.float32),
                    "measures": ((10,), np.float32)
                }

        See the constructor ``field_desc`` parameter for more info. Unlike in
        the field_desc in the constructor, which accepts ints for 1D field
        shapes (e.g., ``5``), this field_desc shows 1D field shapes as tuples of
        1 entry (e.g., ``(5,)``). Since dicts in Python are ordered, note that
        this dict will have the same order as in the constructor.
        """
        return {
            name: (arr.shape[1:], arr.dtype)
            for name, arr in self._fields.items()
        }

    @cached_property
    def field_list(self):
        """list: List of fields in the store.

        Example:

            ::

                store.field_list == ["objective", "measures"]
        """
        # Python dicts are ordered, so this will follow the same order as in the
        # constructor.
        return list(self._fields)

    def retrieve(self, indices, fields=None, return_type="dict"):
        """Collects data at the given indices.

        Args:
            indices (array-like): List of indices at which to collect data.
            fields (str or array-like of str): List of fields to include. By
                default, all fields will be included, with an additional "index"
                as the last field ("index" can also be placed anywhere in this
                list). This can also be a single str indicating a field name.
            return_type (str): Type of data to return. See the ``data`` returned
                below. Ignored if ``fields`` is a str.

        Returns:
            tuple: 2-element tuple consisting of:

            - **occupied**: Array indicating which indices, among those passed
              in, have an associated data entry. For instance, if ``indices`` is
              ``[0, 1, 2]`` and only index 2 has data, then ``occupied`` will be
              ``[False, False, True]``.

              Note that if a given index is not marked as occupied, it can have
              any data value associated with it. For instance, if index 1 was
              not occupied, then the 6.0 returned in the ``dict`` example below
              should be ignored.

            - **data**: The data at the given indices. If ``fields`` was a
              single str, this will just be an array holding data for the given
              field. Otherwise, this data can take the following forms,
              depending on the ``return_type`` argument:

              - ``return_type="dict"``: Dict mapping from the field name to the
                field data at the given indices. For instance, if we have an
                ``objective`` field and request data at indices ``[4, 1, 0]``,
                we would get ``data`` that looks like ``{"objective": [1.5, 6.0,
                2.3], "index": [4, 1, 0]}``. Observe that we also return the
                indices as an ``index`` entry in the dict. The keys in this dict
                can be modified using the ``fields`` arg; duplicate keys will be
                ignored since the dict stores unique keys.

              - ``return_type="tuple"``: Tuple of arrays matching the order
                given in ``fields``. For instance, if ``fields`` was
                ``["objective", "measures"]``, we would receive a tuple of
                ``(objective_arr, measures_arr)``. In this case, the results
                from ``retrieve`` could be unpacked as::

                    occupied, (objective, measures) = store.retrieve(
                        ...,
                        return_type="tuple",
                    )

                Unlike with the ``dict`` return type, duplicate fields will show
                up as duplicate entries in the tuple, e.g.,
                ``fields=["objective", "objective"]`` will result in two
                objective arrays being returned.

                By default, (i.e., when ``fields=None``), the fields in the
                tuple will be ordered according to the ``field_desc`` argument
                in the constructor, along with ``index`` as the last field.

              - ``return_type="pandas"``: A :class:`pandas.DataFrame` with the
                following columns (by default):

                - For fields that are scalars, a single column with the field
                  name. For example, ``objective`` would have a single column
                  called ``objective``.
                - For fields that are 1D arrays, multiple columns with the name
                  suffixed by its index. For instance, if we have a ``measures``
                  field of length 10, we create 10 columns with names
                  ``measures_0``, ``measures_1``, ..., ``measures_9``. We do not
                  currently support fields with >1D data.
                - 1 column of integers (``np.int32``) for the index, named
                  ``index``.

                In short, the dataframe might look like this:

                +-----------+------------+------+-------+
                | objective | measures_0 | ...  | index |
                +===========+============+======+=======+
                |           |            | ...  |       |
                +-----------+------------+------+-------+

                Like the other return types, the columns can be adjusted with
                the ``fields`` parameter.

            All data returned by this method will be a copy, i.e., the data will
            not update as the store changes.

        Raises:
            ValueError: Invalid field name provided.
            ValueError: Invalid return_type provided.
        """
        single_field = isinstance(fields, str)
        indices = np.asarray(indices, dtype=np.int32)
        occupied = self._props["occupied"][indices]  # Induces copy.

        if single_field:
            data = None
        elif return_type in ("dict", "pandas"):
            data = {}
        elif return_type == "tuple":
            data = []
        else:
            raise ValueError(f"Invalid return_type {return_type}.")

        if single_field:
            fields = [fields]
        elif fields is None:
            fields = itertools.chain(self._fields, ["index"])

        for name in fields:
            # Collect array data.
            #
            # Note that fancy indexing with indices already creates a copy, so
            # only `indices` needs to be copied explicitly.
            if name == "index":
                arr = np.copy(indices)
            elif name in self._fields:
                arr = self._fields[name][indices]  # Induces copy.
            else:
                raise ValueError(f"`{name}` is not a field in this ArrayStore.")

            # Accumulate data into the return type.
            if single_field:
                data = arr
            elif return_type == "dict":
                data[name] = arr
            elif return_type == "tuple":
                data.append(arr)
            elif return_type == "pandas":
                if len(arr.shape) == 1:  # Scalar entries.
                    data[name] = arr
                elif len(arr.shape) == 2:  # 1D array entries.
                    for i in range(arr.shape[1]):
                        data[f"{name}_{i}"] = arr[:, i]
                else:
                    raise ValueError(
                        f"Field `{name}` has shape {arr.shape[1:]} -- "
                        "cannot convert fields with shape >1D to Pandas")

        # Postprocess return data.
        if return_type == "tuple":
            data = tuple(data)
        elif return_type == "pandas":
            # Data above are already copied, so no need to copy again.
            data = DataFrame(data, copy=False)

        return occupied, data

    def data(self, fields=None, return_type="dict"):
        """Retrieves data for all entries in the store.

        Equivalent to calling :meth:`retrieve` with :attr:`occupied_list`.

        Args:
            fields (str or array-like of str): See :meth:`retrieve`.
            return_type (str): See :meth:`retrieve`.
        Returns:
            See ``data`` in :meth:`retrieve`. ``occupied`` is not returned since
            all indices are known to be occupied in this method.
        """
        return self.retrieve(self.occupied_list, fields, return_type)[1]

    def add(self, indices, new_data, extra_args, transforms):
        """Adds new data to the store at the given indices.

        The indices, new_data, and add_info are passed through transforms before
        adding to the store. The general idea is that these transforms will
        gradually modify the indices, new_data, and add_info. For instance, they
        can add new fields to new_data (new_data may not initially have all the
        same fields as the store). Alternatively, they can filter out duplicate
        indices, eg if multiple entries are being inserted at the same index we
        can choose one with the best objective. As another example, the
        transforms can add stats to the add_info or delete fields from the
        add_info.

        The signature of a transform is as follows::

            def transform(indices, new_data, add_info, extra_args,
                          occupied, cur_data) ->
                (indices, new_data, add_info):

        Transform parameters:

        - **indices** (array-like): Array of indices at which new_data should be
          inserted.
        - **new_data** (dict): New data for the given indices. Maps from field
          name to the array of new data for that field.
        - **add_info** (dict): Information to return to the user about the
          addition process. Example info includes whether each entry was
          ultimately inserted into the store, as well as general statistics.
          For the first transform, this will be an empty dict.
        - **extra_args** (dict): Additional arguments for the transform.
        - **occupied** (array-like): Whether the given indices are currently
          occupied. Same as that given by :meth:`retrieve`.
        - **cur_data** (dict): Data at the current indices in the store. Same as
          that given by :meth:`retrieve`.

        Transform outputs:

        - **indices** (array-like): Modified indices. We do NOT assume that the
          final indices will be unique.
        - **new_data** (dict): Modified new_data. At the end of the transforms,
          it should have the same keys as the store. If ``indices`` is empty,
          ``new_data`` will be ignored.
        - **add_info** (dict): Modified add_info.

        Args:
            indices (array-like): Initial list of indices for addition.
            new_data (dict): Initial data for addition.
            extra_args (dict): Dict containing additional arguments to pass to
                the transforms. The dict is passed directly (i.e., no unpacking
                like with kwargs).
            transforms (list): List of transforms on the data to be added.

        Returns:
            dict: Final ``add_info`` from the transforms. ``new_data`` and
            ``indices`` are not returned; rather, the ``new_data`` is added into
            the store at ``indices``.

        Raise:
            ValueError: The final version of ``new_data`` does not have the same
                keys as the fields of this store.
            ValueError: The final version of ``new_data`` has fields that have a
                different length than ``indices``.
        """
        self._props["updates"][Update.ADD] += 1

        add_info = {}
        for transform in transforms:
            occupied, cur_data = self.retrieve(indices)
            indices, new_data, add_info = transform(indices, new_data, add_info,
                                                    extra_args, occupied,
                                                    cur_data)

        # Shortcut when there is nothing to add to the store.
        if len(indices) == 0:
            return add_info

        # Verify that the array shapes match the indices.
        for name, arr in new_data.items():
            if len(arr) != len(indices):
                raise ValueError(
                    f"In `new_data`, the array for `{name}` has length "
                    f"{len(arr)} but should be the same length as indices "
                    f"({len(indices)})")

        # Verify that new_data ends up with the correct fields after the
        # transforms.
        if new_data.keys() != self._fields.keys():
            raise ValueError(
                f"`new_data` had keys {new_data.keys()} but should have the "
                f"same keys as this ArrayStore, i.e., {self._fields.keys()}. "
                "You may be seeing this error if your archive has "
                "extra_fields but the fields were not passed into "
                "archive.add() or scheduler.tell().")

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
        self._props["updates"][Update.CLEAR] += 1
        self._props["n_occupied"] = 0  # Effectively clears occupied_list too.
        self._props["occupied"].fill(False)

    def resize(self, capacity):
        """Resizes the store to the given capacity.

        Args:
            capacity (int): New capacity.
        Raises:
            ValueError: The new capacity is less than or equal to the current
                capacity.
        """
        if capacity <= self._props["capacity"]:
            raise ValueError(
                f"New capacity ({capacity}) must be greater than current "
                f"capacity ({self._props['capacity']}.")

        cur_capacity = self._props["capacity"]
        self._props["capacity"] = capacity

        cur_occupied = self._props["occupied"]
        self._props["occupied"] = np.zeros(capacity, dtype=bool)
        self._props["occupied"][:cur_capacity] = cur_occupied

        cur_occupied_list = self._props["occupied_list"]
        self._props["occupied_list"] = np.empty(capacity, dtype=np.int32)
        self._props["occupied_list"][:cur_capacity] = cur_occupied_list

        for name, cur_arr in self._fields.items():
            new_shape = (capacity,) + cur_arr.shape[1:]
            self._fields[name] = np.empty(new_shape, cur_arr.dtype)
            self._fields[name][:cur_capacity] = cur_arr

    def as_raw_dict(self):
        """Returns the raw data in the ArrayStore as a one-level dictionary.

        To collapse the dict, we prefix each key with ``props.`` or ``fields.``,
        so the result looks as follows::

            {
              "props.capacity": ...,
              "props.occupied": ...,
              ...
              "fields.objective": ...,
              ...
            }

        Returns:
            dict: See description above.
        """
        d = {}
        for prefix, attr in [("props", self._props), ("fields", self._fields)]:
            for name, val in attr.items():
                if isinstance(val, np.ndarray):
                    val = readonly(val.view())
                d[f"{prefix}.{name}"] = val
        return d

    @staticmethod
    def from_raw_dict(d):
        """Loads an ArrayStore from a dict of raw info.

        Args:
            d (dict): Dict returned by :meth:`as_raw_dict`.
        Returns:
            ArrayStore: The new ArrayStore created from d.
        Raises:
            ValueError: The loaded props dict has the wrong keys.
        """
        # pylint: disable = protected-access

        store = ArrayStore({}, 0)  # Create an empty store.

        props = {
            name[len("props."):]: arr
            for name, arr in d.items()
            if name.startswith("props.")
        }
        if props.keys() != store._props.keys():
            raise ValueError(
                f"Expected props to have keys {store._props.keys()} but "
                f"only found {props.keys()}")

        fields = {
            name[len("fields."):]: arr
            for name, arr in d.items()
            if name.startswith("fields.")
        }

        store._props = props
        store._fields = fields

        return store
