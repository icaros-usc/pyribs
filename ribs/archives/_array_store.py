"""Provides ArrayStore."""
import pickle as pkl
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import readonly

_FORMATS = ["npz", "npz_compressed", "pkl"]


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
        return readonly(self._props["occupied"].view())

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

    # TODO: Add cur_add_info
    def add(self, indices, new_data, transforms):
        """Adds new data to the archive at the given indices.

        The indices and new_data are passed through transforms before adding to
        the archive. The general idea is that these transforms will gradually
        modify the indices and new_data. For instance, they can add new fields
        to new_data (new_data may not initially have all the same fields as the
        archive). Alternatively, they can filter out duplicate indices, eg if
        multiple entries are being inserted at the same index we can choose one
        with the best objective.

        The signature of a transform is as follows:

            def transform(indices, new_data, add_info, occupied, cur_data) ->
                (indices, new_data, add_info):

        Transform parameters:

            * **indices** (array-like): Array of indices at which new_data
              should be inserted.
            * **new_data** (dict): New data for the given indices. Maps from
              field name to the array of new data for that field.
            * **add_info** (dict): Information to return to the user about the
              addition process. Example info includes whether each entry was
              ultimately inserted into the archive, as well as general
              statistics like update QD score. For the first transform, this
              will be an empty dict.
            * **occupied** (array-like): Whether the given indices are currently
              occupied. Same as that given by :meth:`retrieve`.
            * **cur_data** (dict): Data at the current indices in the archive.
              Same as that given by :meth:`retrieve`.

        Transform outputs:

            * **indices** (array-like): Modified indices.
            * **new_data** (dict): Modified new_data. At the end of the
              transforms, it should have the same keys as the store.
            * **add_info** (dict): Modified add_info.

        Args:
            indices (array-like): Initial list of indices for addition.
            new_data (dict): Initial data for addition.
            transforms (list): List of transforms on the data to be added.
        Raise:
            ValueError: The final version of ``new_data`` does not have the same
                keys as the fields of this store.
            ValueError: The final version of ``new_data`` has fields that have a
                different length than ``indices``.
        """
        add_info = {}
        for transform in transforms:
            occupied, cur_data = self.retrieve(indices)
            indices, new_data, add_info = transform(indices, new_data, add_info,
                                                    occupied, cur_data)

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

        old_capacity = self._props["capacity"]
        self._props["capacity"] = capacity

        old_occupied = self._props["occupied"]
        self._props["occupied"] = np.zeros(capacity, dtype=bool)
        self._props["occupied"][:old_capacity] = old_occupied

        old_occupied_list = self._props["occupied_list"]
        self._props["occupied_list"] = np.empty(capacity, dtype=int)
        self._props["occupied_list"][:old_capacity] = old_occupied_list

        for name, old_arr in self._fields.items():
            new_shape = (capacity,) + old_arr.shape[1:]
            self._fields[name] = np.empty(new_shape, old_arr.dtype)
            self._fields[name][:old_capacity] = old_arr

    def as_dict(self):
        """Returns the data in the ArrayStore as a one-level dictionary.

        To collapse the dict, we prefix each key with ``props.`` or ``fields.``,
        so the result looks as follows::

            {
              "props.capacity": ...,
              "props.occupied": ...,
              ...
              "fields.objective": ...,
            }

        Returns:
            dict: See description above.
        """
        d = {}
        for name, prop in self._props.items():
            if isinstance(prop, np.ndarray):
                prop = readonly(prop.view())
            d[f"props.{name}"] = prop
        for name, arr in self._fields.items():
            if isinstance(arr, np.ndarray):
                arr = readonly(arr.view())
            d[f"fields.{name}"] = arr
        return d

    def save(self, file, fmt=None):
        """Saves the store to a given file.

        Supported formats are:

        * `"npz"`: Saves to the `.npz` file format with :func:`numpy.savez`
        * `"npz_compressed"`: Saves to a compressed `.npz` file format with
          :func:`numpy.savez_compressed`
        * `"pkl"`: Saves to a pickle file with :func:`pickle.dump`

        .. note::

            Internally, this method calls :meth:`as_dict` and saves the
            resulting dictionary to a file. If you need a format that is not
            supported here, you can save the dict from :meth:`as_dict`. To
            reload ArrayStore, load the dict from your format and then pass the
            dict into :meth:`load`.

        Args:
            file (str, pathlib.Path, file): Filename or file object for saving
                the data. We do not modify the filename to include the
                extension.
            fmt (str): File format for saving the data.
        Raises:
            ValueError: Unsupported format.
        """
        d = self.as_dict()

        if fmt == "npz":
            np.savez(file, **d)
        elif fmt == "npz_compressed":
            np.savez_compressed(file, **d)
        elif fmt == "pkl":
            with (open(file, "wb") if isinstance(file, (str, Path)) else
                  nullcontext(file)) as file_obj:
                pkl.dump(d, file_obj)
        else:
            raise ValueError(f"Unsupported value `{fmt}` for fmt. Must be "
                             f"one of {_FORMATS}")

    @staticmethod
    def load(file, fmt=None, allow_pickle=False):
        """Loads the ArrayStore from a dict or file.

        Args:
            file (dict, str, pathlib.Path, file): Data to load. Either a dict
                like that output by :meth:`as_dict`; a path to a file; or a file
                object. In the case of a file object, ``fmt`` must be passed
                (see :meth:`save` for supported formats).
            fmt (str): Format for the file. If not passed in, we will infer the
                format from the extension of ``file``.
            allow_pickle (bool): Only applicable if using ``npz`` or
                ``npz_compressed`` format and the store contains object arrays.
                In this case, pickle is necessary since the object arrays are
                saved with pickle (see :meth:`numpy.load` for more info).
        Raises:
            ValueError: Could not infer ``fmt`` from ``file`` as there is no
                extension.
            ValueError: The loaded props dict has the wrong keys.
        """

        if isinstance(file, dict):
            data = file
        else:
            # Load dict from file.

            if isinstance(file, (str, Path)):
                file = Path(file)
                if fmt is None:
                    fmt = file.suffix[1:]
                    if fmt == "":
                        raise ValueError(
                            f"Could not infer fmt from file `{file}`. Please "
                            "pass the fmt arg.")

            # Now file is either a Path or a file-like object.

            if fmt in ["npz", "npz_compressed"]:
                data = dict(np.load(file, allow_pickle=allow_pickle))
            elif fmt == "pkl":
                with (open(file, "rb") if isinstance(file, (str, Path)) else
                      nullcontext(file)) as file_obj:
                    data = pkl.load(file_obj)

        # Load the store. Here, we create a store with no data in it.
        # pylint: disable = protected-access
        store = ArrayStore({}, 0)

        props = {
            name[len("props."):]: arr
            for name, arr in data.items()
            if name.startswith("props.")
        }
        if props.keys() != store._props.keys():
            raise ValueError(
                f"Expected props to have keys {store._props.keys()} but "
                f"only found {props.keys()}")

        fields = {
            name[len("fields."):]: arr
            for name, arr in data.items()
            if name.startswith("fields.")
        }

        store._props = props
        store._fields = fields

        return store
