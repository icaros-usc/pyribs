"""Provides ArchiveBase."""
from abc import ABC, abstractmethod
from collections import OrderedDict

import numba as nb
import numpy as np
import pandas as pd
from decorator import decorator

from ribs.archives._add_status import AddStatus


@decorator
def require_init(method, self, *args, **kwargs):
    """Decorator for archive methods that forces the archive to be initialized.

    If the archive is not initialized (according to the ``initialized``
    property), a RuntimeError is raised.
    """
    if not self.initialized:
        raise RuntimeError("Archive has not been initialized. "
                           "Please call initialize().")
    return method(self, *args, **kwargs)


class RandomBuffer:
    """An internal class that stores a buffer of random numbers.

    Generating random indices in get_random_elite() takes a lot of time if done
    individually. As such, this class generates many random numbers at once and
    slowly dispenses them. Since the calls in get_random_elite() vary in their
    range, this class does not store random integers; it stores random floats in
    the range [0,1) that can be multiplied to get a number in the range [0, x).

    Args:
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        buf_size (int): How many random floats to store at once in the buffer.
    """

    def __init__(self, seed=None, buf_size=10):
        assert buf_size > 0, "buf_size must be at least 1"

        self._rng = np.random.default_rng(seed)
        self._buf_size = buf_size
        self._buffer = self._rng.random(buf_size)
        self._buf_idx = 0

    def get(self, max_val):
        """Returns a random int in the range [0, max_val)."""
        val = int(self._buffer[self._buf_idx] * max_val)
        self._buf_idx += 1

        # Reset the buffer if necessary.
        if self._buf_idx >= self._buf_size:
            self._buf_idx = 0
            self._buffer = self._rng.random(self._buf_size)

        return val


class ArchiveBase(ABC):  # pylint: disable = too-many-instance-attributes
    """Base class for archives.

    This class assumes all archives use a fixed-size container with bins that
    hold (1) information about whether the bin is occupied (bool), (2) a
    solution (1D array), (3) objective function evaluation of the solution
    (float), (4) behavior space coordinates of the solution (1D array), and (5)
    any additional metadata associated with the solution (object). In this
    class, the container is implemented with separate numpy arrays that share
    common dimensions. Using the ``storage_dims`` and ``behavior_dim`` arguments
    in :meth:`__init__` and the ``solution_dim`` argument in ``initialize``,
    these arrays are as follows:

    +------------------------+------------------------------------+
    | Name                   |  Shape                             |
    +========================+====================================+
    | ``_occupied``          |  ``(*storage_dims)``               |
    +------------------------+------------------------------------+
    | ``_solutions``         |  ``(*storage_dims, solution_dim)`` |
    +------------------------+------------------------------------+
    | ``_objective_values``  |  ``(*storage_dims)``               |
    +------------------------+------------------------------------+
    | ``_behavior_values``   |  ``(*storage_dims, behavior_dim)`` |
    +------------------------+------------------------------------+
    | ``_metadata``          |  ``(*storage_dims)``               |
    +------------------------+------------------------------------+

    All of these arrays are accessed via a common index. If we have index ``i``,
    we access its solution at ``_solutions[i]``, its behavior values at
    ``_behavior_values[i]``, etc.

    Thus, child classes typically override the following methods:

    - :meth:`__init__`: Child classes must invoke this class's :meth:`__init__`
      with the appropriate arguments.
    - :meth:`get_index`: Returns an index into the arrays above when given the
      behavior values of a solution. Usually, the index has a meaning, e.g. in
      :class:`~ribs.archives.CVTArchive` it is the index of a centroid. This
      method should include an explanation of what the index means.
    - :meth:`initialize`: By default, this method sets up the arrays described,
      so child classes should invoke the parent implementation if they are
      overriding it.

    .. note:: Attributes beginning with an underscore are only intended to be
        accessed by child classes (i.e. they are "protected" attributes).

    Args:
        storage_dims (tuple of int): Primary dimensions of the archive storage.
            This is used to create the numpy arrays described above.
        behavior_dim (int): The dimension of the behavior space.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / ``np.float32`` and
            ``"d"`` / ``np.float64``.
    Attributes:
        _rng (numpy.random.Generator): Random number generator, used in
            particular for generating random elites.
        _storage_dims (tuple of int): See ``storage_dims`` arg.
        _behavior_dim (int): See ``behavior_dim`` arg.
        _solution_dim (int): Dimension of the solution space, passed in with
            :meth:`initialize`.
        _occupied (numpy.ndarray): Bool array storing whether each cell in the
            archive is occupied. This attribute is None until :meth:`initialize`
            is called.
        _solutions (numpy.ndarray): Float array storing the solutions
            themselves. This attribute is None until :meth:`initialize` is
            called.
        _objective_values (numpy.ndarray): Float array storing the objective
            value of each solution. This attribute is None until
            :meth:`initialize` is called.
        _behavior_values (numpy.ndarray): Float array storing the behavior
            space coordinates of each solution. This attribute is None until
            :meth:`initialize` is called.
        _metadata (numpy.ndarray): Object array storing the metadata associated
            with each solution. This attribute is None until :meth:`initialize`
            is called.
        _occupied_indices (list of (int or tuple of int)): A list of indices
            that are occupied in the archive. This attribute is None until
            :meth:`initialize` is called.
        _occupied_indices_cols (tuple of list of int): Stores the same data as
            ``_occupied_indices``, but in column-wise fashion. For instance,
            ``_occupied_indices_cols[0]`` holds entry 0 of all the indices in
            ``_occupied_indices``. This attribute is None until
            :meth:`initialize` is called.
    """

    def __init__(self, storage_dims, behavior_dim, seed=None, dtype=np.float64):
        # Intended to be accessed by child classes.
        self._rng = np.random.default_rng(seed)
        self._storage_dims = storage_dims
        self._behavior_dim = behavior_dim
        self._solution_dim = None
        self._occupied = None
        self._solutions = None
        self._objective_values = None
        self._behavior_values = None
        self._metadata = None
        self._occupied_indices = None
        self._occupied_indices_cols = None

        # Not intended to be accessed by children (and thus not mentioned in the
        # docstring).
        self._rand_buf = None
        self._seed = seed
        self._initialized = False
        self._bins = np.product(self._storage_dims)

        self._dtype = self._parse_dtype(dtype)

    @staticmethod
    def _parse_dtype(dtype):
        """Parses the dtype passed into the constructor.

        Returns:
            np.float32 or np.float64
        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        # First convert str dtype's to np.dtype.
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # np.dtype is not np.float32 or np.float64, but it compares equal.
        if dtype == np.float32:
            return np.float32
        if dtype == np.float64:
            return np.float64

        raise ValueError("Unsupported dtype. Must be np.float32 or np.float64")

    @property
    def initialized(self):
        """Whether the archive has been initialized by a call to
        :meth:`initialize`"""
        return self._initialized

    @property
    def empty(self):
        """bool: Whether the archive is empty."""
        return not self._occupied_indices

    @property
    def bins(self):
        """int: Total number of bins in the archive."""
        return self._bins

    @property
    def behavior_dim(self):
        """int: Dimensionality of the behavior space."""
        return self._behavior_dim

    @property
    @require_init
    def solution_dim(self):
        """int: Dimensionality of the solutions in the archive."""
        return self._solution_dim

    @property
    def dtype(self):
        """data-type: The dtype of the solutions, objective values, and behavior
        values."""
        return self._dtype

    def initialize(self, solution_dim):
        """Initializes the archive by allocating storage space.

        Child classes should call this method in their implementation if they
        are overriding it.

        Args:
            solution_dim (int): The dimension of the solution space.
        Raises:
            RuntimeError: The archive is already initialized.
        """
        if self._initialized:
            raise RuntimeError("Cannot re-initialize an archive")
        self._initialized = True

        self._rand_buf = RandomBuffer(self._seed)
        self._solution_dim = solution_dim
        self._occupied = np.zeros(self._storage_dims, dtype=bool)
        self._objective_values = np.empty(self._storage_dims, dtype=self.dtype)
        self._behavior_values = np.empty(
            (*self._storage_dims, self._behavior_dim), dtype=self.dtype)
        self._solutions = np.empty((*self._storage_dims, solution_dim),
                                   dtype=self.dtype)
        self._metadata = np.empty(self._storage_dims, dtype=object)
        self._occupied_indices = []
        self._occupied_indices_cols = tuple(
            [] for _ in range(len(self._storage_dims)))

    @abstractmethod
    def get_index(self, behavior_values):
        """Returns archive indices for the given behavior values.

        See the main :class:`~ribs.archives.ArchiveBase` docstring for more
        info.

        Args:
            behavior_values (numpy.ndarray): (:attr:`behavior_dim`,) array of
                coordinates in behavior space.
        Returns:
            int or tuple of int: Indices of the entry in the archive's storage
            arrays.
        """

    @staticmethod
    @nb.jit(locals={"already_occupied": nb.types.b1}, nopython=True)
    def _add_numba(new_index, new_solution, new_objective_value,
                   new_behavior_values, occupied, solutions, objective_values,
                   behavior_values):
        """Numba helper for inserting solutions into the archive.

        See add() for usage.

        Returns:
            was_inserted (bool): Whether the new values were inserted into the
                archive.
            already_occupied (bool): Whether the index was occupied prior
                to this call; i.e. this is True only if there was already an
                item at the index.
        """
        already_occupied = occupied[new_index]
        if (not already_occupied or
                objective_values[new_index] < new_objective_value):
            # Track this index if it has not been seen before -- important that
            # we do this before inserting the solution.
            if not already_occupied:
                occupied[new_index] = True

            # Insert into the archive.
            objective_values[new_index] = new_objective_value
            behavior_values[new_index] = new_behavior_values
            solutions[new_index] = new_solution

            return True, already_occupied

        return False, already_occupied

    def _add_occupied_index(self, index):
        """Adds a new index to the lists of occupied indices."""
        self._occupied_indices.append(index)

        # Some archives (e.g. CVTArchive) have a 1D index and use ints instead
        # of tuples, so we convert to a singleton tuple here.
        if not isinstance(index, tuple):
            index = (index,)

        for i, idx in enumerate(index):
            self._occupied_indices_cols[i].append(idx)

    @require_init
    def add(self, solution, objective_value, behavior_values, metadata=None):
        """Attempts to insert a new solution into the archive.

        The solution is only inserted if it has a higher ``objective_value``
        than the solution previously in the corresponding bin.

        Args:
            solution (array-like): Parameters of the solution.
            objective_value (float): Objective function evaluation of the
                solution.
            behavior_values (array-like): Coordinates in behavior space of the
                solution.
            metadata (object): Any Python object representing metadata for the
                solution. For instance, this could be a dict with several
                properties.
        Returns:
            tuple: 2-element tuple describing the result of the add operation.
            These outputs are particularly useful for algorithms such as CMA-ME.

                **status** (:class:`AddStatus`): See :class:`AddStatus`.

                **value** (:attr:`dtype`): The meaning of this value depends on
                the value of ``status``:

                - ``NOT_ADDED`` -> the "negative improvement," i.e. objective
                  value of solution passed in minus objective value of the
                  solution still in the archive (this value is negative because
                  the solution did not have a high enough objective value to be
                  added to the archive)
                - ``IMPROVE_EXISTING`` -> the "improvement," i.e. objective
                  value of solution passed in minus objective value of solution
                  previously in the archive
                - ``NEW`` -> the objective value passed in
        """
        solution = np.asarray(solution)
        behavior_values = np.asarray(behavior_values)

        index = self.get_index(behavior_values)
        old_objective = self._objective_values[index]
        was_inserted, already_occupied = self._add_numba(
            index, solution, objective_value, behavior_values, self._occupied,
            self._solutions, self._objective_values, self._behavior_values)

        if was_inserted:
            self._metadata[index] = metadata

        if was_inserted and not already_occupied:
            self._add_occupied_index(index)
            status = AddStatus.NEW
            value = objective_value
        elif was_inserted and already_occupied:
            status = AddStatus.IMPROVE_EXISTING
            value = objective_value - old_objective
        else:
            status = AddStatus.NOT_ADDED
            value = objective_value - old_objective
        return status, self.dtype(value)

    @require_init
    def elite_with_behavior(self, behavior_values):
        """Gets the elite with behavior vals in the same bin as those specified.

        .. note:: Values like ``index`` and ``metadata`` are often not used.
            In such cases, consider ignoring them::

                sol, obj, beh, *_ = archive.elite_with_behavior(...)

        Args:
            behavior_values (array-like): Coordinates in behavior space.
        Returns:
            tuple: 5-element tuple for the elite if it is found:

                **solution** (:class:`numpy.ndarray`): Parameters for the
                solution.

                **objective_value** (:attr:`dtype`): Objective function
                evaluation.

                **behavior_values** (:class:`numpy.ndarray`): Actual behavior
                space coordinates of the elite (may not be exactly the same as
                those specified).

                **index** (int or tuple of int): Index of the entry in the
                archive. See :attr:`get_index` for more info.

                **metadata** (object): Metadata for the solution.

            If there is no elite in the bin, each of the above values is None.
            Thus, something like
            ``sol, obj, beh, idx, meta = archive.elite_with_behavior(...)``
            still works).
        """
        index = self.get_index(np.asarray(behavior_values))
        if self._occupied[index]:
            return (
                self._solutions[index],
                self._objective_values[index],
                self._behavior_values[index],
                index,
                self._metadata[index],
            )
        return (None, None, None, None, None)

    @require_init
    def get_random_elite(self):
        """Selects an elite uniformly at random from one of the archive's bins.

        .. note:: Values like ``index`` and ``metadata`` are often not used.
            In such cases, consider ignoring them::

                sol, obj, beh, *_ = archive.get_random_elite()

        Returns:
            tuple: 5-element tuple containing:

                **solution** (:class:`numpy.ndarray`): Parameters for the
                solution.

                **objective_value** (:attr:`dtype`): Objective function
                evaluation.

                **behavior_values** (:class:`numpy.ndarray`): Behavior space
                coordinates.

                **index** (int or tuple of int): Index of the entry in the
                archive. See :attr:`get_index` for more info.

                **metadata** (object): Metadata for the solution.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        random_idx = self._rand_buf.get(len(self._occupied_indices))
        index = self._occupied_indices[random_idx]

        return (
            self._solutions[index],
            self._objective_values[index],
            self._behavior_values[index],
            index,
            self._metadata[index],
        )

    def data(self):
        """Returns columns containing all data in the archive.

        Namely, this method returns 5 arrays containing all of the solutions,
        objective values, behavior values, indices, and metadata in the archive.
        For example::

            (all_solutions, all_objective_values,
             all_behavior_values, all_indices, all_metadata) = archive.data()

        All the arrays correspond to each other, i.e. ``all_solutions[i]``
        corresponds to ``all_objective_values[i]``, ``all_behavior_values[i]``,
        ``all_indices[i]``, and ``all_metadata[i]``. This means that an
        iteration like the following would work::

            for sol, obj, beh, idx, meta in zip(*archive.data()):
                ...

        This method is also useful when extracting insights from one
        component of the archive. For instance, one can easily extract all the
        objective values and calculate their mean with this method.

        .. note:: This method returns numpy views into existing data in the
            archive, so it does not make any copies. However, the data may
            change if the archive is modified after calling this method.

        Returns:
            tuple: 5-element tuple containing:

                **all_solutions** (:class:`numpy.ndarray` -- shape (n_entries,
                :attr:`solution_dim`)): Parameters for all the solutions in the
                archive.

                **all_objective_values** (:class:`numpy.ndarray` -- shape
                (n_entries,)): Objective value of all entries in the archive.

                **all_behavior_values** (:class:`numpy.ndarray`-- shape
                (n_entries, :attr:`behavior_dim`)): Behavior space coordinates
                of all the entries.

                **all_indices** (:class:`list` -- shape (n_entries,)): Index of
                all entries in the archive. As each index can be either an int
                or a tuple, this is a Python list. See :meth:`get_index` for
                more info.

                **all_metadata** (:class:`numpy.ndarray` -- shape (n_entries,)):
                Object array with metadata of all entries.
        """
        return (
            self._solutions[self._occupied_indices_cols],
            self._objective_values[self._occupied_indices_cols],
            self._behavior_values[self._occupied_indices_cols],
            self._occupied_indices,
            self._metadata[self._occupied_indices_cols],
        )

    def as_pandas(self, include_solutions=True, include_metadata=False):
        """Converts the archive into a Pandas dataframe.

        This base class implementation creates a dataframe consisting of:

        - ``len(self._storage_dims)`` columns for the index, named
          ``index_0, index_1, ...`` In :class:`~ribs.archives.GridArchive` and
          :class:`~ribs.archives.SlidingBoundariesArchive`, there are
          :attr:`behavior_dim` columns. In :class:`~ribs.archives.CVTArchive`,
          there is just one column. See :meth:`get_index` for more info.
        - ``self._behavior_dim`` columns for the behavior characteristics, named
          ``behavior_0, behavior_1, ...``
        - 1 column for the objective values, named ``objective``
        - ``solution_dim`` columns for the solution vectors, named
          ``solution_0, solution_1, ...``
        - 1 column for the metadata objects, named ``metadata``

        In short, the dataframe looks like this:

        +---------+------+-------------+------+------------+-------------+-----+----------+
        | index_0 | ...  | behavior_0  | ...  | objective  | solution_0  | ... | metadata |
        +=========+======+=============+======+============+=============+=====+==========+
        |         | ...  |             | ...  |            |             | ... |          |
        +---------+------+-------------+------+------------+-------------+-----+----------+

        Args:
            include_solutions (bool): Whether to include solution columns.
            include_metadata (bool): Whether to include the metadata column.
                Note that methods like :meth:`~pandas.DataFrame.to_csv` may not
                properly save the dataframe since the metadata objects may not
                be representable in a CSV.
        Returns:
            pandas.DataFrame: See above.
        """ # pylint: disable = line-too-long
        data = OrderedDict()

        index_dim = len(self._storage_dims)
        for i in range(index_dim):
            data[f"index_{i}"] = np.asarray(self._occupied_indices_cols[i],
                                            dtype=int)

        behavior_values = self._behavior_values[self._occupied_indices_cols]
        for i in range(self._behavior_dim):
            data[f"behavior_{i}"] = np.asarray(behavior_values[:, i],
                                               dtype=self.dtype)

        data["objective"] = np.asarray(
            self._objective_values[self._occupied_indices_cols],
            dtype=self.dtype)

        if include_solutions:
            solutions = self._solutions[self._occupied_indices_cols]
            for i in range(self._solution_dim):
                data[f"solution_{i}"] = np.asarray(solutions[:, i],
                                                   dtype=self.dtype)

        if include_metadata:
            metadata = self._metadata[self._occupied_indices_cols]
            data["metadata"] = np.asarray(metadata, dtype=object)

        return pd.DataFrame(data)
