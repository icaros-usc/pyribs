"""Provides ArchiveBase."""
from abc import ABC, abstractmethod

import numpy as np

from ribs._utils import (check_batch_shape, check_finite, check_is_1d,
                         check_shape, parse_float_dtype, validate_batch,
                         validate_single)
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._cqd_score_result import CQDScoreResult
from ribs.archives._transforms import (batch_entries_with_threshold,
                                       compute_best_index,
                                       compute_objective_sum,
                                       single_entry_with_threshold)

_ARCHIVE_FIELDS = {"index", "solution", "objective", "measures", "threshold"}


class ArchiveBase(ABC):
    # pylint: disable = too-many-instance-attributes, too-many-public-methods
    """Base class for archives.

    This class composes archives using an :class:`ArrayStore` that has
    "solution", "objective", "measures", and "threshold" fields.

    Child classes typically override the following methods:

    - ``__init__``: Child classes must invoke this class's ``__init__`` with the
      appropriate arguments.
    - :meth:`index_of`: Returns integer indices into the arrays above when
      given a batch of measures. Usually, each index has a meaning, e.g. in
      :class:`~ribs.archives.CVTArchive` it is the index of a centroid.
      Documentation for this method should describe the meaning of the index.

    .. note:: Attributes beginning with an underscore are only intended to be
        accessed by child classes (i.e. they are "protected" attributes).

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2022
        <https://arxiv.org/abs/2205.10752>`_. Refer to our `CMA-MAE tutorial
        <../../tutorials/cma_mae.html>`_ for more info on thresholds, including
        the ``learning_rate`` and ``threshold_min`` parameters.

    Args:
        solution_dim (int): Dimension of the solution space.
        cells (int): Number of cells in the archive. This is used to create the
            numpy arrays described above for storing archive info.
        measure_dim (int): The dimension of the measure space.
        learning_rate (float): The learning rate for threshold updates. Defaults
            to 1.0.
        threshold_min (float): The initial threshold value for all the cells.
        qd_score_offset (float): Archives often contain negative objective
            values, and if the QD score were to be computed with these negative
            objectives, the algorithm would be penalized for adding new cells
            with negative objectives. Thus, a standard practice is to normalize
            all the objectives so that they are non-negative by introducing an
            offset. This QD score offset will be *subtracted* from all
            objectives in the archive, e.g., if your objectives go as low as
            -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
        extra_fields (dict): Description of extra fields of data that is stored
            next to elite data like solutions and objectives. The description is
            a dict mapping from a field name (str) to a tuple of ``(shape,
            dtype)``. For instance, ``{"foo": ((), np.float32), "bar": ((10,),
            np.float32)}`` will create a "foo" field that contains scalar values
            and a "bar" field that contains 10D values. Note that field names
            must be valid Python identifiers, and names already used in the
            archive are not allowed.

    Attributes:
        _rng (numpy.random.Generator): Random number generator, used in
            particular for generating random elites.
        _store (ribs.archives.ArrayStore): The underlying ArrayStore containing
            data for the archive.

    Raises:
        ValueError: Invalid values for learning_rate and threshold_min.
        ValueError: Invalid names in extra_fields.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 cells,
                 measure_dim,
                 learning_rate=None,
                 threshold_min=-np.inf,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64,
                 extra_fields=None):

        self._dtype = parse_float_dtype(dtype)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._cells = cells
        self._solution_dim = solution_dim
        self._measure_dim = measure_dim
        self._qd_score_offset = self._dtype(qd_score_offset)

        if threshold_min != -np.inf and learning_rate is None:
            raise ValueError(
                "You set threshold_min without setting learning_rate. "
                "Please note that threshold_min is only used in CMA-MAE; "
                "it is not intended to be used for only filtering archive "
                "solutions. If you would like to run CMA-MAE, please also set "
                "learning_rate.")
        if learning_rate is None:
            learning_rate = 1.0  # Default value.
        if threshold_min == -np.inf and learning_rate != 1.0:
            raise ValueError("threshold_min can only be -np.inf if "
                             "learning_rate is 1.0")
        self._learning_rate = self._dtype(learning_rate)
        self._threshold_min = self._dtype(threshold_min)

        self._stats = None
        self._best_elite = None
        # Sum of all objective values in the archive; useful for computing
        # qd_score and obj_mean.
        self._objective_sum = None
        self._stats_reset()

        extra_fields = extra_fields or {}
        if _ARCHIVE_FIELDS & extra_fields.keys():
            raise ValueError("The following names are not allowed in "
                             f"extra_fields: {_ARCHIVE_FIELDS}")

        self._store = ArrayStore(
            field_desc={
                "solution": ((solution_dim,), self.dtype),
                "objective": ((), self.dtype),
                "measures": ((measure_dim,), self.dtype),
                "threshold": ((), self.dtype),
                **extra_fields,
            },
            capacity=self._cells,
        )

    @property
    def field_list(self):
        """list: List of data fields in the archive."""
        return self._store.field_list

    @property
    def cells(self):
        """int: Total number of cells in the archive."""
        return self._cells

    @property
    def measure_dim(self):
        """int: Dimensionality of the measure space."""
        return self._measure_dim

    @property
    def solution_dim(self):
        """int: Dimensionality of the solutions in the archive."""
        return self._solution_dim

    @property
    def learning_rate(self):
        """float: The learning rate for threshold updates."""
        return self._learning_rate

    @property
    def threshold_min(self):
        """float: The initial threshold value for all the cells."""
        return self._threshold_min

    @property
    def qd_score_offset(self):
        """float: The offset which is subtracted from objective values when
        computing the QD score."""
        return self._qd_score_offset

    @property
    def stats(self):
        """:class:`ArchiveStats`: Statistics about the archive.

        See :class:`ArchiveStats` for more info.
        """
        return self._stats

    @property
    def best_elite(self):
        """dict: The elite with the highest objective in the archive.

        None if there are no elites in the archive.

        .. note::
            If the archive is non-elitist (this occurs when using the archive
            with a learning rate which is not 1.0, as in CMA-MAE), then this
            best elite may no longer exist in the archive because it was
            replaced with an elite with a lower objective value. This can happen
            because in non-elitist archives, new solutions only need to exceed
            the *threshold* of the cell they are being inserted into, not the
            *objective* of the elite currently in the cell. See :pr:`314` for
            more info.

        .. note::
            The best elite will contain a "threshold" key. This threshold is the
            threshold of the best elite's cell after the best elite was inserted
            into the archive.
        """
        return self._best_elite

    @property
    def dtype(self):
        """data-type: The dtype of the solutions, objective, and measures."""
        return self._dtype

    @property
    def empty(self):
        """bool: Whether the archive is empty."""
        return len(self._store) == 0

    def __len__(self):
        """Number of elites in the archive."""
        return len(self._store)

    def __iter__(self):
        """Creates an iterator over the elites in the archive.

        Example:

            ::

                for elite in archive:
                    elite["solution"]
                    elite["objective"]
                    ...
        """
        return iter(self._store)

    def clear(self):
        """Removes all elites from the archive.

        After this method is called, the archive will be :attr:`empty`.
        """
        self._store.clear()
        self._stats_reset()

    @abstractmethod
    def index_of(self, measures):
        """Returns archive indices for the given batch of measures.

        If you need to retrieve the index of the measures for a *single*
        solution, consider using :meth:`index_of_single`.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            (numpy.ndarray): (batch_size,) array with the indices of the
            batch of measures in the archive's storage arrays.
        """

    def index_of_single(self, measures):
        """Returns the index of the measures for one solution.

        While :meth:`index_of` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution. If :meth:`index_of` is
        implemented correctly, this method should work immediately (i.e. `"out
        of the box" <https://idioms.thefreedictionary.com/Out-of-the-Box>`_).

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures for
                a single solution.
        Returns:
            int or numpy.integer: Integer index of the measures in the archive's
            storage arrays.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    def _stats_reset(self):
        """Resets the archive stats."""
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=self.dtype(0.0),
            qd_score=self.dtype(0.0),
            norm_qd_score=self.dtype(0.0),
            obj_max=None,
            obj_mean=None,
        )
        self._best_elite = None
        self._objective_sum = self.dtype(0.0)

    def _stats_update(self, new_objective_sum, new_best_index):
        """Updates statistics based on a new sum of objective values
        (new_objective_sum) and the index of a potential new best elite
        (new_best_index)."""
        self._objective_sum = new_objective_sum
        new_qd_score = (self._objective_sum -
                        self.dtype(len(self)) * self._qd_score_offset)

        _, new_best_elite = self._store.retrieve([new_best_index])

        if (self._stats.obj_max is None or
                new_best_elite["objective"] > self._stats.obj_max):
            # Convert batched values to single values.
            new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

            new_obj_max = new_best_elite["objective"]
            self._best_elite = new_best_elite
        else:
            new_obj_max = self._stats.obj_max

        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=self.dtype(new_qd_score / self.cells),
            obj_max=new_obj_max,
            obj_mean=self._objective_sum / self.dtype(len(self)),
        )

    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        Each solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of
        ``learning_rate`` and ``threshold_min``, this threshold is simply the
        objective value of the elite previously in the cell.  If multiple
        solutions in the batch end up in the same cell, we only insert the
        solution with the highest objective. If multiple solutions end up in the
        same cell and tie for the highest objective, we insert the solution that
        appears first in the batch.

        For the default values of ``learning_rate`` and ``threshold_min``, the
        threshold for each cell is updated by taking the maximum objective value
        among all the solutions that landed in the cell, resulting in the same
        behavior as in the vanilla MAP-Elites archive. However, for other
        settings, the threshold is updated with the batch update rule described
        in the appendix of `Fontaine 2022 <https://arxiv.org/abs/2205.10752>`_.

        .. note:: The indices of all arguments should "correspond" to each
            other, i.e. ``solution[i]``, ``objective[i]``,
            ``measures[i]``, and should be the solution parameters,
            objective, and measures for solution ``i``.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solution parameters.
            objective (array-like): (batch_size,) array with objective function
                evaluations of the solutions.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with
                measure space coordinates of all the solutions.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`int`): An array of
              integers that represent the "status" obtained when attempting to
              insert each solution in the batch. Each item has the following
              possible values:

              - ``0``: The solution was not added to the archive.
              - ``1``: The solution improved the objective value of a cell
                which was already in the archive.
              - ``2``: The solution discovered a new cell in the archive.

              All statuses (and values, below) are computed with respect to the
              *current* archive. For example, if two solutions both introduce
              the same new archive cell, then both will be marked with ``2``.

              The alternative is to depend on the order of the solutions in the
              batch -- for example, if we have two solutions ``a`` and ``b``
              which introduce the same new cell in the archive, ``a`` could be
              inserted first with status ``2``, and ``b`` could be inserted
              second with status ``1`` because it improves upon ``a``. However,
              our implementation does **not** do this.

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus` e.g. with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"value"`` (:class:`numpy.ndarray` of :attr:`dtype`): An array
              with values for each solution in the batch. With the default
              values of ``learning_rate = 1.0`` and ``threshold_min = -np.inf``,
              the meaning of each value depends on the corresponding ``status``
              and is identical to that in CMA-ME (`Fontaine 2020
              <https://arxiv.org/abs/1912.02400>`_):

              - ``0`` (not added): The value is the "negative improvement," i.e.
                the objective of the solution passed in minus the objective of
                the elite still in the archive (this value is negative because
                the solution did not have a high enough objective to be added to
                the archive).
              - ``1`` (improve existing cell): The value is the "improvement,"
                i.e. the objective of the solution passed in minus the objective
                of the elite previously in the archive.
              - ``2`` (new cell): The value is just the objective of the
                solution.

              In contrast, for other values of ``learning_rate`` and
              ``threshold_min``, each value is equivalent to the objective value
              of the solution minus the threshold of its corresponding cell in
              the archive.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf
                or NaN).
        """
        data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        add_info = self._store.add(
            self.index_of(data["measures"]),
            data,
            {
                "dtype": self._dtype,
                "learning_rate": self._learning_rate,
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            [
                batch_entries_with_threshold,
                compute_objective_sum,
                compute_best_index,
            ],
        )

        objective_sum = add_info.pop("objective_sum")
        best_index = add_info.pop("best_index")
        if not np.all(add_info["status"] == 0):
            self._stats_update(objective_sum, best_index)

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        The solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of
        ``learning_rate`` and ``threshold_min``, this threshold is simply the
        objective value of the elite previously in the cell.  The threshold is
        also updated if the solution was inserted.

        .. note::
            To make it more amenable to modifications, this method's
            implementation is designed to be readable at the cost of
            performance, e.g., none of its operations are modified. If you need
            performance, we recommend using :meth:`add`.

        Args:
            solution (array-like): Parameters of the solution.
            objective (float): Objective function evaluation of the solution.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains ``status`` and ``value`` keys; refer to :meth:`add`
            for the meaning of status and value.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures``
                has non-finite values.
        """
        data = validate_single(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        for name, arr in data.items():
            data[name] = np.expand_dims(arr, axis=0)

        add_info = self._store.add(
            np.expand_dims(self.index_of_single(measures), axis=0),
            data,
            {
                "dtype": self._dtype,
                "learning_rate": self._learning_rate,
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            [
                single_entry_with_threshold,
                compute_objective_sum,
                compute_best_index,
            ],
        )

        objective_sum = add_info.pop("objective_sum")
        best_index = add_info.pop("best_index")

        for name, arr in add_info.items():
            add_info[name] = arr[0]

        if add_info["status"]:
            self._stats_update(objective_sum, best_index)

        return add_info

    def retrieve(self, measures):
        """Retrieves the elites with measures in the same cells as the measures
        specified.

        This method operates in batch, i.e., it takes in a batch of measures and
        outputs the batched data for the elites::

            occupied, elites = archive.retrieve(...)
            elites["solution"]  # Shape: (batch_size, solution_dim)
            elites["objective"]
            elites["measures"]
            elites["threshold"]
            elites["index"]

        If the cell associated with ``elites["measures"][i]`` has an elite in
        it, then ``occupied[i]`` will be True. Furthermore,
        ``elites["solution"][i]``, ``elites["objective"][i]``,
        ``elites["measures"][i]``, ``elites["threshold"][i]``, and
        ``elites["index"][i]`` will be set to the properties of the elite. Note
        that ``elites["measures"][i]`` may not be equal to the ``measures[i]``
        passed as an argument, since the measures only need to be in the same
        archive cell.

        If the cell associated with ``measures[i]`` *does not* have any elite in
        it, then ``occupied[i]`` will be set to False. Furthermore, the
        corresponding outputs will be set to empty values -- namely:

        * NaN for floating-point fields
        * -1 for the "index" field
        * 0 for integer fields
        * None for object fields

        If you need to retrieve a *single* elite associated with some measures,
        consider using :meth:`retrieve_single`.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            tuple: 2-element tuple of (occupied array, dict). The occupied array
            indicates whether each of the cells indicated by the coordinates in
            ``measures`` has an elite, while the dict contains the data of those
            elites. The dict maps from field name to the corresponding array.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self._store.retrieve(self.index_of(measures))
        unoccupied = ~occupied

        for name, arr in data.items():
            if arr.dtype == object:
                fill_val = None
            elif name == "index":
                fill_val = -1
            elif np.issubdtype(arr.dtype, np.integer):
                fill_val = 0
            else:  # Floating-point and other fields.
                fill_val = np.nan

            arr[unoccupied] = fill_val

        return occupied, data

    def retrieve_single(self, measures):
        """Retrieves the elite with measures in the same cell as the measures
        specified.

        While :meth:`retrieve` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution and returns a single bool and a
        dict with single entries.

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures.
        Returns:
            tuple: If there is an elite with measures in the same cell as the
            measures specified, then this method returns a True value and a dict
            where all the fields hold the info of the elite. Otherwise, this
            method returns a False value and a dict filled with the same "empty"
            values described in :meth:`retrieve`.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])

        return occupied[0], {field: arr[0] for field, arr in data.items()}

    def sample_elites(self, n):
        """Randomly samples elites from the archive.

        Currently, this sampling is done uniformly at random. Furthermore, each
        sample is done independently, so elites may be repeated in the sample.
        Additional sampling methods may be supported in the future.

        Example:

            ::

                elites = archive.sample_elites(16)
                elites["solution"]  # Shape: (16, solution_dim)
                elites["objective"]
                ...

        Args:
            n (int): Number of elites to sample.
        Returns:
            dict: Holds a batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, elites = self._store.retrieve(selected_indices)
        return elites

    def data(self, fields=None, return_type="dict"):
        """Retrieves data for all elites in the archive.

        Args:
            fields (str or array-like of str): List of fields to include. By
                default, all fields will be included, with an additional "index"
                as the last field ("index" can also be placed anywhere in this
                list). This can also be a single str indicating a field name.
            return_type (str): Type of data to return. See below. Ignored if
                ``fields`` is a str.

        Returns:
            The data for all entries in the archive. If ``fields`` was a single
            str, this will just be an array holding data for the given field.
            Otherwise, this data can take the following forms, depending on the
            ``return_type`` argument:

            - ``return_type="dict"``: Dict mapping from the field name to the
              field data at the given indices. An example is::

                  {
                    "solution": [[1.0, 1.0, ...], ...],
                    "objective": [1.5, ...],
                    "measures": [[1.0, 2.0], ...],
                    "threshold": [0.8, ...],
                    "index": [4, ...],
                  }

              Observe that we also return the indices as an ``index`` entry in
              the dict. The keys in this dict can be modified with the
              ``fields`` arg; duplicate fields will be ignored since the dict
              stores unique keys.

            - ``return_type="tuple"``: Tuple of arrays matching the field order
              given in ``fields``. For instance, if ``fields`` was
              ``["objective", "measures"]``, we would receive a tuple of
              ``(objective_arr, measures_arr)``. In this case, the results
              from ``retrieve`` could be unpacked as::

                  objective, measures = archive.data(["objective", "measures"],
                                                     return_type="tuple")

              Unlike with the ``dict`` return type, duplicate fields will show
              up as duplicate entries in the tuple, e.g.,
              ``fields=["objective", "objective"]`` will result in two
              objective arrays being returned.

              By default, (i.e., when ``fields=None``), the fields in the tuple
              will be ordered according to the :attr:`field_list` along with
              ``index`` as the last field.

            - ``return_type="pandas"``: A
              :class:`~ribs.archives.ArchiveDataFrame` with the following
              columns:

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

              In short, the dataframe might look like this by default:

              +------------+------+-----------+------------+------+-----------+-------+
              | solution_0 | ...  | objective | measures_0 | ...  | threshold | index |
              +============+======+===========+============+======+===========+=======+
              |            | ...  |           |            | ...  |           |       |
              +------------+------+-----------+------------+------+-----------+-------+

              Like the other return types, the columns can be adjusted with
              the ``fields`` parameter.

            All data returned by this method will be a copy, i.e., the data will
            not update as the archive changes.
        """ # pylint: disable = line-too-long
        data = self._store.data(fields, return_type)
        if return_type == "pandas":
            data = ArchiveDataFrame(data)
        return data

    def as_pandas(self, include_solutions=True, include_metadata=False):
        """DEPRECATED."""
        # pylint: disable = unused-argument
        raise RuntimeError(
            "as_pandas has been deprecated. Please use "
            "archive.data(..., return_type='pandas') instead, or consider "
            "retrieving individual fields, e.g., "
            "objective = archive.data('objective')")

    def cqd_score(self,
                  iterations,
                  target_points,
                  penalties,
                  obj_min,
                  obj_max,
                  dist_max=None,
                  dist_ord=None):
        """Computes the CQD score of the archive.

        The Continuous Quality Diversity (CQD) score was introduced in
        `Kent 2022 <https://dl.acm.org/doi/10.1145/3520304.3534018>`_.

        .. note:: This method by default assumes that the archive has an
            ``upper_bounds`` and ``lower_bounds`` property which delineate the
            bounds of the measure space, as is the case in
            :class:`~ribs.archives.GridArchive`,
            :class:`~ribs.archives.CVTArchive`, and
            :class:`~ribs.archives.SlidingBoundariesArchive`.  If this is not
            the case, ``dist_max`` must be passed in, and ``target_points`` must
            be an array of custom points.

        Args:
            iterations (int): Number of times to compute the CQD score. We
                return the mean CQD score across these iterations.
            target_points (int or array-like): Number of target points to
                generate, or an (iterations, n, measure_dim) array which
                lists n target points to list on each iteration. When an int is
                passed, the points are sampled uniformly within the bounds of
                the measure space.
            penalties (int or array-like): Number of penalty values over which
                to compute the score (the values are distributed evenly over the
                range [0,1]). Alternatively, this may be a 1D array which
                explicitly lists the penalty values. Known as :math:`\\theta` in
                Kent 2022.
            obj_min (float): Minimum objective value, used when normalizing the
                objectives.
            obj_max (float): Maximum objective value, used when normalizing the
                objectives.
            dist_max (float): Maximum distance between points in measure space.
                Defaults to the distance between the extremes of the measure
                space bounds (the type of distance is computed with the order
                specified by ``dist_ord``). Known as :math:`\\delta_{max}` in
                Kent 2022.
            dist_ord: Order of the norm to use for calculating measure space
                distance; this is passed to :func:`numpy.linalg.norm` as the
                ``ord`` argument. See :func:`numpy.linalg.norm` for possible
                values. The default is to use Euclidean distance (L2 norm).
        Returns:
            The mean CQD score obtained with ``iterations`` rounds of
            calculations.
        Raises:
            RuntimeError: The archive does not have the bounds properties
                mentioned above, and dist_max is not specified or the target
                points are not provided.
            ValueError: target_points or penalties is an array with the wrong
                shape.
        """
        if (not (hasattr(self, "upper_bounds") and
                 hasattr(self, "lower_bounds")) and
            (dist_max is None or np.isscalar(target_points))):
            raise RuntimeError(
                "When the archive does not have lower_bounds and "
                "upper_bounds properties, dist_max must be specified, "
                "and target_points must be an array")

        if np.isscalar(target_points):
            # pylint: disable = no-member
            target_points = self._rng.uniform(
                low=self.lower_bounds,
                high=self.upper_bounds,
                size=(iterations, target_points, self.measure_dim),
            )
        else:
            # Copy since we return this.
            target_points = np.copy(target_points)
            if (target_points.ndim != 3 or
                    target_points.shape[0] != iterations or
                    target_points.shape[2] != self.measure_dim):
                raise ValueError(
                    "Expected target_points to be a 3D array with "
                    f"shape ({iterations}, n, {self.measure_dim}) "
                    "(i.e. shape (iterations, n, measure_dim)) but it had "
                    f"shape {target_points.shape}")

        if dist_max is None:
            # pylint: disable = no-member
            dist_max = np.linalg.norm(self.upper_bounds - self.lower_bounds,
                                      ord=dist_ord)

        if np.isscalar(penalties):
            penalties = np.linspace(0, 1, penalties)
        else:
            penalties = np.copy(penalties)  # Copy since we return this.
            check_is_1d(penalties, "penalties")

        objective_batch = self._store.data("objective")
        measures_batch = self._store.data("measures")

        norm_objectives = objective_batch / (obj_max - obj_min)

        scores = np.zeros(iterations)

        for itr in range(iterations):
            # Distance calculation -- start by taking the difference between
            # each measure i and all the target points.
            distances = measures_batch[:, None] - target_points[itr]

            # (len(archive), n_target_points) array of distances.
            distances = np.linalg.norm(distances, ord=dist_ord, axis=2)

            norm_distances = distances / dist_max

            for penalty in penalties:
                # Known as omega in Kent 2022 -- a (len(archive),
                # n_target_points) array.
                values = norm_objectives[:, None] - penalty * norm_distances

                # (n_target_points,) array.
                max_values_per_target = np.max(values, axis=0)

                scores[itr] += np.sum(max_values_per_target)

        return CQDScoreResult(
            iterations=iterations,
            mean=np.mean(scores),
            scores=scores,
            target_points=target_points,
            penalties=penalties,
            obj_min=obj_min,
            obj_max=obj_max,
            dist_max=dist_max,
            dist_ord=dist_ord,
        )
