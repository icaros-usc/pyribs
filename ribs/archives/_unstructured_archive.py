"""Contains the UnstructuredArchive class."""
from functools import cached_property
from typing import Any, Dict, List, Union

import numpy as np

from ribs._utils import (check_batch_shape, check_finite, check_shape,
                         np_scalar, validate_batch, validate_single)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._transforms import (batch_entries_with_threshold,
                                       compute_best_index,
                                       compute_objective_sum,
                                       single_entry_with_threshold)


class UnstructuredArchive(ArchiveBase):
    """An archive that adds new solutions based on their novelty.

    This archive is described in `Lehman 2011
    <https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_gecco11.pdf>`_.  # pylint: disable=C0301
    If a solution is in a sparse area of metric space it is added
    unconditionally. When a solution is in an overdense region it is added to
    the archive only if its objective improves upon the nearest existing
    solution. The archive uses the mean distance of the k-nearest neighbors to
    determine the sparsity of metric space

    .. note:: The archive is initialized as empty

    Args:
        solution_dim (int): Dimension of the solution space.
        measure_dim (int): The dimension of the measure space.
        k_neighbors (int): The number of nearest neighbors to use for
            determining sparseness.
        sparsity_threshold (float): The level of sparsity required to add a
            solution to the archive unconditionally
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
        dtype (str or data-type or dict): Data type of the solutions,
            objectives, and measures. We only support ``"f"`` / ``np.float32``
            and ``"d"`` / ``np.float64``. Alternatively, this can be a dict
            specifying separate dtypes, of the form ``{"solution": <dtype>,
            "objective": <dtype>, "measures": <dtype>}``.
        extra_fields (dict): Description of extra fields of data that is stored
            next to elite data like solutions and objectives. The description is
            a dict mapping from a field name (str) to a tuple of ``(shape,
            dtype)``. For instance, ``{"foo": ((), np.float32), "bar": ((10,),
            np.float32)}`` will create a "foo" field that contains scalar values
            and a "bar" field that contains 10D values. Note that field names
            must be valid Python identifiers, and names already used in the
            archive are not allowed.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 measure_dim,
                 k_neighbors,
                 sparsity_threshold,
                 learning_rate=None,
                 threshold_min=-np.inf,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64,
                 extra_fields=None):

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=0,
            measure_dim=measure_dim,
            learning_rate=learning_rate,
            threshold_min=threshold_min,
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        self._k_neighbors = int(k_neighbors)
        self._sparsity_threshold = np_scalar(sparsity_threshold,
                                             dtype=self.dtypes["measures"])

    @property
    def k_neighbors(self) -> int:
        """int: The number of nearest neighbors to use for determining
        sparsity."""
        return self._k_neighbors

    @property
    def sparsity_threshold(self) -> float:
        """:attr:`dtype` : The degree of sparseness in metric space required for
        a solution to be added unconditionally."""
        return self._sparsity_threshold

    def index_of(self, measures, resize: bool = False) -> np.ndarray:
        """Returns archive indices for the given batch of measures.

        First, if the archive is empty it is resized to fit the new data.

        Then the distance between all solutions in the archive and the incoming
        `measures` is calculated.

        Next the distances between all the incoming measures and themselves is
        also calculated and concatenated with the distances from before.

        Next, we compute the mean distance of the k-nearest neighbors for each
        incoming measure. If any of the mean distances are *above* the
        `sparsity_threshold` then the archive is resized to accomodate them.
        They are each assigned new indices in the archive.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing
            the flattened grid coordinates.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        batch_size = measures.shape[0]

        if not batch_size:
            return np.array([], dtype=np.int32)

        if self.empty:
            # empty archive: the indices are all new so we can just
            # resize the store
            if batch_size > self._store.capacity:
                self._store.resize(batch_size)
            self._cells = self._store.capacity

            if batch_size == 1:
                # no need to bother with expensive operations
                return np.array([0], dtype=np.int32)

            distances = np.square(measures[:, None, ...] -
                                  measures[None]).sum(axis=2)

            # find the entries which are the same point in measure space
            # at this point we don't care about the k-nearest neighbors, just
            # whether two points are essentially the same
            index_array = np.isclose(distances, 0)
            indices = index_array.argmax(axis=1)
            # remove gaps in indices
            cell_map = {unq: idx for idx, unq in enumerate(np.unique(indices))}
            return np.array([cell_map[s] for s in indices])

        store_measures = self._store.data("measures")
        distances = np.square(measures[:, None] - store_measures).sum(axis=2)

        # return early so that we don't go through the expensive function calls
        # unless necessary
        if not resize:
            return self._store.occupied_list[np.argmin(distances,
                                                       axis=1).astype(np.int32)]
        curr_occ = self._store._props["n_occupied"]  # pylint: disable=W0212
        # calculate the distances from the measures to themselves and
        # concatenate this with the existing distances, while ignoring
        # self-comparisons.
        meas_distances = np.square(measures[:, None] - measures).sum(axis=2)
        distances = np.concatenate([distances, meas_distances], axis=1)

        # find the nearest [k+1]-neighbors to determine the sparsity
        # use k+1 since self is included in the first k-neighbors
        top_k = np.argsort(distances, axis=1)[:, :min(self._k_neighbors +
                                                      1, distances.shape[1])]

        # determine which entries can be newly inserted and resize
        # the store to accommodate them
        where_mask = np.zeros(distances.shape, dtype=np.bool_)
        where_mask[np.repeat(np.arange(batch_size, dtype=np.int32)[:, None],
                             top_k.shape[1],
                             axis=1), top_k] = True
        new_entries = np.sqrt(distances[where_mask]).reshape(
            batch_size,
            -1).sum(axis=1) / self._k_neighbors > self._sparsity_threshold
        new_entry_indicies = np.argwhere(new_entries) + curr_occ

        # use the nearest index already in the archive or the closest one which
        # will be added
        indices = np.array([
            tk[np.logical_or(
                tk < curr_occ,
                np.array([t in new_entry_indicies for t in tk], dtype=np.bool_),
            )][0] for tk in top_k
        ],
                           dtype=np.int32)

        # a mapping from top-k indices to their location in the archive
        cell_map = {
            unq: unq
            for unq in self._store.occupied_list[indices[indices < curr_occ]]
        }
        if np.any(new_entries):
            unique_indices = np.unique(indices)
            unique_indices = unique_indices[unique_indices >= curr_occ]

            additions = new_entries.sum()
            if curr_occ + additions > self._store.capacity:
                # we have to add more new entries than our archive can hold
                self._store.resize(curr_occ + additions)
                self._cells = self._store.capacity

            # assign new entries to unoccupied indices
            free_indices = list(np.argwhere(~self._store.occupied))
            for unq in unique_indices:
                cell_map[unq] = free_indices.pop(0).item()
        return np.array([cell_map.get(s, s) for s in indices])

    def index_of_single(self, measures, resize: bool = False) -> np.ndarray:
        """Returns the index of the measures for one solution.

        While :meth:`index_of` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution. If :meth:`index_of` is
        implemented correctly, this method should work immediately (i.e. `"out
        of the box" <https://idioms.thefreedictionary.com/Out-of-the-Box>`_).

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures for
                a single solution.
            resize (bool) : whether or not to allow resizing of the archive to
                fit measures which are not "close" to existing solutions
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
        return self.index_of(measures[None], resize=resize)[0]

    def add(self, solution, objective, measures, **fields) -> Union[Any, Dict]:
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
            self.index_of(data["measures"], resize=True),
            data,
            {
                "dtype": self.dtypes["threshold"],
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

        # clear the cached properties since they've now changed
        if "upper_bounds" in self.__dict__:
            del self.__dict__["upper_bounds"]
        if "lower_bounds" in self.__dict__:
            del self.__dict__["lower_bounds"]
        if "boundaries" in self.__dict__:
            del self.__dict__["boundaries"]

        return add_info

    def add_single(self, solution, objective, measures,
                   **fields) -> Union[Any, Dict]:
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
            np.expand_dims(self.index_of_single(measures, resize=True), axis=0),
            data,
            {
                "dtype": self.dtypes["threshold"],
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

    @cached_property
    def upper_bounds(self) -> np.ndarray:
        """
        The upper bounds of the measures in the archive.

        Since the archive can grow arbitrarily this is calculated based on the
        maximum measure values of the solutions in the archive.

        If the archive only has a single solution, then the upper bound would
        equal the lower bound which can cause problems. So we add 1 to the
        upper bound to make sure the boundaries form an interval rather than
        a point.
        """
        upper = np.max(self._store.data("measures"), axis=0)
        if self._store._props["n_occupied"] > 1:  # pylint: disable=W0212
            return upper
        return upper + 1

    @cached_property
    def lower_bounds(self) -> np.ndarray:
        """
        The lower bounds of the measures in the archive.

        Since the archive can grow arbitrarily this is calculated based on the
        minimum measure values of the solutions in the archive.

        If the archive only has a single solution, then the upper bound would
        equal the lower bound which can cause problems. So we subtract 1 from
        the lower bound to make sure the boundaries form an interval rather
        than a point.
        """
        lower = np.min(self._store.data("measures"), axis=0)
        if self._store._props["n_occupied"] > 1:  # pylint: disable=W0212
            return lower
        return lower - 1

    @cached_property
    def boundaries(self) -> List[np.ndarray]:
        """
        The boundaries of the measures in the archive.

        Since the archive can grow arbitrarily this is calculated based on the
        maximum and minimum measure values of the solutions in the archive.

        The intervals are subdivided into an evenly-spaced grid.
        """
        measures = self._store.data("measures")
        bounds = []
        for dim in range(self.measure_dim):
            dim_measures = measures[:, dim]
            # use unique since it sorts for us and in the case where
            # some measures coincide we don't have duplicate bounds
            _, indices = np.unique(dim_measures, return_index=True)

            # upper and lower bounds are min/max endpoints in the measures
            # so we can just use them as-is. Otherwise we take the midpoint
            # between two measures
            # in the case of a single solution the `for` loop iterates over
            # an empty list so we are left with just the upper and lower bounds
            # with the solution in the middle
            curr_bounds = [self.lower_bounds[dim]]
            for idx, ind in enumerate(indices[1:], start=1):
                curr_bounds.append(
                    (dim_measures[ind] + dim_measures[indices[idx - 1]]) / 2)
            curr_bounds.append(self.upper_bounds[dim])
            bounds.append(np.array(curr_bounds, dtype=self.dtypes["measures"]))

        return bounds
