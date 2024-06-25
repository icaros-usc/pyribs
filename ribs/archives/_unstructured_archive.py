"""Contains the UnstructuredArchive class."""
from functools import cached_property
from typing import Any, Dict, Union

import numpy as np
from scipy.spatial import cKDTree

from ribs._utils import (check_batch_shape, check_finite, check_shape,
                         np_scalar, validate_batch, validate_single)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._transforms import (batch_entries_with_threshold,
                                       compute_best_index,
                                       compute_objective_sum,
                                       single_entry_with_threshold)


class UnstructuredArchive(ArchiveBase):
    """An archive that adds new solutions based on their novelty.

    This archive originates in Novelty Search and is described in `Lehman 2011
    <http://eplex.cs.ucf.edu/papers/lehman_ecj11.pdf>`_. Solutions are added to
    the archive if their `novelty` exceeds a certain threshold. `Novelty`
    :math:`\\rho` is defined as the average (Euclidean) distance in measure
    space to the :math:`k`-nearest neighbors of the solution in the archive:

    .. math::

        \\rho(x) = \\frac{1}{k}\\sum_{i=1}^{k}\\text{dist}(x, \\mu_i)

    Where :math:`x` is the measure value of some solution, and
    :math:`\\mu_{1..k}` are the measure values of the :math:`k`-nearest
    neighbors in measure space.

    Args:
        solution_dim (int): Dimension of the solution space.
        measure_dim (int): The dimension of the measure space.
        k_neighbors (int): The maximum number of nearest neighbors to use for
            computing novelty (`maximum` here is indicated for cases when there
            are fewer than ``k_neighbors`` solutions in the archive).
        novelty_threshold (float): The level of novelty required to add a
            solution to the archive.
        compare_to_batch (bool): When solutions are inserted into the archive,
            the default behavior is to compute their novelty with respect to the
            solutions currently in the archive. If this argument is True, the
            novelty will be computed with respect to both the archive and the
            incoming solutions in the batch. In other words, solutions will need
            to be novel with respect to both the archive and the incoming
            solutions.
        initial_capacity (int): Since this archive is unstructured, it does not
            have a fixed size, and it will grow as solutions are added. In the
            implementation, we store solutions in fixed-size arrays, and every
            time the capacity of these arrays is reached, we double their sizes
            (similar to the vector in C++). This parameter determines the
            initial capacity of the archive's arrays. It may be useful when it
            is known in advance how large the archive will grow.
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
        ckdtree_kwargs (dict): When computing nearest neighbors, we construct a
            :class:`~scipy.spatial.cKDTree`. This parameter will pass additional
            kwargs when constructing the tree. By default, we do not pass in any
            kwargs.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 measure_dim,
                 k_neighbors,
                 novelty_threshold,
                 compare_to_batch=False,
                 initial_capacity=128,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64,
                 extra_fields=None,
                 ckdtree_kwargs=None):

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=initial_capacity,
            measure_dim=measure_dim,
            # learning_rate and threhsold_min take on default values since we do
            # not use CMA-MAE threshold in this archive.
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        self._k_neighbors = int(k_neighbors)
        self._novelty_threshold = np_scalar(novelty_threshold,
                                            dtype=self.dtypes["measures"])
        self._compare_to_batch = bool(compare_to_batch)
        self._ckdtree_kwargs = ({} if ckdtree_kwargs is None else
                                ckdtree_kwargs.copy())

        # k-D tree with current measures in the archive. Updated on add().
        self._cur_kd_tree = cKDTree(self._store.data("measures"),
                                    **self._ckdtree_kwargs)

    @property
    def k_neighbors(self):
        """int: The number of nearest neighbors to use for determining
        novelty."""
        return self._k_neighbors

    @property
    def novelty_threshold(self):
        """dtypes["measures"]: The degree of novelty required add a solution to
        the archive."""
        return self._novelty_threshold

    @property
    def compare_to_batch(self):
        """bool: Whether novelty is computed with respect to other solutions in
        the batch when adding solutions."""
        return self._compare_to_batch

    @property
    def cells(self):
        """int: Total number of cells in the archive. Since this archive is
        unstructured and grows over time, the number of cells is equal to the
        number of solutions currently in the archive."""
        return len(self)

    @property
    def capacity(self):
        """int: The number of solutions that can currently be stored in this
        archive. The capacity doubles every time the archive fills up, similar
        to a C++ vector."""
        return self._store.capacity

    def index_of(self, measures) -> np.ndarray:
        """Returns the index of the closest solution to the given measures.

        Unlike the structured archives like :class:`~ribs.archives.GridArchive`,
        this archive does not have indexed cells where each measure "belongs."
        Thus, this method instead returns the index of the closest measure to
        each solution passed in.

        This means that :meth:`retrieve` will return the solution with the
        closest measure to each measure passed into that method.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing
              the location of the solution in the archive.
        Raises:
            RuntimeError: There were no entries in the archive.
            ValueError: ``measures`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self.empty:
            raise RuntimeError(
                "There were no solutions in the archive. "
                "`UnstructuredArchive.index_of` computes the nearest neighbor "
                "to the input measures, so there must be at least one solution "
                "present in the archive.")

        _, indices = self._cur_kd_tree.query(measures)
        return indices.astype(np.int32)

    # TODO: API for diversity optimization?
    # TODO: Update docstring, comment on how objectives are not considered.
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

        reference_measures = self.data("measures")
        if self._compare_to_batch:
            reference_measures = np.concatenate((reference_measures, measures),
                                                axis=0)

        if self.compare_to_batch:
            # We need 1 more neighbor since the first neighbor will be the
            # solution itself.
            k_neighbors = min(len(reference_measures), self.k_neighbors + 1)
            first_neighbor_idx = 1
        else:
            k_neighbors = min(len(reference_measures), self.k_neighbors)
            first_neighbor_idx = 0

        if self.compare_to_batch:
            kdt = cKDTree(reference_measures, **self._ckdtree_kwargs)
        else:
            kdt = self._cur_kd_tree

        if len(reference_measures) == 0:
            novelty = np.full(
                len(measures),
                np.inf,
                dtype=self.dtypes["measures"],
            )
            eligible = np.ones(len(measures), dtype=bool)
        else:
            dists, _ = kdt.query(measures, k=k_neighbors)

            # Since query() automatically squeezes the last dim.
            if k_neighbors == 1:
                dists = dists[:, None]

            # If compare_to_batch, then the first nearest neighbor will be the
            # solution itself.
            first_neighbor_idx = 1 if self._compare_to_batch else 0

            novelty = np.mean(dists[:, first_neighbor_idx:], axis=1)
            eligible = novelty >= self.novelty_threshold

        n_eligible = np.sum(eligible)
        new_size = len(self) + n_eligible

        if new_size > self.capacity:
            # Resize the store by doubling its capacity. We may need to double
            # the capacity multiple times. The log2 below indicates how many
            # times we would need to double the capacity. We obtain the final
            # capacity by raising to a power of 2.
            multiplier = 2**int(np.ceil(np.log2(new_size / self.capacity)))
            self._store.resize(multiplier * self.capacity)

        # todo: non-eligible solutions?

        add_info = self._store.add(
            np.arange(len(self), new_size),
            {
                key: val[eligible] for key, val in data.items()
            },
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
            # TODO: Remove stats that don't exist for this archive.
            self._stats_update(objective_sum, best_index)

            self._cur_kd_tree = cKDTree(self._store.data("measures"),
                                        **self._ckdtree_kwargs)

            # Clear the cached properties since they have now changed.
            if "upper_bounds" in self.__dict__:
                del self.__dict__["upper_bounds"]
            if "lower_bounds" in self.__dict__:
                del self.__dict__["lower_bounds"]

        # TODO: Clean up.
        status = np.zeros(len(measures), dtype=np.int32)
        status[eligible] = add_info["status"]
        add_info["status"] = status

        add_info["value"] = novelty

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

        return self.add(**{key: [val] for key, val in data.items()})

        #  for name, arr in data.items():
        #      data[name] = np.expand_dims(arr, axis=0)

        #  add_info = self._store.add(
        #      np.expand_dims(self.index_of_single(measures, resize=True), axis=0),
        #      data,
        #      {
        #          "dtype": self.dtypes["threshold"],
        #          "learning_rate": self._learning_rate,
        #          "threshold_min": self._threshold_min,
        #          "objective_sum": self._objective_sum,
        #      },
        #      [
        #          single_entry_with_threshold,
        #          compute_objective_sum,
        #          compute_best_index,
        #      ],
        #  )

        #  objective_sum = add_info.pop("objective_sum")
        #  best_index = add_info.pop("best_index")

        #  for name, arr in add_info.items():
        #      add_info[name] = arr[0]

        #  if add_info["status"]:
        #      self._stats_update(objective_sum, best_index)

        #  return add_info

    @cached_property
    def upper_bounds(self) -> np.ndarray:
        """The upper bounds of the measures in the archive.

        Since the archive can grow arbitrarily, this is calculated based on the
        maximum measure values of the solutions in the archive.

        Raises:
            RuntimeError: There are no solutions in the archive, so the
            upper bounds do not exist.
        """
        if self.empty:
            raise RuntimeError("There are no solutions in the archive, so the "
                               "upper bounds do not exist.")

        return np.max(self._store.data("measures"), axis=0)

    @cached_property
    def lower_bounds(self) -> np.ndarray:
        """The lower bounds of the measures in the archive.

        Since the archive can grow arbitrarily, this is calculated based on the
        minimum measure values of the solutions in the archive.

        Raises:
            RuntimeError: There are no solutions in the archive, so the
            lower bounds do not exist.
        """
        if self.empty:
            raise RuntimeError("There are no solutions in the archive, so the "
                               "lower bounds do not exist.")

        return np.min(self._store.data("measures"), axis=0)

    def cqd_score(self,
                  iterations,
                  target_points,
                  penalties,
                  obj_min,
                  obj_max,
                  dist_max=None,
                  dist_ord=None):
        """Computes the CQD score of the archive.

        Refer to the documentation in :meth:`ArchiveBase.cqd_score` for more
        info. The key difference from the base implementation is that the
        implementation in ArchiveBase assumes the archive has a pre-defined
        measure space with lower and upper bounds. However, by nature of being
        unstructured, this archive has lower and upper bounds that change over
        time. Thus, it is required to directly pass in ``target_points`` and
        ``dist_max``.

        Raises:
            ValueError: dist_max and target_points were not passed in.
        """

        if dist_max is None or np.isscalar(target_points):
            raise ValueError(
                "In UnstructuredArchive, dist_max must be passed "
                "in, and target_points must be passed in as a custom "
                "array of points.")

        return super().cqd_score(
            iterations=iterations,
            target_points=target_points,
            penalties=penalties,
            obj_min=obj_min,
            obj_max=obj_max,
            dist_max=dist_max,
            dist_ord=dist_ord,
        )
