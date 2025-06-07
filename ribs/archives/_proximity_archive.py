"""Contains the ProximityArchive."""
import numpy as np
from scipy.spatial import cKDTree

from ribs._utils import (check_batch_shape, check_finite, check_shape,
                         np_scalar, validate_batch, validate_single)
from ribs.archives._archive_base_2 import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._transforms import (batch_entries_with_threshold,
                                       compute_best_index,
                                       compute_objective_sum)
from ribs.archives._utils import parse_dtype


class ProximityArchive(ArchiveBase):
    # pylint: disable = too-many-public-methods
    """An archive that adds new solutions based on novelty, where novelty is
    defined via proximity to other solutions in measure space.

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

    This archive also supports the local competition computation from Novelty
    Search with Local Competition, described in `Lehman 2011b
    <http://eplex.cs.ucf.edu/papers/lehman_gecco11.pdf>`_.

    .. note:: When used for diversity optimization, this archive does not
        require any objectives, and ``objective=None`` can be passed into
        :meth:`add`. For consistency with the rest of pyribs, ``objective=None``
        will result in a default objective value of 0, which will also cause
        stats like QD score and best objective to be 0. Alternatively, it is
        possible to associate objectives with the solutions by passing
        ``objective`` to :meth:`add` just like in other archives.

    .. note:: Some statistics will behave differently than in other archives:

        - If this archive has any solutions in it, the coverage
          (``archive.stats.coverage``) will always be reported as 1. This is
          because the archive is unbounded, so there is no predefined number of
          cells to fill. As such, ``archive.stats.num_elites`` may provide a
          more meaningful coverage metric. It is also common to create a
          :class:`~ribs.archives.GridArchive` or
          :class:`~ribs.archives.CVTArchive` as a result archive, from which a
          meaningful coverage can be computed.
        - Since the number of archive cells equals the number of elites in the
          archive, the normalized QD score (``archive.stats.norm_qd_score``)
          will always equal the mean objective (``archive.stats.obj_mean``).

    Args:
        solution_dim (int): Dimensionality of the solution space.
        measure_dim (int): Dimensionality of the measure space.
        k_neighbors (int): The maximum number of nearest neighbors for computing
            novelty (`maximum` here is indicated since there may be fewer than
            ``k_neighbors`` solutions in the archive).
        novelty_threshold (float): The level of novelty required to add a
            solution to the archive.
        local_competition (bool): Whether to turn on local competition behavior.
            If turned on, the archive will require objectives to be passed in
            during :meth:`add`. Furthermore, the ``add_info`` returned by
            :meth:`add` will include local competition information. Finally,
            solutions can be replaced in the archive. Specifically, if a
            candidate solution's novelty is below the novelty threshold, its
            objective will be compared to that of its nearest neighbor. If the
            candidate's objective is higher, it will replace the nearest
            neighbor.
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
            objectives, and measures. This can be ``"f"`` / ``np.float32``,
            ``"d"`` / ``np.float64``, or a dict specifying separate dtypes, of
            the form ``{"solution": <dtype>, "objective": <dtype>, "measures":
            <dtype>}``.
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
    Raises:
        ValueError: ``initial_capacity`` must be at least 1.
    """

    def __init__(
        self,
        *,
        solution_dim,
        measure_dim,
        k_neighbors,
        novelty_threshold,
        local_competition=False,
        initial_capacity=128,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
        ckdtree_kwargs=None,
    ):
        self._rng = np.random.default_rng(seed)

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        # Set up the ArrayStore, which is a data structure that stores all the
        # elites' data in arrays sharing a common index.
        extra_fields = extra_fields or {}
        reserved_fields = {
            "solution", "objective", "measures", "threshold", "index"
        }
        if reserved_fields & extra_fields.keys():
            raise ValueError("The following names are not allowed in "
                             f"extra_fields: {reserved_fields}")
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1.")
        dtype = parse_dtype(dtype)
        self._store = ArrayStore(
            field_desc={
                "solution": ((self.solution_dim,), dtype["solution"]),
                "objective": ((), dtype["objective"]),
                "measures": ((self.measure_dim,), dtype["measures"]),
                # Must be same dtype as the objective since they share
                # calculations.
                "threshold": ((), dtype["objective"]),
                **extra_fields,
            },
            capacity=initial_capacity,
        )

        # Set up constant properties.
        self._k_neighbors = int(k_neighbors)
        self._novelty_threshold = np_scalar(novelty_threshold,
                                            dtype=self.dtypes["measures"])
        self._local_competition = local_competition
        self._ckdtree_kwargs = ({} if ckdtree_kwargs is None else
                                ckdtree_kwargs.copy())
        self._qd_score_offset = np_scalar(qd_score_offset,
                                          self.dtypes["objective"])

        # Set up k-D tree with current measures in the archive. Updated on
        # add().
        self._cur_kd_tree = cKDTree(self._store.data("measures"),
                                    **self._ckdtree_kwargs)

        # TODO: Refactor stats?

        # Set up statistics.
        self._stats = None
        self._best_elite = None
        # Sum of all objective values in the archive; useful for computing
        # qd_score and obj_mean.
        self._objective_sum = None
        self._stats_reset()

    ## Properties inherited from ArchiveBase ##

    @property
    def field_list(self):
        return self._store.field_list

    @property
    def stats(self):
        return self._stats

    @property
    def empty(self):
        return len(self._store) == 0

    @property
    def dtypes(self):
        return self._store.dtypes

    ## Properties that are not in ArchiveBase ##
    ## Roughly ordered by the parameter list in the constructor. ##

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
    def k_neighbors(self):
        """int: The number of nearest neighbors for computing novelty."""
        return self._k_neighbors

    @property
    def novelty_threshold(self):
        """dtypes["measures"]: The degree of novelty required add a solution to
        the archive."""
        return self._novelty_threshold

    @property
    def local_competition(self):
        """bool: Whether local competition behavior is turned on."""
        return self._local_competition

    @property
    def capacity(self):
        """int: The number of solutions that can currently be stored in this
        archive. The capacity doubles every time the archive fills up."""
        return self._store.capacity

    @property
    def cells(self):
        """int: Strictly speaking, this archive does not have "cells" since it
        does not have a tessellation like other archives. However, for API
        compatibility, we set the number of cells as equal to the number of
        solutions currently in the archive."""
        return len(self)

    @property
    def qd_score_offset(self):
        """float: The offset which is subtracted from objective values when
        computing the QD score."""
        return self._qd_score_offset

    ## dunder methods ##

    def __len__(self):
        return len(self._store)

    def __iter__(self):
        return iter(self._store)

    ## Utilities ##

    def _stats_reset(self):
        """Resets the archive stats."""
        zero = np_scalar(0.0, dtype=self.dtypes["objective"])

        self._stats = ArchiveStats(
            num_elites=0,
            coverage=zero,
            qd_score=zero,
            norm_qd_score=zero,
            obj_max=None,
            obj_mean=None,
        )
        self._best_elite = None
        self._objective_sum = zero

    def _stats_update(self, new_objective_sum, new_best_index):
        """Updates statistics based on a new sum of objective values
        (new_objective_sum) and the index of a potential new best elite
        (new_best_index)."""
        self._objective_sum = new_objective_sum
        new_qd_score = (self._objective_sum -
                        np_scalar(len(self), dtype=self.dtypes["objective"]) *
                        self._qd_score_offset)

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
            coverage=np_scalar(len(self) / self.cells,
                               dtype=self.dtypes["objective"]),
            qd_score=new_qd_score,
            norm_qd_score=np_scalar(new_qd_score / self.cells,
                                    dtype=self.dtypes["objective"]),
            obj_max=new_obj_max,
            obj_mean=np_scalar(self._objective_sum / len(self),
                               dtype=self.dtypes["objective"]),
        )

    def index_of(self, measures) -> np.ndarray:
        """Returns the index of the closest solution to the given measures.

        Unlike the structured archives like :class:`~ribs.archives.GridArchive`,
        this archive does not have indexed cells where each measure "belongs."
        Thus, this method instead returns the index of the solution with the
        closest measure to each solution passed in.

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
                "`ProximityArchive.index_of` computes the nearest "
                "neighbor to the input measures, so there must be at least one "
                "solution present in the archive.")

        _, indices = self._cur_kd_tree.query(measures)
        return indices.astype(np.int32)

    def index_of_single(self, measures):
        """Returns the index of the measures for one solution.

        See :meth:`index_of`.

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

    def compute_novelty(self, measures, local_competition=None):
        """Computes the novelty and local competition of the given measures.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
            local_competition (None or array-like): This can be None to
                indicate not to compute local competition. Otherwise, it can be
                a (batch_size,) array of objective values to use as references
                for computing objective values.
        Returns:
            numpy.ndarray or tuple: Either one value or a tuple of two values:

            - numpy.ndarray: (batch_size,) array holding the novelty score of
              each measure. If the archive is empty, the novelty is set to the
              :attr:`novelty_threshold`.
            - numpy.ndarray: If ``local_competition`` is passed in, a
              (batch_size,) array holding the local competition of each solution
              will also be returned. If the archive is empty, the local
              competition will be set to 0.
        """
        measures = np.asarray(measures)
        batch_size = len(measures)

        use_local_competition = local_competition is not None
        if use_local_competition:
            objectives = np.asarray(local_competition)

        if self.empty:
            # Set default values for novelty and local competition when archive
            # is empty.
            novelty = np.full(batch_size,
                              self.novelty_threshold,
                              dtype=self.dtypes["measures"])

            if use_local_competition:
                local_competition_scores = np.zeros(len(novelty),
                                                    dtype=np.int32)
        else:
            # Compute nearest neighbors.
            k_neighbors = min(len(self), self.k_neighbors)
            dists, indices = self._cur_kd_tree.query(measures, k=k_neighbors)

            # Expand since query() automatically squeezes the last dim when k=1.
            dists = dists[:, None] if k_neighbors == 1 else dists

            novelty = np.mean(dists, axis=1)

            if use_local_competition:
                indices = indices[:, None] if k_neighbors == 1 else indices

                # The first item returned by `retrieve` is `occupied` -- all
                # these indices are occupied since they are indices of solutions
                # in the archive.
                neighbor_objectives = self._store.retrieve(
                    indices.ravel(), "objective")[1]
                neighbor_objectives = neighbor_objectives.reshape(indices.shape)

                # Local competition is the number of neighbors who have a lower
                # objective.
                local_competition_scores = np.sum(
                    neighbor_objectives < objectives[:, None],
                    axis=1,
                    dtype=np.int32,
                )

        if use_local_competition:
            # pylint: disable-next = used-before-assignment
            return novelty, local_competition_scores
        else:
            return novelty

    ## Methods for writing to the archive ##

    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        Solutions are inserted if they have a high enough novelty score as
        discussed in the documentation for this class. The novelty is determined
        by comparing to solutions currently in the archive.

        If :attr:`local_competition` is turned on, solutions can also replace
        existing solutions in the archive. Namely, if the solution was not novel
        enough to be added, it will be compared to its nearest neighbor, and if
        it exceeds the objective value of its nearest neighbor, it will replace
        the nearest neighbor. If there are conflicts where multiple solutions
        may replace a single solution, the highest-performing is chosen.

        .. note:: The indices of all arguments should "correspond" to each
            other, i.e. ``solution[i]``, ``objective[i]``,
            ``measures[i]``, and should be the solution parameters,
            objective, and measures for solution ``i``.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solution parameters.
            objective (None or array-like): A value of None will cause the
                objective values to default to 0. However, if the user wishes to
                associate an objective with each solution, this can be a
                (batch_size,) array with objective function evaluations of the
                solutions. If :attr:`local_competition` is turned on, this
                argument must be provided.
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
              - ``1``: The solution replaced an existing solution in the
                archive due to having a higher objective (only applies if
                :attr:`local_competition` is turned on).
              - ``2``: The solution was added to the archive due to being
                sufficiently novel.

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus` e.g. with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"novelty"`` (:class:`numpy.ndarray` of :attr:`dtypes`
              ["measures"]): The computed novelty of the solutions passed in. If
              there were no solutions to compute novelty with respect to (i.e.,
              the archive was empty), the novelty is set to the
              :attr:`novelty_threshold`.

            - ``"local_competition"`` (:class:`numpy.ndarray` of :class:`int`):
              Only available if :attr:`local_competition` is turned on.
              Indicates, for each solution, how many of the nearest neighbors
              had lower objective values. Maximum value is :attr:`k_neighbors`.
              If there were no solutions to compute novelty with respect to,
              (i.e., the archive was empty), the local competition is set to 0.

            - ``"value"`` (:class:`numpy.ndarray` of
              :attr:`dtypes` ["objective"]): Only available if
              :attr:`local_competition` is turned on. The meaning of each value
              depends on the corresponding ``status`` and is inspired by the
              values in CMA-ME (`Fontaine 2020
              <https://arxiv.org/abs/1912.02400>`_):

              - ``0`` (not added): The value is the "negative improvement," i.e.
                the objective of the solution passed in minus the objective of
                the nearest neighbor (this value is negative because the
                solution did not have a high enough objective to be added to the
                archive).
              - ``1`` (replace/improve existing solution): The value is the
                "improvement," i.e. the objective of the solution passed in
                minus the objective of the elite that was replaced.
              - ``2`` (new solution): The value is just the objective of the
                solution.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf
                or NaN).
            ValueError: ``local_competition` is turned on but objective was not
                passed in.
        """
        if objective is None:
            if self.local_competition:
                raise ValueError("If local competition is turned on, objective "
                                 "must be passed in to add().")
            else:
                objective = np.zeros(len(solution),
                                     dtype=self.dtypes["objective"])

        data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        if self.local_competition:
            novelty, local_competition = self.compute_novelty(
                measures=data["measures"],
                local_competition=data["objective"],
            )
        else:
            novelty = self.compute_novelty(measures=data["measures"])

        novel_enough = novelty >= self.novelty_threshold
        n_novel_enough = np.sum(novel_enough)
        new_size = len(self) + n_novel_enough

        if self.local_competition:
            # In the case of local competition, we consider all solutions for
            # addition.
            add_indices = np.empty(len(novelty), dtype=np.int32)

            # New solutions are assigned the new indices.
            add_indices[novel_enough] = np.arange(len(self), new_size)

            # Solutions that were not novel enough have the potential to replace
            # their nearest neighbors in the archive.
            not_novel_enough = ~novel_enough
            n_not_novel_enough = len(novelty) - n_novel_enough
            if n_not_novel_enough > 0:
                add_indices[not_novel_enough] = \
                    self.index_of(data["measures"][not_novel_enough])

            add_data = data
        else:
            # Without local competition, the only solutions that can be added
            # are the ones that were novel enough.
            add_indices = np.arange(len(self), new_size)
            add_data = {key: val[novel_enough] for key, val in data.items()}

        if new_size > self.capacity:
            # Resize the store by doubling its capacity. We may need to double
            # the capacity multiple times. The log2 below indicates how many
            # times we would need to double the capacity. We obtain the final
            # multiplier by raising to a power of 2.
            multiplier = 2**int(np.ceil(np.log2(new_size / self.capacity)))
            self._store.resize(multiplier * self.capacity)

        add_info = self._store.add(
            add_indices,
            add_data,
            {
                "dtype": self.dtypes["objective"],
                "learning_rate": 1.0,
                # Note that when only novelty is considered, objectives default
                # to 0, so all solutions specified will be added because the
                # threshold_min is -np.inf.
                "threshold_min": -np.inf,
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

        # Add novelty to the data.
        add_info["novelty"] = novelty

        if self.local_competition:
            # add_info contains results for all solutions. We also want to
            # return local_competition info.
            add_info["local_competition"] = local_competition
        else:
            # add_info only contains results for the solutions that were novel
            # enough. Here we create an add_info that contains results for all
            # solutions.
            all_status = np.zeros(len(data["measures"]), dtype=np.int32)
            all_status[novel_enough] = add_info["status"]
            add_info["status"] = all_status

            # We ignore objective/threshold when only novelty is considered.
            del add_info["value"]

        if not np.all(add_info["status"] == 0):
            self._stats_update(objective_sum, best_index)

            # Make a new tree with the updated solutions.
            self._cur_kd_tree = cKDTree(self._store.data("measures"),
                                        **self._ckdtree_kwargs)

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        Args:
            solution (array-like): Parameters of the solution.
            objective (None or float): Set to None to get the default value of
                0; otherwise, a valid objective value is also acceptable.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains ``status`` and ``novelty`` keys; refer to :meth:`add`
            for the meaning of status and novelty.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures``
                has non-finite values.
            ValueError: ``local_competition` is turned on but objective was not
                passed in.
        """
        if objective is None:
            if self.local_competition:
                raise ValueError("If local competition is turned on, objective "
                                 "must be passed in to add_single().")
            else:
                objective = 0.0

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

    def clear(self):
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## Methods for reading from the archive ##
    ## Refer to ArchiveBase for documentation of these methods. ##

    # TODO: Refactor?
    def retrieve(self, measures):
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
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])

        return occupied[0], {field: arr[0] for field, arr in data.items()}

    def data(self, fields=None, return_type="dict"):
        return self._store.data(fields, return_type)

    def sample_elites(self, n):
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, elites = self._store.retrieve(selected_indices)
        return elites

    ## CQD Score ##

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
                "In ProximityArchive, dist_max must be passed "
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
