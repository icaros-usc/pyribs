"""Contains the DiscountArchive."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ribs._utils import validate_batch
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._utils import parse_all_dtypes
from ribs.discount_models import DiscountModelManager
from ribs.typing import BatchData, Float, Int

_RESULT_ARCHIVE_ERROR = "result_archive must be a GridArchive or CVTArchive."


# Developer Note: The documentation for this class is hacked. To list new methods,
# manually modify the template in docs/_templates/autosummary/class.rst


class DiscountArchive(ArchiveBase):
    """Archive that represents the discount function with a model.

    This archive originates in the Discount Model Search algorithm in `Tjanaka 2026
    <https://discount-models.github.io/>`_ and is based around a discount model, i.e., a
    model that maps from measures to discount values. In short, the discount model
    serves as a smooth representation of the discount function, compared to the
    histogram representation in CMA-MAE.

    To elaborate, CMA-MAE computes improvement values using a "soft archive" that stores
    a discount value in each cell -- the soft archive is essentially a histogram of
    discount values. In contrast, this archive computes improvement values using a
    discount model, which provides a smooth representation of the discount function. As
    Discount Model Search proceeds, the discount model is trained using data from two
    sources. First is solutions sampled by the emitters, and second is "empty points",
    i.e., measure space points sampled at the centers of unoccupied cells in the
    ``result_archive``. The emitter solutions represent where the search is progressing,
    while the empty points force the discount model to maintain low values in areas of
    the archive that have not been explored yet.

    From an implementation perspective, this archive focuses on providing the correct
    *training data* to the discount model. This data consists of pairs of measure values
    and discount value targets. Meanwhile, a
    :class:`~ribs.discount_models.DiscountModelManager` handles the process of training
    on that data and performing inference. As such, this archive includes configuration
    parameters like ``threshold_min`` and ``empty_points``, while
    :class:`~ribs.discount_models.DiscountModelManager` takes in the discount model's
    neural network itself. Overall, we assume this archive can produce training data,
    and the discount model manager can successfully regress the discount model to that
    data.

    .. note::

        The usage for this archive is similar to other archives like
        :class:`~ribs.archives.GridArchive` in that it is initialized, passed to the
        scheduler, and then has solutions added to it. However, it differs in that users
        must also call the training functions :meth:`init_discount_model` (before
        starting the main loop of their algorithm) and :meth:`train_discount_model`
        (during the main loop of their algorithm, *after* calling
        :meth:`~ribs.schedulers.Scheduler.tell`).

        We adopt this API because the training methods can be computationally intensive.
        Allowing the user to call the methods themselves enables them to control exactly
        when the training happens; furthermore, they can collect information like
        training statistics. Otherwise, the discount model training would be placed in
        ``__init__()`` and ``add()``, potentially causing those methods to take a long
        time.

    Args:
        solution_dim: Dimensionality of the solution space.
        measure_dim: Dimensionality of the measure space.
        learning_rate: The learning rate for discount updates.
        threshold_min: Minimum discount value. Used when initializing the discount model
            and when regressing to empty points.
        discount_model_manager: An object that handles training and inference for the
            discount model. The archive accesses the discount model through this
            manager.
        result_archive: The archive storing results for the algorithm. This is used for
            sampling empty points. Currently, only GridArchive and CVTArchive are
            supported.
        init_train_points: Number of points to use for initializing the discount model.
        empty_points: Number of empty points to sample when training the discount model.
        seed: Value to seed the random number generator. Set to None to avoid a fixed
            seed.
        solution_dtype: Data type of the solutions. Defaults to float64 (numpy's default
            floating point type).
        objective_dtype: Data type of the objectives. Defaults to float64 (numpy's
            default floating point type).
        measures_dtype: Data type of the measures. Defaults to float64 (numpy's default
            floating point type).
        dtype: Shortcut for providing data type of the solutions, objectives, and
            measures. Defaults to float64 (numpy's default floating point type). This
            parameter sets all the dtypes simultaneously. To set individual dtypes, pass
            ``solution_dtype``, ``objective_dtype``, and ``measures_dtype``. Note that
            ``dtype`` cannot be used at the same time as those parameters.

    Raises:
        ValueError: Invalid values were provided for arguments.
    """

    def __init__(
        self,
        *,
        solution_dim: Int | tuple[Int, ...],
        measure_dim: Int,
        learning_rate: Float,
        threshold_min: Float,
        discount_model_manager: DiscountModelManager,
        result_archive: GridArchive | CVTArchive,
        init_train_points: Int,
        empty_points: Int,
        seed: Int | None = None,
        solution_dtype: DTypeLike = None,
        objective_dtype: DTypeLike = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        solution_dtype, objective_dtype, measures_dtype = parse_all_dtypes(
            dtype, solution_dtype, objective_dtype, measures_dtype, np
        )
        self._dtypes = {
            "solution": solution_dtype,
            "objective": objective_dtype,
            "measures": measures_dtype,
        }
        self._rng = np.random.default_rng(seed)

        if not np.isfinite(threshold_min):
            raise ValueError("Unlike in CMA-MAE, threshold_min must be a finite value.")
        self._learning_rate = np.asarray(learning_rate, dtype=self.dtypes["measures"])
        self._threshold_min = np.asarray(threshold_min, dtype=self.dtypes["measures"])

        self._discount_model_manager = discount_model_manager
        self._result_archive = result_archive
        if not isinstance(result_archive, (GridArchive, CVTArchive)):
            raise ValueError(_RESULT_ARCHIVE_ERROR)

        self._init_train_points = init_train_points
        self._empty_points = empty_points

        # The data and add_info are cached during add() so that they can be used in
        # train_discount_model().
        self._cached_data = None
        self._cached_add_info = None

    ## Properties inherited from ArchiveBase ##

    # Necessary to implement this since `Scheduler` calls it.
    @property
    def empty(self) -> bool:
        """Whether the archive is empty; always ``False``.

        Since the archive does not store elites, we always mark it as not empty.
        """
        return False

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return self._dtypes

    ## Properties that are not in ArchiveBase ##

    @property
    def learning_rate(self) -> float:
        """The learning rate for discount updates."""
        return self._learning_rate

    @property
    def threshold_min(self) -> float:
        """Minimum discount value."""
        return self._threshold_min

    @property
    def discount_model_manager(self) -> DiscountModelManager:
        """The discount model manager for this archive."""
        return self._discount_model_manager

    @property
    def init_train_points(self) -> Int:
        """Number of points to use for initializing the discount model."""
        return self._init_train_points

    @property
    def empty_points(self) -> Int:
        """Number of empty points to sample."""
        return self._empty_points

    ## Training ##

    def _sample_empty_archive_centers(self, n: int) -> np.ndarray:
        """Samples random cells in the result archive and returns their centers.

        For GridArchive, the "center" is the center of each grid cell. For CVTArchive,
        the center is the centroid of each cell.

        Args:
            n: Number of centers to sample. Note that the actual number of empty cells
                in the archive may be fewer than n, in which case fewer than n points
                will be returned.
        """
        if isinstance(self._result_archive, GridArchive):
            empty_indices = np.arange(self._result_archive.cells)[
                ~self._result_archive._store.occupied  # pylint: disable = protected-access
            ]
            empty_indices = self._rng.choice(
                empty_indices,
                size=min(len(empty_indices), n),
                replace=False,
            )
            empty_grid_indices = self._result_archive.int_to_grid_index(empty_indices)

            # Find the center of the corresponding cells.
            empty_measures = (
                (empty_grid_indices + 0.5) / self._result_archive.dims
            ) * self._result_archive.interval_size + self._result_archive.lower_bounds
            return empty_measures

        elif isinstance(self._result_archive, CVTArchive):
            # Sample empty indices in the CVT archive to determine where
            # the threshold should be held at discount_min.
            empty_indices = np.arange(self._result_archive.cells)[
                ~self._result_archive._store.occupied  # pylint: disable = protected-access
            ]
            empty_indices = self._rng.choice(
                empty_indices,
                size=min(len(empty_indices), n),
                replace=False,
            )
            empty_measures = self._result_archive.centroids[empty_indices]
            return empty_measures

        else:
            raise ValueError(_RESULT_ARCHIVE_ERROR)

    def init_discount_model(self) -> dict:
        """Initializes the discount model so that it outputs threshold_min everywhere.

        To obtain measure values, this method samples :attr:`init_train_points` centers
        from the result archive. The discount value target for each measure is set to
        :attr:`threshold_min`. Finally,
        :meth:`ribs.discount_models.DiscountModelManager.training_loop` is called to
        train the discount model with this data.

        Returns:
            Info from training. See :meth:`train_discount_model` for info on this dict.
            The format is identical, but note that "solution_measures" and
            "solution_targets" are arrays of length 0.
        """
        empty_measures = self._sample_empty_archive_centers(self.init_train_points)
        empty_targets = np.full(len(empty_measures), self.threshold_min)

        losses = self.discount_model_manager.training_loop(
            empty_measures, empty_targets
        )

        return {
            "solution_measures": np.empty((0, self.measure_dim)),
            "solution_targets": np.empty((0,)),
            "empty_measures": empty_measures,
            "epochs": len(losses),
            "losses": losses,
        }

    def train_discount_model(self) -> dict:
        """Trains the discount model.

        The training process is described in Section 5 of `Tjanaka 2026
        <https://discount-models.github.io/>`_. The first data source for training the
        discount model comes from solutions sampled by the emitters -- this data is
        cached in the archive during a prior call to :meth:`add`, which happens inside
        the scheduler's :meth:`~ribs.schedulers.Scheduler.tell` method. The second data
        source, "empty points", are sampled from the centers of unoccupied cells in the
        result archive. The number of empty points is controlled by the
        :attr:`empty_points` property.

        Returns:
            Info from training. The dict contains the following keys:

            - "solution_measures": Measures associated with solutions sampled by the
              emitters, i.e., measures that were previously passed into :meth:`add`.
            - "solution_targets": Array of target discount values for the aforementioned
              "solution_measures". Note that the target for empty points is always
              :attr:`threshold_min`, so we do not include an "empty_targets" key.
            - "empty_measures": Array of empty points sampled from the result archive.
              Note that the number of points may be fewer than :attr:`empty_points` if
              the result archive did not have enough unoccupied cells.
            - "epochs": Number of epochs for which the discount model was trained.
            - "losses": List with loss from each epoch of training the discount model.
        """
        data = self._cached_data
        add_info = self._cached_add_info

        empty_measures = self._sample_empty_archive_centers(self.empty_points)

        # Measures from the solutions result in the threshold update rule adapted from
        # CMA-MAE (Equation 1 in Tjanaka 2026).
        solution_targets = np.where(
            data["objective"] > add_info["discount"],
            (1.0 - self.learning_rate) * add_info["discount"]
            + self.learning_rate * data["objective"],
            add_info["discount"],
        )

        # Empty measures get threshold_min.
        empty_targets = np.full(len(empty_measures), self.threshold_min)

        train_measures = np.concatenate((data["measures"], empty_measures))
        train_targets = np.concatenate((solution_targets, empty_targets))

        losses = self.discount_model_manager.training_loop(
            train_measures, train_targets
        )

        return {
            "solution_measures": data["measures"],
            "solution_targets": solution_targets,
            "empty_measures": empty_measures,
            "epochs": len(losses),
            "losses": losses,
        }

    ## Methods for writing to the archive ##

    def add(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> BatchData:
        """Computes the improvement values for the given solutions.

        Unlike other archives (see :meth:`ribs.archives.ArchiveBase.add`), this archive
        does not store any solutions itself. Rather, calling this function only results
        in a computation of the improvement value, based on the discount values output
        by the discount model. Training of the discount model happens separately in
        :meth:`train_discount_model`.

        Args:
            solution: (batch_size, :attr:`solution_dim`) array of solution parameters.
            objective: (batch_size, :attr:`objective_dim`) array with objective function
                evaluations of the solutions.
            measures: (batch_size, :attr:`measure_dim`) array with measure space
                coordinates of all the solutions.
            fields: Additional data for each solution. Each argument should be an array
                with ``batch_size`` as the first dimension.

        Returns:
            Information describing the result of the add operation. The dict contains
            the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`np.int32`): An array of
              integers that represent the "status" obtained when attempting to insert
              each solution in the batch. If a solution exceeds the discount value
              output by the discount model, it is considered to be a "new" solution and
              thus assigned a status of 2, in accordance with :class:`AddStatus`.
              Otherwise, it is considered to be "not added" to the archive and thus
              assigned a status of 0.

            - ``"value"`` (:class:`numpy.ndarray` of the objective dtype): An array of
              improvement values, computed by subtracting the discount values output by
              the discount model from the objective values passed in.

            - ``"discount"`` (:class:`numpy.ndarray` of the objective dtype): An array
              with the discount values computed by the discount model for each solution.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf or
                NaN).
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

        discount = self.discount_model_manager.inference(data["measures"])

        # Cast so that we can remain faithful to the archive's dtypes.
        discount = discount.astype(self.dtypes["objective"])

        added = data["objective"] > discount
        status = (2 * added).astype(np.int32)
        value = data["objective"] - discount
        add_info = {"status": status, "value": value, "discount": discount}
        self._cached_data = data
        self._cached_add_info = add_info
        return add_info
