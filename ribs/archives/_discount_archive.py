"""Contains the DiscountArchive."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike

from ribs._utils import validate_batch
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._utils import parse_all_dtypes
from ribs.discount_models import DiscountModelManager
from ribs.typing import BatchData, Float, Int

_RESULT_ARCHIVE_ERROR = "result_archive must be a GridArchive or CVTArchive."


class DiscountArchive(ArchiveBase):
    """Archive that represents the discount function with a model.

    The discount model maps from measures to discount values.

    This archive provides several configuration parameters related to the discount
    model. These parameters focus on how the *training data* is specified for the
    discount model, whereas the discount model itself handles the process of training on
    that data. Overall, we assume this archive can produce training data, and the
    discount model can successfully regress to this data.

    Args:
        solution_dim: Dimensionality of solutions.
        measure_dim: Dimensionality of the measures.
        learning_rate: The learning rate for discount updates.
        threshold_min: Minimum discount value. Used when initializing the discount model
            and when regressing to "empty points."
        discount_model: Model of the discount function.
        device: PyTorch device where the discount model is located.
        result_archive: The archive storing results for the algorithm. This is used for
            sampling empty points. Currently, only GridArchive and CVTArchive are
            supported.
        initial_train_points: Number of points to use for initializing the discount
            model.
        empty_points: Number of empty points to sample.
        train_freq: How often (in terms of iterations) to train the discount model.
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
        solution_dim: Int | tuple[Int, ...],
        measure_dim: Int,
        learning_rate: Float,
        threshold_min: Float,
        discount_model: DiscountModelManager,
        device: torch.device,
        result_archive: GridArchive | CVTArchive,
        initial_train_points: Int,
        empty_points: Int,
        train_freq: Int,
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

        self._discount_model = discount_model
        self._device = device
        self._result_archive = result_archive
        if not isinstance(result_archive, (GridArchive, CVTArchive)):
            raise ValueError(_RESULT_ARCHIVE_ERROR)

        self._initial_train_points = initial_train_points
        self._empty_points = empty_points
        self._train_freq = train_freq

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
    def discount_model(self) -> DiscountModelManager:
        """The discount model managed by this archive."""
        return self._discount_model

    @property
    def device(self) -> torch.device:
        """PyTorch device where the discount model is located."""
        return self._device

    @property
    def initial_train_points(self) -> Int:
        """Number of points to use for initializing the discount model."""
        return self._initial_train_points

    @property
    def empty_points(self) -> Int:
        """Number of empty points to sample."""
        return self._empty_points

    @property
    def train_freq(self) -> Int:
        """How often (in terms of iterations) to train the discount model."""
        return self._train_freq

    ## Training ##
    # We make the training methods public because they tend to be pretty intensive, so
    # it is helpful to call them in the main loop rather than only in the archive. This
    # way, the user gets exact control of when the training happens and can also look at
    # info like training statistics. Otherwise, __init__() and add() may take a while
    # since we would be training the discount model in those methods.

    def _sample_empty_archive_centers(self, n: int) -> np.ndarray:
        """Samples random cells in the result archive and returns their centers.

        For GridArchive, the "center" is the center of each grid cell. For CVTArchive,
        the center is the centroid of each cell.

        Args:
            n: Number of centers to sample. Note that the actual number of empty cells
                in the archive may be less than n, in which case fewer than n points
                will be returned.
        """
        if isinstance(self._result_archive, GridArchive):
            empty_indices = np.arange(self._result_archive.cells)[
                ~self._result_archive._store.occupied
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
                ~self._result_archive._store.occupied
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
        """Initializes the discount model so that it (roughly) outputs threshold_min everywhere.

        Returns:
            Dict with info from training.
        """
        empty_measures = self._sample_empty_archive_centers(self.initial_train_points)

        # TODO: Remove torch usage
        # TODO: Remove device usage
        train_measures = torch.tensor(
            empty_measures,
            dtype=torch.float32,
            device=self._device,
        )
        train_targets = torch.full(
            (len(train_measures),),
            float(self.threshold_min),
            dtype=torch.float32,
            device=self._device,
        )

        losses = self.discount_model.training_loop(train_measures, train_targets)

        return {
            # Number of points marked empty.
            "n_empty": len(empty_measures),
            # New measures from the emitters.
            "new_measures": np.empty((0, self.measure_dim)),
            # Measures that were marked as empty.
            "empty_measures": empty_measures,
            # Training losses.
            "losses": losses,
            # Training epochs.
            "epochs": len(losses),
        }

    def train_discount_model(self) -> dict:
        """Trains the discount model based on information from evaluations.

        Some of the data for training is cached when calling :meth:`add` and retrieved
        in this method.

        Returns:
            Dict with info from training.
        """
        data = self._cached_data
        add_info = self._cached_add_info

        empty_measures = self._sample_empty_archive_centers(self.empty_points)
        n_empty = len(empty_measures)

        measure_list = [
            data["measures"],
            empty_measures,
        ]
        target_list = [
            # Measures from the data result in the threshold update rule.
            np.where(
                data["objective"] > add_info["discount"],
                (1.0 - self.learning_rate) * add_info["discount"]
                + self.learning_rate * data["objective"],
                add_info["discount"],
            ),
            # Empty measures get threshold_min.
            np.full(len(empty_measures), self.threshold_min),
        ]

        train_measures = torch.tensor(
            np.concatenate(measure_list),
            dtype=torch.float32,
            device=self.device,
        )
        train_targets = torch.tensor(
            np.concatenate(target_list),
            dtype=torch.float32,
            device=self.device,
        )

        losses = self.discount_model.training_loop(train_measures, train_targets)

        return {
            # Number of points marked empty.
            "n_empty": n_empty,
            # New measures from the emitters.
            "new_measures": data["measures"],
            # Measures that were marked as empty.
            "empty_measures": empty_measures,
            # Training losses.
            "losses": losses,
            # Training epochs.
            "epochs": len(losses),
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

        discount = (
            self.discount_model.chunked_inference(data["measures"])
            .detach()
            .cpu()
            .numpy()
        )

        added = data["objective"] > discount
        status = 2 * added
        value = data["objective"] - discount
        add_info = {"status": status, "value": value, "discount": discount}
        self._cached_data = data
        self._cached_add_info = add_info
        return add_info
