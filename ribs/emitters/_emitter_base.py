"""Provides EmitterBase."""

from __future__ import annotations

import numbers
from abc import ABC
from collections.abc import Collection

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ribs._utils import check_shape
from ribs.archives import ArchiveBase
from ribs.typing import BatchData, Float, Int


class EmitterBase(ABC):
    """Base class for emitters.

    Every emitter has an :meth:`ask` method that generates a batch of solutions, and a
    :meth:`tell` method that inserts solutions into the emitter's archive. Child classes
    are only required to override :meth:`ask`.

    DQD emitters should also override :meth:`ask_dqd` and :meth:`tell_dqd` methods.

    Args:
        archive: Archive of solutions, e.g., :class:`ribs.archives.GridArchive`.
        solution_dim: The dimensionality of solutions produced by this emitter.
        bounds: Bounds of the solution space. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each dim. Each
            element in this array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or ``upper_bound`` may
            be None to indicate no bound. Unbounded upper bounds are set to +inf, and
            unbounded lower bounds are set to -inf.
        lower_bounds: Instead of specifying ``bounds``, ``lower_bounds`` and
            ``upper_bounds`` may be specified. This is useful if, for instance,
            solutions are multi-dimensional. Here, pass None to indicate there are no
            bounds (i.e., bounds are set to -inf), or pass an array specifying the lower
            bounds of the solution space.
        upper_bounds: Upper bounds of the solution space; see ``lower_bounds`` above.
            Pass None to indicate there are no bounds (i.e., bounds are set to inf).

    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(
        self,
        archive: ArchiveBase,
        *,
        solution_dim: Int | tuple[Int, ...],
        bounds: Collection[tuple[None | Float, None | Float]] | None,
        lower_bounds: ArrayLike | None,
        upper_bounds: ArrayLike | None,
    ) -> None:
        self._archive = archive
        self._solution_dim = solution_dim

        # Bounds handling.
        use_bounds = bounds is not None
        use_lower_upper = lower_bounds is not None or upper_bounds is not None
        if use_bounds and use_lower_upper:
            raise ValueError(
                "Cannot specify both bounds and lower_bounds/upper_bounds; "
                "either specify bounds or specify lower_bounds/upper_bounds."
            )
        elif use_bounds:
            (self._lower_bounds, self._upper_bounds) = self._process_bounds(
                bounds, self._solution_dim, archive.dtypes["solution"]
            )
        else:
            # Covers both `use_lower_upper` and the default case where no bounds are
            # passed in.
            self._lower_bounds = (
                np.full(solution_dim, -np.inf, dtype=archive.dtypes["solution"])
                if lower_bounds is None
                else np.asarray(lower_bounds, dtype=archive.dtypes["solution"])
            )
            self._upper_bounds = (
                np.full(solution_dim, np.inf, dtype=archive.dtypes["solution"])
                if upper_bounds is None
                else np.asarray(upper_bounds, dtype=archive.dtypes["solution"])
            )
            check_shape(
                self._lower_bounds, "lower_bounds", self._solution_dim, "solution_dim"
            )
            check_shape(
                self._upper_bounds, "upper_bounds", self._solution_dim, "solution_dim"
            )

    @staticmethod
    def _process_bounds(
        bounds: Collection[tuple[None | Float, None | Float]],
        solution_dim: Int,
        dtype: DTypeLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Processes the input bounds.

        Returns:
            tuple: Two arrays containing all the lower bounds and all the upper bounds.

        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        lower_bounds = np.full(solution_dim, -np.inf, dtype=dtype)
        upper_bounds = np.full(solution_dim, np.inf, dtype=dtype)

        if bounds is None:
            return lower_bounds, upper_bounds

        # Handle array-like bounds.
        if len(bounds) != solution_dim:
            raise ValueError(
                "If it is an array-like, bounds must have length solution_dim"
            )
        for idx, bnd in enumerate(bounds):
            if bnd is None:
                continue  # Bounds already default to -inf and inf.
            if len(bnd) != 2:
                raise ValueError("All entries of bounds must be length 2")
            lower_bounds[idx] = -np.inf if bnd[0] is None else bnd[0]
            upper_bounds[idx] = np.inf if bnd[1] is None else bnd[1]
        return lower_bounds, upper_bounds

    @property
    def archive(self) -> ArchiveBase:
        """Stores solutions generated by this emitter."""
        return self._archive

    @property
    def solution_dim(self) -> Int | tuple[Int, ...]:
        """Dimensionality of solutions produced by this emitter."""
        return self._solution_dim

    @property
    def lower_bounds(self) -> np.ndarray:
        """``(solution_dim,)`` array with lower bounds of solution space.

        For instance, ``[-1, -1, -1]`` indicates that every dimension of the solution
        space has a lower bound of -1.
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """``(solution_dim,)`` array with upper bounds of solution space.

        For instance, ``[1, 1, 1]`` indicates that every dimension of the solution space
        has an upper bound of 1.
        """
        return self._upper_bounds

    def ask(self) -> np.ndarray:
        """Generates a ``(batch_size, solution_dim)`` array of solutions.

        Returns an empty array by default.
        """
        solution_dim = (
            (self.solution_dim,)
            if isinstance(self.solution_dim, numbers.Integral)
            else self.solution_dim
        )
        return np.empty((0, *solution_dim), dtype=self.archive.dtypes["solution"])

    def tell(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> None:
        """Gives the emitter results from evaluating solutions.

        This base class implementation (in :class:`~ribs.emitters.EmitterBase`) does
        nothing by default.

        Args:
            solution: Array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective: 1D array containing the objective function value of each
                solution.
            measures: ``(n, <measure space dimension>)`` array with the measure space
                coordinates of each solution.
            add_info: Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields: Additional data for each solution. Each argument should be an array
                with batch_size as the first dimension.
        """

    def ask_dqd(self) -> np.ndarray:
        """Generates solutions for which gradient information must be computed.

        The solutions should be a ``(batch_size, solution_dim)`` array.

        This method only needs to be implemented by emitters used in DQD. It returns an
        empty array by default.
        """
        solution_dim = (
            (self.solution_dim,)
            if isinstance(self.solution_dim, numbers.Integral)
            else self.solution_dim
        )
        return np.empty((0, *solution_dim), dtype=self.archive.dtypes["solution"])

    def tell_dqd(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        jacobian: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> None:
        """Gives the emitter results from evaluating the gradient of the solutions.

        This method is the counterpart of :meth:`ask_dqd`. It is only used by DQD
        emitters.

        Args:
            solution: ``(batch_size, :attr:`solution_dim`)`` array of solutions
                generated by this emitter's :meth:`ask()` method.
            objective: 1-dimensional array containing the objective function value of
                each solution.
            measures: ``(batch_size, measure space dimension)`` array with the measure
                space coordinates of each solution.
            jacobian: ``(batch_size, 1 + measure_dim, solution_dim)`` array consisting
                of Jacobian matrices of the solutions obtained from :meth:`ask_dqd`.
                Each matrix should consist of the objective gradient of the solution
                followed by the measure gradients.
            add_info: Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields: Additional data for each solution. Each argument should be an array
                with batch_size as the first dimension.
        """
