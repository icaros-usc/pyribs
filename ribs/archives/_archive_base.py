"""Provides ArchiveBase."""

from __future__ import annotations

from abc import ABC
from collections.abc import Collection, Iterator
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.typing import BatchData, Int, SingleData


class ArchiveBase(ABC):
    """Base class for archives.

    An archive stores *elites*. Each elite consists of several data *fields*: at a
    minimum, the elite has a *solution* and the evaluated *objective* and *measures* of
    the solution. The elite may also include additional data fields. Besides elites,
    archives can store components like k-D trees and density estimators.

    The primary method of an archive is to write new solutions to it with :meth:`add`.
    There are also methods to read from the archive, such as :meth:`retrieve` and
    :meth:`data`. These methods typically operate over batches of inputs (e.g., adding
    multiple solutions at once with :meth:`add`), but methods such as :meth:`add_single`
    and :meth:`retrieve_single` support single inputs.

    Due to the flexibility of workflows available in pyribs, it is possible to design
    archives that require only a small subset of the methods in this base class. As
    such, none of the methods listed here are required to be implemented in child
    classes, although by default they will raise :class:`NotImplementedError` when
    called.

    Args:
        solution_dim: Dimensionality of the solution space. Scalar or multi-dimensional
            solution shapes are allowed by passing an empty tuple or tuple of integers,
            respectively.
        objective_dim: Dimensionality of the objective space. For single-objective
            optimization problems where the objective is a scalar, this argument should
            be an empty tuple ``()``. In multi-objective optimization problems, this
            argument should be an integer indicating the number of objectives.
        measure_dim: Dimensionality of the measure space.
    """

    def __init__(
        self,
        *,
        solution_dim: Int | tuple[Int, ...],
        objective_dim: tuple[()] | Int,
        measure_dim: Int,
    ) -> None:
        self._solution_dim = solution_dim
        self._objective_dim = objective_dim
        self._measure_dim = measure_dim

    ## Properties of the archive ##

    @property
    def solution_dim(self) -> Int | tuple[Int, ...]:
        """Dimensionality of the solution space."""
        return self._solution_dim

    @property
    def objective_dim(self) -> tuple[()] | Int:
        """Dimensionality of the objective space.

        The empty tuple ``()`` indicates a scalar objective.
        """
        return self._objective_dim

    @property
    def measure_dim(self) -> Int:
        """Dimensionality of the measure space."""
        return self._measure_dim

    @property
    def field_list(self) -> list[str]:
        """List of data fields in the archive."""
        raise NotImplementedError(
            "`field_list` has not been implemented in this archive"
        )

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Mapping from field name to dtype for all fields in the archive."""
        raise NotImplementedError("`dtypes` has not been implemented in this archive")

    @property
    def stats(self) -> ArchiveStats:
        """Statistics about the archive.

        See :class:`~ribs.archives.ArchiveStats` for more info.
        """
        raise NotImplementedError("`stats` has not been implemented in this archive")

    @property
    def empty(self) -> bool:
        """Whether the archive is empty."""
        raise NotImplementedError("`empty` has not been implemented in this archive")

    ## dunder methods ##

    def __len__(self) -> int:
        """Number of elites in the archive."""
        raise NotImplementedError("`__len__` has not been implemented in this archive")

    def __iter__(self) -> Iterator[SingleData]:
        """Creates an iterator over the elites in the archive.

        Example:
            ::

                for elite in archive:
                    elite["solution"]
                    elite["objective"]
                    elite["measures"]
                    ...
        """
        raise NotImplementedError("`__iter__` has not been implemented in this archive")

    ## Methods for writing to the archive ##

    def add(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> BatchData:
        """Inserts a batch of solutions and their data into the archive.

        The indices of all arguments should "correspond" to each other, i.e.,
        ``solution[i]``, ``objective[i]``, and ``measures[i]`` should be the solution
        parameters, objective, and measures for solution ``i``.

        For API consistency, all child classes should take in ``solution``,
        ``objective``, and ``measures``. There may be cases where one of these
        parameters is not necessary, e.g., ``objective`` is not required in diversity
        optimization settings. In such cases, it should be possible to pass in ``None``
        as the argument.

        Args:
            solution: (batch_size, :attr:`solution_dim`) array of solution parameters.
            objective: (batch_size, :attr:`objective_dim`) array with objective function
                evaluations of the solutions.
            measures: (batch_size, :attr:`measure_dim`) array with measure space
                coordinates of all the solutions.
            fields: Additional data for each solution. Each argument should be an array
                with ``batch_size`` as the first dimension.

        Returns:
            Dict describing the result of the add operation. Each entry should be an
            array that provides the information for each solution, e.g., one entry might
            be a "status" array of shape ``(batch_size,)`` that provides the status of
            each solution. The exact keys and values are determined by child classes.
        """
        raise NotImplementedError("`add` has not been implemented in this archive")

    def add_single(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> SingleData:
        """Inserts a single solution and its data into the archive.

        Args:
            solution: Parameters of the solution.
            objective: Objective function evaluation of the solution.
            measures: Coordinates in measure space of the solution.
            fields: Additional data for the solution.

        Returns:
            Information describing the result of the add operation. As in :meth:`add`,
            the content of this dict is determined by child classes.
        """
        raise NotImplementedError(
            "`add_single` has not been implemented in this archive"
        )

    def clear(self) -> None:
        """Resets the archive, e.g., by removing all elites in it.

        After calling this method, the archive should be :attr:`empty`.
        """
        raise NotImplementedError("`clear` has not been implemented in this archive")

    ## Methods for reading from the archive ##

    def retrieve(self, measures: ArrayLike) -> tuple[np.ndarray, BatchData]:
        """Queries the archive for elites with the given batch of measures.

        This method operates in batch. It takes in a batch of measures and outputs the
        batched data for the elites::

            occupied, elites = archive.retrieve(...)
            occupied  # Shape: (batch_size,)
            elites["solution"]  # Shape: (batch_size, solution_dim)
            elites["objective"]  # Shape: (batch_size, objective_dim)
            elites["measures"]  # Shape: (batch_size, measure_dim)
            ...

        ``occupied`` indicates whether an elite was found for each measure, i.e.,
        whether the archive was *occupied* at each queried measure. If ``occupied[i]``
        is True, then ``elites["solution"][i]``, ``elites["objective"][i]``,
        ``elites["measures"][i]``, and other fields will contain the data of the elite
        for the input ``measures[i]``. If ``occupied[i]`` is False, then those fields
        will instead have arbitrary values, e.g., ``elites["solution"][i]`` may be set
        to all NaN.

        Args:
            measures: (batch_size, :attr:`measure_dim`) array of measure space points at
                which to retrieve solutions.

        Returns:
            2-element tuple of (boolean ``occupied`` array, dict of elite data). See
            above for description.

        Raises:
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        raise NotImplementedError("`retrieve` has not been implemented in this archive")

    def retrieve_single(self, measures: ArrayLike) -> tuple[bool, SingleData]:
        """Queries the archive for an elite with the given measures.

        While :meth:`retrieve` takes in a *batch* of measures, this method takes in the
        measures for only *one* solution and returns a single bool and a dict with
        single entries::

            occupied, elite = archive.retrieve_single(...)
            occupied  # Bool
            elite["solution"]  # Shape: (solution_dim,)
            elite["objective"]  # Shape: (objective_dim,)
            elite["measures"]  # Shape: (measure_dim,)
            ...

        Args:
            measures: (:attr:`measure_dim`,) array of measures.

        Returns:
            2-element tuple of (boolean, dict of data for one elite)

        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        raise NotImplementedError(
            "`retrieve_single` has not been implemented in this archive"
        )

    @overload
    def data(
        self,
        fields: str,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["dict"] = "dict",
    ) -> BatchData: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["tuple"] = "tuple",
    ) -> tuple[np.ndarray]: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["pandas"] = "pandas",
    ) -> ArchiveDataFrame: ...

    def data(
        self,
        fields: None | Collection[str] | str = None,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray | BatchData | tuple[np.ndarray] | ArchiveDataFrame:
        """Returns data of the elites in the archive.

        Args:
            fields: List of fields to include, such as ``"solution"``, ``"objective"``,
                ``"measures"``, and other fields in the archive. This can also be a
                single str indicating a field name.
            return_type: Data to return; see below. Ignored if ``fields`` is a str.

        Returns:
            The data for all elites in the archive. All data returned by this method
            will be a copy, i.e., the data will not update as the archive changes. If
            ``fields`` was a single str, the returned data will just be an array holding
            data for the given field, such as::

                  measures = archive.data("measures")

            Otherwise, the returned data can take the following forms, depending on the
            ``return_type`` argument:

            - ``return_type="dict"``: Dict mapping from the field name to the field data
              at the given indices. An example is::

                  {
                    "solution": [[1.0, 1.0, ...], ...],
                    "objective": [1.5, ...],
                    "measures": [[1.0, 2.0], ...],
                    ...
                  }

              The keys in this dict can be modified with the ``fields`` arg; duplicate
              fields will be ignored since the dict stores unique keys.

            - ``return_type="tuple"``: Tuple of arrays matching the field order in
              ``fields``. For instance, if ``fields`` is ``["objective", "measures"]``,
              this method would return a tuple of ``(objective_arr, measures_arr)`` that
              could be unpacked as::

                  objective, measures = archive.data(["objective", "measures"],
                                                     return_type="tuple")

              Unlike with the ``dict`` return type, duplicate fields will show up as
              duplicate entries in the tuple, e.g., ``fields=["objective",
              "objective"]`` will result in two objective arrays being returned.

              When ``fields=None`` (the default case), the fields in the tuple will be
              ordered according to the :attr:`field_list`.

            - ``return_type="pandas"``: An :class:`~ribs.archives.ArchiveDataFrame` with
              the following columns:

              - For fields that are scalars, a single column with the field name. For
                example, ``objective`` would have a single column called ``objective``.
              - For fields that are 1D arrays, multiple columns with the name suffixed
                by its index. To illustrate, for a ``measures`` field of length 10, the
                dataframe would contain 10 columns with names ``measures_0``,
                ``measures_1``, ..., ``measures_9``. **The output format for fields with
                >1D data is currently not defined.**

              In short, the dataframe might look like this by default:

              +------------+------+-----------+------------+------+
              | solution_0 | ...  | objective | measures_0 | ...  |
              +============+======+===========+============+======+
              |            | ...  |           |            | ...  |
              +------------+------+-----------+------------+------+

              Like the other return types, the columns returned can be adjusted with the
              ``fields`` parameter.

        Raises:
            ValueError: Invalid field name provided.
            ValueError: Invalid return_type provided.
            ValueError: Passed ``return_type="pandas"`` when one of the fields has >1D
                data.
        """
        raise NotImplementedError("`data` has not been implemented in this archive")

    def sample_elites(self, n: Int, replace: bool = True) -> BatchData:
        """Randomly samples elites from the archive.

        Currently, this sampling is done uniformly at random. Additional sampling
        methods may be supported in the future.

        Example:
            ::

                elites = archive.sample_elites(16)
                elites["solution"]  # Shape: (16, solution_dim)
                elites["objective"]
                elites["measures"]
                ...

        Args:
            n: Number of elites to sample.
            replace: Whether to replace the elites when sampling. If True, the elites
                will be replaced and thus will be sampled independently.

        Returns:
            A batch of elites randomly selected from the archive.

        Raises:
            IndexError: The archive is empty.
            ValueError: ``n`` was greater than the number of elites in the archive when
                ``replace=False``.
        """
        raise NotImplementedError(
            "`sample_elites` has not been implemented in this archive"
        )
