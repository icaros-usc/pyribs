"""Provides the Scheduler."""

import warnings
from collections import defaultdict

import numpy as np


class Scheduler:
    """A basic class that composes an archive with multiple emitters.

    To use this class, first create an archive and list of emitters for the QD
    algorithm. Then, construct the Scheduler with these arguments. Finally, repeatedly
    call :meth:`ask` to collect solutions to analyze, and return the objective and
    measures of those solutions **in the same order** using :meth:`tell`.

    As all solutions go into the same archive, the emitters passed in must emit
    solutions with the same dimension (that is, their ``solution_dim`` attribute must be
    the same).

    .. warning:: If constructing many emitters at once, do not pass something like
        ``[EmitterClass(...)] * 5``, as this creates a list with the same instance of
        ``EmitterClass`` in each position. Instead, use ``[EmitterClass(...) for _ in
        range 5]``, which creates 5 unique instances of ``EmitterClass``.

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, e.g.,
            :class:`~ribs.archives.GridArchive`.
        emitters (list of ribs.emitters.EmitterBase): A list of emitter objects,
            e.g., :class:`~ribs.emitters.EvolutionStrategyEmitter`.
        result_archive (ribs.archives.ArchiveBase): An additional archive where all
            solutions are added. For example, in CMA-MAE, this archive stores all the
            best-performing solutions, since the main archive does not store all the
            best-performing solutions.
        add_mode (str): Indicates how solutions should be added to the archive. The
            default is "batch", which adds all solutions with one call to
            :meth:`~ribs.archives.ArchiveBase.add`. Alternatively, use "single" to add
            the solutions one at a time with
            :meth:`~ribs.archives.ArchiveBase.add_single`. "single" mode is included for
            legacy reasons, as it was the only mode of operation in pyribs 0.4.0 and
            before. We highly recommend using "batch" mode since it is significantly
            faster.
    Raises:
        TypeError: The ``emitters`` argument was not a list of emitters.
        ValueError: The emitters passed in do not have the same solution dimensions.
        ValueError: There is no emitter passed in.
        ValueError: The same emitter instance was passed in multiple times. Each emitter
            should be a unique instance (see the warning above).
        ValueError: Invalid value for ``add_mode``.
        ValueError: The ``result_archive`` and ``archive`` are the same object
            (``result_archive`` should not be passed in this case).
    """

    def __init__(self, archive, emitters, result_archive=None, *, add_mode="batch"):
        try:
            if len(emitters) == 0:
                raise ValueError("Pass in at least one emitter to the scheduler.")
        except TypeError as exception:
            # TypeError will be raised by len(). We avoid directly checking if
            # `emitters` is an instance of list since we do not want to be too
            # restrictive.
            raise TypeError(
                "`emitters` must be a list of emitter objects."
            ) from exception

        emitter_ids = set(id(e) for e in emitters)
        if len(emitter_ids) != len(emitters):
            raise ValueError(
                "Not all emitters passed in were unique (i.e. some emitters "
                "had the same id). If emitters were created with something "
                "like [EmitterClass(...)] * n, instead use "
                "[EmitterClass(...) for _ in range(n)] so that all emitters "
                "are unique instances."
            )

        self._solution_dim = emitters[0].solution_dim

        for idx, emitter in enumerate(emitters[1:]):
            if emitter.solution_dim != self._solution_dim:
                raise ValueError(
                    "All emitters must have the same solution dim, but "
                    f"Emitter {idx} has dimension {emitter.solution_dim}, "
                    f"while Emitter 0 has dimension {self._solution_dim}"
                )

        if add_mode not in ["single", "batch"]:
            raise ValueError(
                f"add_mode must either be 'batch' or 'single', but it was '{add_mode}'"
            )

        if archive is result_archive:
            raise ValueError(
                "`archive` has same id as `result_archive` -- "
                "Note that `Scheduler.result_archive` already "
                "defaults to be the same as `archive` if you pass "
                "`result_archive=None`"
            )

        self._archive = archive
        self._emitters = emitters
        self._add_mode = add_mode

        self._result_archive = result_archive

        # Helps track which scheduler method should be called next.
        self._last_called = None
        # The last set of solutions returned by ask() or ask_dqd().
        self._cur_solutions = []
        # The number of solutions created by each emitter.
        self._num_emitted = [None for _ in self._emitters]

    @property
    def archive(self):
        """ribs.archives.ArchiveBase: Archive for storing solutions found in this
        scheduler."""
        return self._archive

    @property
    def emitters(self):
        """list of ribs.archives.EmitterBase: Emitters for generating solutions in this
        scheduler."""
        return self._emitters

    @property
    def result_archive(self):
        """ribs.archives.ArchiveBase: An additional archive for storing solutions found
        in this scheduler.

        If `result_archive` was not passed to the constructor, this property is
        the same as :attr:`archive`.
        """
        return self._archive if self._result_archive is None else self._result_archive

    def ask_dqd(self):
        """Generates a batch of solutions by calling ask_dqd() on all DQD emitters.

        .. note:: The order of the solutions returned from this method is important, so
            do not rearrange them.

        Returns:
            (batch_size, dim) array: An array of n solutions to evaluate. Each row
            contains a single solution.
        Raises:
            RuntimeError: This method was called immediately after calling an ask
                method.
        """
        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError(
                "ask_dqd cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask_dqd"

        self._cur_solutions = []

        for i, emitter in enumerate(self._emitters):
            emitter_sols = emitter.ask_dqd()
            self._cur_solutions.append(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        # In case the emitters didn't return any solutions.
        self._cur_solutions = (
            np.concatenate(self._cur_solutions, axis=0)
            if self._cur_solutions
            else np.empty((0, self._solution_dim))
        )
        return self._cur_solutions

    def ask(self):
        """Generates a batch of solutions by calling ask() on all emitters.

        .. note:: The order of the solutions returned from this method is important, so
            do not rearrange them.

        Returns:
            (batch_size, dim) array: An array of n solutions to evaluate. Each row
            contains a single solution.
        Raises:
            RuntimeError: This method was called immediately after calling an ask
                method.
        """
        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError(
                "ask cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask"

        self._cur_solutions = []

        for i, emitter in enumerate(self._emitters):
            emitter_sols = emitter.ask()
            self._cur_solutions.append(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        # In case the emitters didn't return any solutions.
        self._cur_solutions = (
            np.concatenate(self._cur_solutions, axis=0)
            if self._cur_solutions
            else np.empty((0, self._solution_dim))
        )
        return self._cur_solutions

    def _check_length(self, name, arr):
        """Raises a ValueError if array does not have the same length as the
        solutions."""
        if len(arr) != len(self._cur_solutions):
            raise ValueError(
                f"{name} should have length {len(self._cur_solutions)} "
                "(this is the number of solutions output by ask()) but "
                f"has length {len(arr)}"
            )

    def _validate_tell_data(self, data):
        """Preprocesses data passed into tell methods."""
        for name, arr in data.items():
            # Fields are allowed to be None to indicate they are not present, e.g.,
            # `objective` is None in diversity optimization.
            if arr is None:
                continue

            data[name] = np.asarray(arr)
            self._check_length(name, arr)

        # Convenient for solutions be part of data, so that everything is just one dict.
        data["solution"] = self._cur_solutions

        return data

    EMPTY_WARNING = (
        "`{name}` was empty before adding solutions, and it is still empty "
        "after adding solutions. "
        "One potential cause is that `threshold_min` is too high in this "
        "archive, i.e., solutions are not being inserted because their "
        "objective value does not exceed `threshold_min`."
    )

    def _add_to_archives(self, data):
        """Adds solutions to both the regular archive and the result archive."""

        archive_empty_before = self.archive.empty
        if self._result_archive is not None:
            # Check self._result_archive here since self.result_archive is a property
            # that always provides a proper archive.
            result_archive_empty_before = self.result_archive.empty

        # Add solutions to the archive.
        if self._add_mode == "batch":
            add_info = self.archive.add(**data)

            # Add solutions to result_archive.
            if self._result_archive is not None:
                self._result_archive.add(**data)
        elif self._add_mode == "single":
            add_info = defaultdict(list)

            for i in range(len(self._cur_solutions)):
                single_data = {
                    name: None if arr is None else arr[i] for name, arr in data.items()
                }
                single_info = self.archive.add_single(**single_data)
                for name, val in single_info.items():
                    add_info[name].append(val)

                # Add solutions to result_archive.
                if self._result_archive is not None:
                    self._result_archive.add_single(**single_data)

            for name, arr in add_info.items():
                add_info[name] = np.asarray(arr)

        # Warn the user if nothing was inserted into the archives.
        if archive_empty_before and self.archive.empty:
            warnings.warn(self.EMPTY_WARNING.format(name="archive"))
        if self._result_archive is not None:
            if result_archive_empty_before and self.result_archive.empty:
                warnings.warn(self.EMPTY_WARNING.format(name="result_archive"))

        return add_info

    def tell_dqd(self, objective, measures, jacobian, **fields):
        """Returns info for solutions from :meth:`ask_dqd`.

        .. note:: The objective, measures, and jacobian arrays must be in the same order
            as the solutions created by :meth:`ask_dqd`; i.e. ``objective[i]``,
            ``measures[i]``, and ``jacobian[i]`` should be the objective, measures, and
            jacobian for ``solution[i]``.

        Args:
            objective ((batch_size,) array or None): Each entry of this array contains
                the objective function evaluation of a solution. This can also be None
                to indicate there is no objective -- this would be the case in diversity
                optimization problems.
            measures ((batch_size, measure_dim) array): Each row of this array contains
                a solution's coordinates in measure space.
            jacobian (numpy.ndarray): ``(batch_size, 1 + measure_dim, solution_dim)``
                array consisting of Jacobian matrices of the solutions obtained from
                :meth:`ask_dqd`. Each matrix should consist of the objective gradient of
                the solution followed by the measure gradients.
            fields (keyword arguments): Additional data for each solution. Each argument
                should be an array with batch_size as the first dimension.
        Raises:
            RuntimeError: This method was called without first calling :meth:`ask_dqd`.
            ValueError: One of the inputs has the wrong shape.
        """
        if self._last_called != "ask_dqd":
            raise RuntimeError("tell_dqd() was called without calling ask_dqd().")
        self._last_called = "tell_dqd"

        data = self._validate_tell_data(
            {
                "objective": objective,
                "measures": measures,
                **fields,
            }
        )

        jacobian = np.asarray(jacobian)
        self._check_length("jacobian", jacobian)

        add_info = self._add_to_archives(data)

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self._emitters, self._num_emitted):
            end = pos + n
            emitter.tell_dqd(
                **{
                    name: None if arr is None else arr[pos:end]
                    for name, arr in data.items()
                },
                jacobian=jacobian[pos:end],
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end

    def tell(self, objective, measures, **fields):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective and measures arrays must be in the same order as the
            solutions created by :meth:`ask_dqd`; i.e. ``objective[i]`` and
            ``measures[i]`` should be the objective and measures for ``solution[i]``.

        Args:
            objective ((batch_size,) array or None): Each entry of this array contains
                the objective function evaluation of a solution. This can also be None
                to indicate there is no objective -- this would be the case in diversity
                optimization problems.
            measures ((batch_size, measures_dm) array): Each row of this array contains
                a solution's coordinates in measure space.
            fields (keyword arguments): Additional data for each solution. Each argument
                should be an array with batch_size as the first dimension.
        Raises:
            RuntimeError: This method was called without first calling :meth:`ask`.
            ValueError: One of the inputs has the wrong shape.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        data = self._validate_tell_data(
            {
                "objective": objective,
                "measures": measures,
                **fields,
            }
        )

        add_info = self._add_to_archives(data)

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self._emitters, self._num_emitted):
            end = pos + n
            emitter.tell(
                **{
                    name: None if arr is None else arr[pos:end]
                    for name, arr in data.items()
                },
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end
