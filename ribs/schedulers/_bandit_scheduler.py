"""Provides the Bandit Scheduler."""

import warnings
from collections import defaultdict

import numpy as np

from ribs._utils import readonly
from ribs.schedulers._scheduler import Scheduler


class BanditScheduler:
    """Schedules emitters with a bandit algorithm.

    This implementation is based on `Cully 2021 <https://arxiv.org/abs/2007.05352>`_.

    .. note:: This class follows the similar ask-tell framework as :class:`Scheduler`,
        and enforces similar constraints in the arguments and methods. Please refer to
        the documentation of :class:`Scheduler` for more details.

    .. note:: The main difference between :class:`BanditScheduler` and
        :class:`Scheduler` is that, unlike :class:`Scheduler`, DQD emitters are not
        supported by :class:`BanditScheduler`.

    To initialize this class, first create an archive and a list of emitters for the QD
    algorithm. The BanditScheduler will schedule the emitters using the Upper Confidence
    Bound - 1 algorithm (UCB1). Everytime :meth:`ask` is called, the emitters are sorted
    based on the potential reward function from UCB1. Then, the top `num_active`
    emitters are used for ask-tell.

    .. warning:: If constructing many emitters at once, do not pass something like
        ``[EmitterClass(...)] * 5``, as this creates a list with the same instance of
        ``EmitterClass`` in each position. Instead, use ``[EmitterClass(...) for _ in
        range 5]``, which creates 5 unique instances of ``EmitterClass``.

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, e.g.,
            :class:`~ribs.archives.GridArchive`.
        emitter_pool (list of ribs.archives.EmitterBase): A pool of emitters to select
            from, e.g. :class:`ribs.emitters.GaussianEmitter`. On the first iteration,
            the first `num_active` emitters from the emitter_pool will be activated.
        result_archive (ribs.archives.ArchiveBase): An additional archive where all
            solutions are added. For example, in CMA-MAE, this archive stores all the
            best-performing solutions, since the main archive does not store all the
            best-performing solutions.
        num_active (int): The number of active emitters at a time. Active emitters are
            used when calling ask-tell.
        reselect (str): Indicates how emitters are reselected from the pool. The default
            is "terminated", where only terminated/restarted emitters are deactivated
            and reselected (but they might be selected again). Alternatively, use "all"
            to reselect all active emitters every iteration.
        zeta (float): Hyperparamter of UCB1 that balances the trade-off between the
            accuracy and the uncertainty of the emitters. Increasing this parameter will
            emphasize the uncertainty of the emitters. Refer to the original paper for
            more information.
        add_mode (str): Indicates how solutions should be added to the archive. The
            default is "batch", which adds all solutions with one call to
            :meth:`~ribs.archives.ArchiveBase.add`. Alternatively, use "single" to add
            the solutions one at a time with
            :meth:`~ribs.archives.ArchiveBase.add_single`. "single" mode is included for
            legacy reasons, as it was the only mode of operation in pyribs 0.4.0 and
            before. We highly recommend using "batch" mode since it is significantly
            faster.
    Raises:
        TypeError: The ``emitter_pool`` argument was not a list of emitters.
        ValueError: Number of active emitters is less than one.
        ValueError: Fewer emitters in the pool than the number of active emitters.
        ValueError: The emitters passed in do not have the same solution dimensions.
        ValueError: The same emitter instance was passed in multiple times. Each
            emitter should be a unique instance (see the warning above).
        ValueError: Invalid value for ``add_mode``.
        ValueError: The ``result_archive`` and ``archive`` are the same object
            (``result_archive`` should not be passed in this case).
    """

    def __init__(
        self,
        archive,
        emitter_pool,
        result_archive=None,
        *,
        num_active,
        reselect="terminated",
        zeta=0.05,
        add_mode="batch",
    ):
        if num_active < 1:
            raise ValueError("num_active cannot be less than 1.")

        try:
            if len(emitter_pool) < num_active:
                raise ValueError(
                    f"The emitter pool must contain at least"
                    f"num_active emitters, but only"
                    f"{len(emitter_pool)} are given."
                )
        except TypeError as exception:
            # TypeError will be raised by len(). We avoid directly checking if
            # `emitter_pool` is an instance of list since we do not want to be too
            # restrictive.
            raise TypeError(
                "`emitter_pool` must be a list of emitter objects."
            ) from exception

        emitter_ids = set(id(e) for e in emitter_pool)
        if len(emitter_ids) != len(emitter_pool):
            raise ValueError(
                "Not all emitters passed in were unique (i.e. some emitters "
                "had the same id). If emitters were created with something "
                "like [EmitterClass(...)] * n, instead use "
                "[EmitterClass(...) for _ in range(n)] so that all emitters "
                "are unique instances."
            )

        self._solution_dim = emitter_pool[0].solution_dim

        for idx, emitter in enumerate(emitter_pool[1:]):
            if emitter.solution_dim != self._solution_dim:
                raise ValueError(
                    "All emitters must have the same solution dim, but "
                    f"Emitter {idx} has dimension {emitter.solution_dim}, "
                    f"while Emitter 0 has dimension {self._solution_dim}"
                )

        if reselect not in ["terminated", "all"]:
            raise ValueError(
                f"add_mode must either be 'terminated' or 'all',but it was '{reselect}'"
            )

        if add_mode not in ["single", "batch"]:
            raise ValueError(
                f"add_mode must either be 'batch' or 'single', but it was '{add_mode}'"
            )

        if archive is result_archive:
            raise ValueError(
                "`archive` has same id as `result_archive` -- "
                "Note that `BanditScheduler.result_archive` already "
                "defaults to be the same as `archive` if you pass "
                "`result_archive=None`"
            )

        self._archive = archive
        self._emitter_pool = np.array(emitter_pool)
        self._num_active = num_active
        self._add_mode = add_mode
        self._result_archive = result_archive
        self._reselect = reselect

        # Boolean mask of the active emitters. Initializes to the first num_active
        # emitters in the emitter pool.
        self._active_arr = np.zeros_like(self._emitter_pool, dtype=bool)

        # Used by UCB1 to select emitters.
        self._success = np.zeros_like(self._emitter_pool, dtype=float)
        self._selection = np.zeros_like(self._emitter_pool, dtype=float)
        self._restarts = np.zeros_like(self._emitter_pool, dtype=int)
        self._zeta = zeta

        # Helps track which scheduler method should be called next.
        self._last_called = None
        # The last set of solutions returned by ask().
        self._cur_solutions = []
        # The number of solutions created by each emitter.
        self._num_emitted = np.array([None for _ in self._active_arr])

    @property
    def archive(self):
        """ribs.archives.ArchiveBase: Archive for storing solutions found in this
        scheduler."""
        return self._archive

    @property
    def emitter_pool(self):
        """list of ribs.archives.EmitterBase: The pool of emitters available in the
        scheduler."""
        return self._emitter_pool

    @property
    def active(self):
        """numpy.ndarray: Boolean array indicating which emitters in the
        :attr:`emitter_pool` are currently active."""
        return readonly(self._active_arr.view())

    @property
    def result_archive(self):
        """ribs.archives.ArchiveBase: An additional archive for storing solutions found
        in this scheduler.

        If `result_archive` was not passed to the constructor, this property is
        the same as :attr:`archive`.
        """
        return self._archive if self._result_archive is None else self._result_archive

    def ask_dqd(self):
        """This method is not supported for this scheduler and throws an error.

        Raises:
            NotImplementedError: This method is not supported by this scheduler.
        """
        raise NotImplementedError("ask_dqd() is not supported by BanditScheduler.")

    def ask(self):
        """Generates a batch of solutions by calling ask() on all active emitters.

        The emitters used by ask are determined by the UCB1 algorithm. Briefly, emitters
        that have never been selected before are prioritized, then emitters are sorted
        in descending order based on the accurary of their past prediction.

        .. note:: The order of the solutions returned from this method is important, so
            do not rearrange them.

        Returns:
            (batch_size, dim) array: An array of n solutions to evaluate. Each row
            contains a single solution.
        Raises:
            RuntimeError: This method was called immediately after calling an ask
                method.
        """
        if self._last_called == "ask":
            raise RuntimeError(
                "ask cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask"

        if self._reselect == "terminated":
            # Reselect terminated emitters. Emitters are terminated if their restarts
            # attributes have incremented.
            emitter_restarts = np.array(
                [
                    emitter.restarts if hasattr(emitter, "restarts") else -1
                    for emitter in self._emitter_pool
                ]
            )
            reselect = emitter_restarts > self._restarts

            # If the emitter does not have "restarts" attribute, assume it restarts
            # every iteration.
            reselect[emitter_restarts < 0] = True

            self._restarts = emitter_restarts
        else:
            # Reselect all emitters.
            reselect = self._active_arr.copy()

        # If not enough emitters are active, activate the first num_active. This always
        # happens on the first iteration(s).
        num_needed = self._num_active - self._active_arr.sum()
        i = 0
        while num_needed > 0:
            reselect[i] = False
            if not self._active_arr[i]:
                self._active_arr[i] = True
                num_needed -= 1
            i += 1

        # Deactivate emitters to be reselected.
        self._active_arr[reselect] = False

        # Select emitters based on the UCB1 formula.
        # The ranking of emitters also follows these rules:
        # - Emitters that have never been selected are prioritized.
        # - If reselect is "terminated", then only active emitters that have
        #   terminated/restarted will be reselected. Otherwise, if reselect is "all",
        #   then all emitters are reselected.
        if reselect.any():
            ucb1 = np.full_like(
                self._emitter_pool, np.inf
            )  # np.inf forces to select emitters that were not yet selected
            update_ucb = self._selection != 0
            if update_ucb.any():
                ucb1[update_ucb] = self._success[update_ucb] / self._selection[
                    update_ucb
                ] + self._zeta * np.sqrt(
                    np.log(self._success.sum()) / self._selection[update_ucb]
                )
            # Activate top emitters based on UCB1, until there are num_active active
            # emitters. Activate only inactive emitters.
            activate = np.argsort(ucb1)[::-1]
            cur_active = self._active_arr.sum()
            for i in activate:
                if cur_active >= self._num_active:
                    break
                if not self._active_arr[i]:
                    self._active_arr[i] = True
                    cur_active += 1

        self._cur_solutions = []

        for i in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[i]
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

    # pylint: disable-next = protected-access
    _validate_tell_data = Scheduler._validate_tell_data

    def tell_dqd(self, objective, measures, jacobian):
        """This method is not supported for this scheduler and throws an error.

        Raises:
            NotImplementedError: This method is not supported by this scheduler.
        """
        raise NotImplementedError("tell_dqd() is not supported by BanditScheduler.")

    def tell(self, objective, measures, **fields):
        """Returns info for solutions from :meth:`ask`.

        The emitters are the same with those used in the last call to :meth:`ask`.

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
            warnings.warn(Scheduler.EMPTY_WARNING.format(name="archive"))
        if self._result_archive is not None:
            if result_archive_empty_before and self.result_archive.empty:
                warnings.warn(Scheduler.EMPTY_WARNING.format(name="result_archive"))

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for i in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[i]
            n = self._num_emitted[i]

            end = pos + n
            self._selection[i] += n
            self._success[i] += np.count_nonzero(add_info["status"][pos:end])
            emitter.tell(
                **{
                    name: None if arr is None else arr[pos:end]
                    for name, arr in data.items()
                },
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end
