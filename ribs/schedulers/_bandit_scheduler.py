"""Provides the Bandit Scheduler."""
import warnings

import numpy as np

from ribs.schedulers._scheduler import Scheduler


class BanditScheduler:
    """Schedules emitters with a bandit algorithm.

    This implementation is based on `Cully 2021
    <https://arxiv.org/abs/2007.05352>`_.

    .. note::
        This class follows the similar ask-tell framework as
        :class:`Scheduler`, and enforces similar constraints in the arguments
        and methods. Please refer to the documentation of :class:`Scheduler`
        for more details.

    .. note::
        The main difference between :class:`BanditScheduler` and
        :class:`Scheduler` is that, unlike :class:`Scheduler`, DQD emitters are
        not supported by :class:`BanditScheduler`.

    To initialize this class, first create an archive and a list of emitters
    for the QD algorithm. The BanditScheduler will schedule the emitters using
    the Upper Confidence Bound - 1 algorithm (UCB1). Everytime :meth:`ask` is
    called, the emitters are sorted based on the potential reward function from
    UCB1. Then, the top `num_active` emitters are used for ask-tell.

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, e.g. one
            selected from :mod:`ribs.archives`.
        emitter_pool (list of ribs.archives.EmitterBase): A pool of emitters to
            select from, e.g. :class:`ribs.emitters.GaussianEmitter`. On the
            first iteration, the first `num_active` emitters from the
            emitter_pool will be activated.
        num_active (int): The number of active emitters at a time. Active
            emitters are used when calling ask-tell.
        zeta (float): Hyperparamter of UCB1 that balances the trade-off between
            the accuracy and the uncertainty of the emitters. Increasing this
            parameter will emphasize the uncertainty of the emitters. Refer to
            the original paper for more information.
        reselect (str): Indicates how emitters are reselected from the pool.
            The default is "terminated", where only terminated/restarted
            emitters are deactivated and reselected (but they might be selected
            again). Alternatively, use "all" to reselect all active emitters
            every iteration.
        add_mode (str): Indicates how solutions should be added to the archive.
            The default is "batch", which adds all solutions with one call to
            :meth:`~ribs.archives.ArchiveBase.add`. Alternatively, use "single"
            to add the solutions one at a time with
            :meth:`~ribs.archives.ArchiveBase.add_single`. "single" mode is
            included for legacy reasons, as it was the only mode of operation
            in pyribs 0.4.0 and before. We highly recommend using "batch" mode
            since it is significantly faster.
        result_archive (ribs.archives.ArchiveBase): In some algorithms, such as
            CMA-MAE, the archive does not store all the best-performing
            solutions. The `result_archive` is a secondary archive where we can
            store all the best-performing solutions.
    Raises:
        ValueError: Number of active emitter is less than one.
        ValueError: Less emitters in the pool than the number of active
            emitters.
        ValueError: The emitters passed in do not have the same solution
            dimensions.
        ValueError: The same emitter instance was passed in multiple times.
            Each emitter should be a unique instance (see the warning above).
        ValueError: Invalid value for `add_mode`.
    """

    def __init__(self,
                 archive,
                 emitter_pool,
                 num_active,
                 *,
                 reselect="terminated",
                 zeta=0.05,
                 result_archive=None,
                 add_mode="batch"):
        if num_active < 1:
            raise ValueError("num_active cannot be less than 1.")

        if len(emitter_pool) < num_active:
            raise ValueError(f"The emitter pool must contain at least"
                             f"num_active emitters, but only"
                             f"{len(emitter_pool)} are given.")

        emitter_ids = set(id(e) for e in emitter_pool)
        if len(emitter_ids) != len(emitter_pool):
            raise ValueError(
                "Not all emitters passed in were unique (i.e. some emitters "
                "had the same id). If emitters were created with something "
                "like [EmitterClass(...)] * n, instead use "
                "[EmitterClass(...) for _ in range(n)] so that all emitters "
                "are unique instances.")

        self._solution_dim = emitter_pool[0].solution_dim

        for idx, emitter in enumerate(emitter_pool[1:]):
            if emitter.solution_dim != self._solution_dim:
                raise ValueError(
                    "All emitters must have the same solution dim, but "
                    f"Emitter {idx} has dimension {emitter.solution_dim}, "
                    f"while Emitter 0 has dimension {self._solution_dim}")

        if reselect not in ["terminated", "all"]:
            raise ValueError("add_mode must either be 'terminated' or 'all',"
                             f"but it was '{reselect}'")

        if add_mode not in ["single", "batch"]:
            raise ValueError("add_mode must either be 'batch' or 'single', but "
                             f"it was '{add_mode}'")

        if archive is result_archive:
            raise ValueError("`archive` has same id as `result_archive` -- "
                             "Note that `Scheduler.result_archive` already "
                             "defaults to be the same as `archive` if you pass "
                             "`result_archive=None`")

        self._archive = archive
        self._emitter_pool = np.array(emitter_pool)
        self._num_active = num_active
        self._add_mode = add_mode
        self._result_archive = result_archive
        self._reselect = reselect

        # Boolean mask of the active emitters. Initializes to the first
        # num_active emitters in the emitter pool.
        self._active_arr = np.zeros_like(self._emitter_pool, dtype=bool)

        # Used by UCB1 to select emitters.
        self._success = np.zeros_like(self._emitter_pool, dtype=float)
        self._selection = np.zeros_like(self._emitter_pool, dtype=float)
        self._restarts = np.zeros_like(self._emitter_pool, dtype=int)
        self._zeta = zeta

        # Keeps track of whether the scheduler should be receiving a call to
        # ask() or tell().
        self._last_called = None
        # The last set of solutions returned by ask().
        self._solution_batch = []
        # The number of solutions created by each emitter.
        self._num_emitted = np.array([None for _ in self._active_arr])

    @property
    def archive(self):
        """ribs.archives.ArchiveBase: Archive for storing solutions found in
        this scheduler."""
        return self._archive

    @property
    def emitters(self):
        """list of ribs.archives.EmitterBase: Emitters for generating solutions
        in this scheduler."""
        return self._active_arr

    @property
    def result_archive(self):
        """ribs.archives.ArchiveBase: Another archive for storing solutions
        found in this optimizer.
        If `result_archive` was not passed to the constructor, this property is
        the same as :attr:`archive`.
        """
        return (self._archive
                if self._result_archive is None else self._result_archive)

    def ask_dqd(self):
        """Generates a batch of solutions by calling ask_dqd() on all DQD
        emitters.

        This method is not supported for this scheduler and throws an error if
        called.

        Raises:
            NotImplementedError: This method is not supported by this
                scheduler.
        """
        raise NotImplementedError("ask_dqd() is not supported by"
                                  "BanditScheduler.")

    def ask(self):
        """Generates a batch of solutions by calling ask() on all active
        emitters.

        The emitters used by ask are determined by the UCB1 algorithm. Briefly,
        emitters that have never been selected before are prioritized, then
        emitters are sorted in descending order based on the accurary of their
        past prediction.

        .. note:: The order of the solutions returned from this method is
            important, so do not rearrange them.

        Returns:
            (batch_size, dim) array: An array of n solutions to evaluate. Each
            row contains a single solution.
        Raises:
            RuntimeError: This method was called without first calling
                :meth:`tell`.
        """
        if self._last_called == "ask":
            raise RuntimeError("ask cannot be called immediately after " +
                               self._last_called)
        self._last_called = "ask"

        if self._reselect == "terminated":
            # Reselect terminated emitters. Emitters are terminated if their
            # restarts attribute have incremented.
            emitter_restarts = np.array([
                emitter.restarts if hasattr(emitter, "restarts") else -1
                for emitter in self._emitter_pool
            ])
            reselect = emitter_restarts > self._restarts

            # If the emitter does not have "restarts" attribute, assume it
            # restarts every iteration.
            reselect[emitter_restarts < 0] = True

            self._restarts = emitter_restarts
        else:
            # Reselect all emitters.
            reselect = self._active_arr.copy()

        # If no emitters are active, activate the first num_active.
        if not self._active_arr.any():
            reselect[:] = False
            self._active_arr[:self._num_active] = True

        # Deactivate emitters to be reselected.
        self._active_arr[reselect] = False

        # Select emitters based on the UCB1 formula.
        # The ranking of emitters also follows these rules:
        # - Emitters that have never been selected are prioritized.
        # - If reselect is "terminated", then only active emitters that have
        #   terminated/restarted will be reselected. Otherwise, if reselect is
        #   "all", then all emitters are reselected.
        if reselect.any():
            ucb1 = np.full_like(self._emitter_pool, np.inf)
            update_ucb = (self._selection != 0)
            if update_ucb.any():
                ucb1[update_ucb] = (
                    self._success[update_ucb] / self._selection[update_ucb] +
                    self._zeta * np.sqrt(
                        np.log(self._success.sum()) /
                        self._selection[update_ucb]))
            # Activate top emitters based on UCB1.
            activate = np.argsort(ucb1)[-reselect.sum():]
            self._active_arr[activate] = True

        self._solution_batch = []

        for i in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[i]
            emitter_sols = emitter.ask()
            self._solution_batch.append(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        # In case the emitters didn't return any solutions.
        self._solution_batch = np.concatenate(
            self._solution_batch, axis=0) if self._solution_batch else np.empty(
                (0, self._solution_dim))
        return self._solution_batch

    def _check_length(self, name, array):
        """Raises a ValueError if array does not have the same length as the
        solutions."""
        if len(array) != len(self._solution_batch):
            raise ValueError(
                f"{name} should have length {len(self._solution_batch)} "
                "(this is the number of solutions output by ask()) but "
                f"has length {len(array)}")

    def tell_dqd(self,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 metadata_batch=None):
        """Returns info for solutions from :meth:`ask_dqd`.

        This method is not supported for this scheduler and throws an error if
        called.

        Raises:
            NotImplementedError: This method is not supported by this
                scheduler.
        """
        raise NotImplementedError("tell_dqd() is not supported by"
                                  "BanditScheduler.")

    def tell(self, objective_batch, measures_batch, metadata_batch=None):
        """Returns info for solutions from :meth:`ask`.

        The emitters are the same with those used in the last call to
        :meth:`ask`.

        .. note:: The objective batch, measures batch, and metadata batch must
            be in the same order as the solutions created by :meth:`ask`;
            i.e.  ``objective_batch[i]``, ``measures_batch[i]``, and
            ``metadata_batch[i]`` should be the objective, measures, and
            metadata for ``solution_batch[i]``.

        Args:
            objective_batch ((batch_size,) array): Each entry of this array
                contains the objective function evaluation of a solution.
            measures_batch ((batch_size, measures_dm) array): Each row of
                this array contains a solution's coordinates in measure space.
            metadata_batch ((batch_size,) array): Each entry of this array
                contains an object holding metadata for a solution.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
            ValueError: ``objective_batch``, ``measures_batch``, or
                ``metadata_batch`` has the wrong shape.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        metadata_batch = (np.empty(len(self._solution_batch), dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        self._check_length("objective_batch", objective_batch)
        self._check_length("measures_batch", measures_batch)
        self._check_length("metadata_batch", metadata_batch)

        archive_empty_before = self.archive.empty
        if self._result_archive is not None:
            # Check self._result_archive here since self.result_archive is a
            # property that always provides a proper archive.
            result_archive_empty_before = self.result_archive.empty

        # Add solutions to the archive.
        if self._add_mode == "batch":
            status_batch, value_batch = self.archive.add(
                self._solution_batch,
                objective_batch,
                measures_batch,
                metadata_batch,
            )

            # Add solutions to result_archive.
            if self._result_archive is not None:
                self._result_archive.add(self._solution_batch, objective_batch,
                                         measures_batch, metadata_batch)
        elif self._add_mode == "single":
            status_batch = []
            value_batch = []
            for solution, objective, measure, metadata in zip(
                    self._solution_batch, objective_batch, measures_batch,
                    metadata_batch):
                status, value = self.archive.add_single(solution, objective,
                                                        measure, metadata)
                status_batch.append(status)
                value_batch.append(value)

                # Add solutions to result_archive.
                if self._result_archive is not None:
                    self._result_archive.add_single(solution, objective,
                                                    measure, metadata)
            status_batch = np.asarray(status_batch)
            value_batch = np.asarray(value_batch)

        # Warn the user if nothing was inserted into the archives.
        if archive_empty_before and self.archive.empty:
            warnings.warn(Scheduler.EMPTY_WARNING.format(name="archive"))
        if self._result_archive is not None:
            if result_archive_empty_before and self.result_archive.empty:
                warnings.warn(
                    Scheduler.EMPTY_WARNING.format(name="result_archive"))

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for i in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[i]
            n = self._num_emitted[i]

            end = pos + n
            self._selection[i] += n
            self._success[i] += np.count_nonzero(status_batch[pos:end])
            emitter.tell(self._solution_batch[pos:end],
                         objective_batch[pos:end], measures_batch[pos:end],
                         status_batch[pos:end], value_batch[pos:end],
                         metadata_batch[pos:end])
            pos = end
