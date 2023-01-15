"""Provides the Bandit Scheduler."""
import numpy as np


class BanditScheduler:
    """Schedules emitters with a bandit algorithm.

    This implementation is based on `Cully 2021
    <https://arxiv.org/pdf/2007.05352.pdf>`_.

    .. note::
        This class follows the similar ask-tell framework as
        :class:`Scheduler`, and enforces similar constraints in the arguments
        and methods. Please refer to the documentation of :class:`Scheduler`
        for more details.

    To initialize this class, first create an archive and a list of emitters
    for the QD algorithm. The BanditScheduler will schedule the emitters using
    the Upper Confidence Bound - 1 algorithm (UCB1). Everytime :meth:`ask` is
    called, the emitters are sorted based on the potential reward function from
    UCB1. Then, the top `num_active` emitters are used for ask-tell.

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, e.g. one
            selected from :mod:`ribs.archives`.
        emitters (list of ribs.archives.EmitterBase): A pool of emitter objects
            to be selected from, e.g. :class:`ribs.emitters.GaussianEmitter`.
        num_active (int): The number of emitters used for ask-tell.
        zeta (float): Hyperparamter of UBC1 that balances the trade-off between
            the accuracy and the uncertainty of the emitters. Increasing this
            parameter will emphasize the uncertainty of the emitters. Refer to
            the original paper for more information.
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

        # Used for UCB-1.
        n = len(self._emitter_pool)
        self._active_arr = np.empty(n)  # Index of active emitters.
        self._success = np.zeros(n)
        self._selection = np.zeros(n)
        self._zeta = zeta

        self._result_archive = result_archive

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

    def ask(self):
        """Generates a batch of solutions by calling ask() on all emitters.

        The emitters used by ask are determined by the UCB1 algorithm. Briefly,
        emitters that have never been selected are prioritized, then emitters
        are sorted in descending order based the accurary of their past
        prediction.

        .. note:: The order of the solutions returned from this method is
            important, so do not rearrange them.

        Returns:
            (batch_size, dim) array: An array of n solutions to evaluate. Each
            row contains a single solution.
        Raises:
            RuntimeError: This method was called without first calling
                :meth:`tell`.
        """
        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError("ask_dqd cannot be called immediately after " +
                               self._last_called)
        self._last_called = "ask"

        # Select emitters based on UCB-1.
        # NOTE: Right now we replace all emitters every round. This is
        # different from the original paper, which only replaces emitters that
        # have terminated.
        ucb1 = np.empty_like(self._emitter_pool)
        update_ucb = self._selection != 0
        if update_ucb.any():
            ucb1[update_ucb] = (
                self._success[update_ucb] / self._selection[update_ucb] +
                self._zeta * np.sqrt(
                    np.log(self._success.sum()) / self._selection[update_ucb])
            )
        self._active_arr = np.argsort(ucb1)[-self._num_active:]

        self._solution_batch = []

        for i in self._active_arr:
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
                ``metadata`` has the wrong shape.
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

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for i in self._active_arr:
            emitter = self._emitter_pool[i]
            n = self._num_emitted[i]

            end = pos + n
            self._selection[i] += n
            self._success[i] += status_batch[pos:end].astype(bool).sum()
            emitter.tell(self._solution_batch[pos:end],
                         objective_batch[pos:end], measures_batch[pos:end],
                         status_batch[pos:end], value_batch[pos:end],
                         metadata_batch[pos:end])
            pos = end
