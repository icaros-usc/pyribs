"""Provides the Bandit Scheduler."""
import numpy as np


class BanditScheduler:
    """A basic class that composes an archive with multiple emitters.

    To use this class, first create an archive and list of emitters for the
    QD algorithm. Then, construct the Scheduler with these arguments. Finally,
    repeatedly call :meth:`ask` to collect solutions to analyze, and return the
    objective and measures of those solutions **in the same order** using
    :meth:`tell`.

    As all solutions go into the same archive, the  emitters passed in must emit
    solutions with the same dimension (that is, their ``solution_dim`` attribute
    must be the same).

    .. warning:: If you are constructing many emitters at once, do not do
        something like ``[EmitterClass(...)] * 5``, as this creates a list with
        the same instance of ``EmitterClass`` in each position. Instead, use
        ``[EmitterClass(...) for _ in range 5]``, which creates 5 unique
        instances of ``EmitterClass``.

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, e.g. one
            selected from :mod:`ribs.archives`.
        emitters (list of ribs.archives.EmitterBase): A list of emitter objects,
            e.g. :class:`ribs.emitters.GaussianEmitter`.
        add_mode (str): Indicates how solutions should be added to the archive.
            The default is "batch", which adds all solutions with one call to
            :meth:`~ribs.archives.ArchiveBase.add`. Alternatively, use "single"
            to add the solutions one at a time with
            :meth:`~ribs.archives.ArchiveBase.add_single`. "single" mode is
            included for legacy reasons, as it was the only mode of operation in
            pyribs 0.4.0 and before. We highly recommend using "batch" mode
            since it is significantly faster.
        result_archive (ribs.archives.ArchiveBase): In some algorithms, such as
            CMA-MAE, the archive does not store all the best-performing
            solutions. The `result_archive` is a secondary archive where we can
            store all the best-performing solutions.
    Raises:
        ValueError: The emitters passed in do not have the same solution
            dimensions.
        ValueError: There is no emitter passed in.
        ValueError: The same emitter instance was passed in multiple times. Each
            emitter should be a unique instance (see the warning above).
        ValueError: Invalid value for `add_mode`.
    """

    def __init__(self,
                 archive,
                 emitters,
                 num_active,
                 *,
                 zeta=0.05,
                 result_archive=None,
                 add_mode="batch"):
        if len(emitters) == 0:
            raise ValueError("Pass in at least one emitter to the scheduler.")

        emitter_ids = set(id(e) for e in emitters)
        if len(emitter_ids) != len(emitters):
            raise ValueError(
                "Not all emitters passed in were unique (i.e. some emitters "
                "had the same id). If emitters were created with something "
                "like [EmitterClass(...)] * n, instead use "
                "[EmitterClass(...) for _ in range(n)] so that all emitters "
                "are unique instances.")

        self._solution_dim = emitters[0].solution_dim

        for idx, emitter in enumerate(emitters[1:]):
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
        self._emitters = np.array(emitters)
        self._num_active = num_active
        self._add_mode = add_mode

        # Used for UCB-1.
        n = len(self._emitters)
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
        # Note: Right now we replace all emitters every round.
        ucb1 = (
            self._success / self._selection +
            self._zeta * np.sqrt(np.log(self._success.sum()) / self._selection))
        self._active_arr = np.argsort(ucb1)[-self._num_active:]

        self._solution_batch = []

        for i, emitter in zip(self._active_arr,
                              self._emitters[self._active_arr]):
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

        .. note:: The objective batch, measures batch, and metadata batch must
            be in the same order as the solutions created by :meth:`ask_dqd`;
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
        for i, emitter, n in zip(self._active_arr,
                                 self._emitters[self._active_arr],
                                 self._num_emitted[self._active_arr]):
            end = pos + n
            emitter.tell(self._solution_batch[pos:end],
                         objective_batch[pos:end], measures_batch[pos:end],
                         status_batch[pos:end], value_batch[pos:end],
                         metadata_batch[pos:end])
            # Track selection and success.
            self._selection[i] += n
            self._success[i] += status_batch[pos:end].astype(bool).sum()
            pos = end
