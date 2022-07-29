"""Provides the Optimizer."""
import numpy as np
from threadpoolctl import threadpool_limits

from ribs.emitters import DQDEmitterBase


class Optimizer:
    """A basic class that composes an archive with multiple emitters.

    To use this class, first create an archive and list of emitters for the
    QD algorithm. Then, construct the Optimizer with these arguments. Finally,
    repeatedly call :meth:`ask` to collect solutions to analyze, and return the
    objective values and measures values of those solutions **in the same
    order** using :meth:`tell`.

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
    Raises:
        ValueError: The emitters passed in do not have the same solution
            dimensions.
        ValueError: There is no emitter passed in.
        ValueError: The same emitter instance was passed in multiple times. Each
            emitter should be a unique instance (see the warning above).
        ValueError: Invalid value for `add_mode`.
    """

    def __init__(self, archive, emitters, add_mode="batch"):
        if len(emitters) == 0:
            raise ValueError("Pass in at least one emitter to the optimizer.")

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

        self._archive = archive
        self._emitters = emitters
        self._add_mode = add_mode

        # Keeps track of whether the Optimizer should be receiving a call to
        # ask() or tell().
        self._asked = False
        self._asked_dqd = False
        # The last set of solutions returned by ask().
        self._solution_batch = []
        # The number of solutions created by each emitter.
        self._num_emitted = [None for _ in self._emitters]

    @property
    def archive(self):
        """ribs.archives.ArchiveBase: Archive for storing solutions found in
        this optimizer."""
        return self._archive

    @property
    def emitters(self):
        """list of ribs.archives.EmitterBase: Emitters for generating solutions
        in this optimizer."""
        return self._emitters

    def ask_dqd(self):
        if self._asked_dqd:
            raise RuntimeError("ask_dqd() was called twice in a row.")
        self._asked_dqd = True

        self._solution_batch = []

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            for i, emitter in enumerate(self._emitters):
                # check if emitter is dqd
                if isinstance(emitter, DQDEmitterBase):
                    emitter_sols = emitter.ask_dqd()
                    self._solution_batch.append(emitter_sols)
                    self._num_emitted[i] = len(emitter_sols)

        self._solution_batch = np.concatenate(self._solution_batch, axis=0)
        return self._solution_batch

    def ask(self):
        """Generates a batch of solutions by calling ask() on all emitters.

        .. note:: The order of the solutions returned from this method is
            important, so do not rearrange them.

        Returns:
            (n_solutions, dim) array: An array of n solutions to evaluate. Each
            row contains a single solution.
        Raises:
            RuntimeError: This method was called without first calling
                :meth:`tell`.
        """
        if self._asked:
            raise RuntimeError("ask() was called twice in a row.")
        self._asked = True

        self._solution_batch = []

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            for i, emitter in enumerate(self._emitters):
                emitter_sols = emitter.ask()
                self._solution_batch.append(emitter_sols)
                self._num_emitted[i] = len(emitter_sols)

        self._solution_batch = np.concatenate(self._solution_batch, axis=0)
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
                 jacobian_batch,
                 objective_batch,
                 measures_batch,
                 metadata_batch=None):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective batch, measures batch, and metadata must be in
            the same order as the solutions created by :meth:`ask`; i.e.
            ``objective_batch[i]``, ``measures_batch[i]``, and ``metadata[i]``
            should be the objective batch, measures batch, and metadata for
            ``solutions[i]``.

        Args:
            objective_batch ((n_solutions,) array): Each entry of this array
                contains the objective function evaluation of a solution.
            measures_batch ((n_solutions, measures_dm) array): Each row of
                this array contains a solution's coordinates in measure space.
            metadata ((n_solutions,) array): Each entry of this array contains
                an object holding metadata for a solution.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
            ValueError: ``objective_batch``, ``measures_batch``, or
                ``metadata`` has the wrong shape.
        """
        if not self._asked_dqd:
            raise RuntimeError("tell() was called without calling ask().")
        self._asked_dqd = False

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
        elif self._add_mode == "single":
            status_batch, value_batch = zip(*[
                self.archive.add_single(
                    solution,
                    objective,
                    measure,
                    metadata,
                ) for solution, objective, measure, metadata in zip(
                    self._solution_batch,
                    objective_batch,
                    measures_batch,
                    metadata_batch,
                )
            ])
            status_batch = np.asarray(status_batch)
            value_batch = np.asarray(value_batch)

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            # Keep track of pos because emitters may have different batch sizes.
            pos = 0
            for emitter, n in zip(self._emitters, self._num_emitted):
                end = pos + n
                emitter.tell_dqd(jacobian_batch[pos:end])
                pos = end

    def tell(self, objective_batch, measures_batch, metadata_batch=None):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective batch, measures batch, and metadata must be in
            the same order as the solutions created by :meth:`ask`; i.e.
            ``objective_batch[i]``, ``measures_batch[i]``, and ``metadata[i]``
            should be the objective batch, measures batch, and metadata for
            ``solutions[i]``.

        Args:
            objective_batch ((n_solutions,) array): Each entry of this array
                contains the objective function evaluation of a solution.
            measures_batch ((n_solutions, measures_dm) array): Each row of
                this array contains a solution's coordinates in measure space.
            metadata ((n_solutions,) array): Each entry of this array contains
                an object holding metadata for a solution.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
            ValueError: ``objective_batch``, ``measures_batch``, or
                ``metadata`` has the wrong shape.
        """
        if not self._asked:
            raise RuntimeError("tell() was called without calling ask().")
        self._asked = False

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
        elif self._add_mode == "single":
            status_batch, value_batch = zip(*[
                self.archive.add_single(
                    solution,
                    objective,
                    measure,
                    metadata,
                ) for solution, objective, measure, metadata in zip(
                    self._solution_batch,
                    objective_batch,
                    measures_batch,
                    metadata_batch,
                )
            ])
            status_batch = np.asarray(status_batch)
            value_batch = np.asarray(value_batch)

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            # Keep track of pos because emitters may have different batch sizes.
            pos = 0
            for emitter, n in zip(self._emitters, self._num_emitted):
                end = pos + n
                emitter.tell(self._solution_batch[pos:end],
                             objective_batch[pos:end], measures_batch[pos:end],
                             status_batch[pos:end], value_batch[pos:end],
                             metadata_batch[pos:end])
                pos = end
