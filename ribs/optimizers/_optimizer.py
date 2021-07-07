"""Provides the Optimizer."""
import itertools

import numpy as np
from threadpoolctl import threadpool_limits


class Optimizer:
    """A basic class that composes an archive with multiple emitters.

    To use this class, first create an archive and list of emitters for the
    QD algorithm. Then, construct the Optimizer with these arguments. Finally,
    repeatedly call :meth:`ask` to collect solutions to analyze, and return the
    objective values and behavior values of those solutions **in the same
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
        archive (ribs.archives.ArchiveBase): An archive object, selected from
            :mod:`ribs.archives`.
        emitters (list of ribs.archives.EmitterBase): A list of emitter objects,
            such as :class:`ribs.emitters.GaussianEmitter`.
    Raises:
        ValueError: The emitters passed in do not have the same solution
            dimensions.
        ValueError: There is no emitter passed in.
        ValueError: The same emitter instance was passed in multiple times. Each
            emitter should be a unique instance (see the warning above).
    """

    def __init__(self, archive, emitters):
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

        self._archive = archive
        self._archive.initialize(self._solution_dim)
        self._emitters = emitters

        # Keeps track of whether the Optimizer should be receiving a call to
        # ask() or tell().
        self._asked = False
        # The last set of solutions returned by ask().
        self._solutions = []
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

    @staticmethod
    def _process_emitter_kwargs(emitter_kwargs):
        """Converts emitter_kwargs to an iterable so it can zip with the
        emitters."""
        if emitter_kwargs is None:
            emitter_kwargs = itertools.repeat({})
        if isinstance(emitter_kwargs, dict):
            emitter_kwargs = itertools.repeat(emitter_kwargs)
        return emitter_kwargs  # Assume it is a list/iterable of dicts.

    def ask(self, emitter_kwargs=None):
        """Generates a batch of solutions by calling ask() on all emitters.

        .. note:: The order of the solutions returned from this method is
            important, so do not rearrange them.

        Args:
            emitter_kwargs (dict or list of dict): kwargs to pass to the
                emitters' :meth:`~ribs.emitters.EmitterBase.ask` method. If one
                dict is passed in, its kwargs are passed to all the emitters. If
                a list of dicts is passed in, each dict is passed to each
                emitter (e.g.  ``dict[0]`` goes to :attr:`emitters` [0]).
                Emitters are in the same order as when the optimizer was
                constructed.
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

        self._solutions = []
        emitter_kwargs = self._process_emitter_kwargs(emitter_kwargs)

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            for i, (emitter,
                    kwargs) in enumerate(zip(self._emitters, emitter_kwargs)):
                emitter_sols = emitter.ask(**kwargs)
                self._solutions.append(emitter_sols)
                self._num_emitted[i] = len(emitter_sols)

        self._solutions = np.concatenate(self._solutions, axis=0)
        return self._solutions

    def tell(self,
             objective_values,
             behavior_values,
             metadata=None,
             emitter_kwargs=None):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective values, behavior values, and metadata must be in
            the same order as the solutions created by :meth:`ask`; i.e.
            ``objective_values[i]``, ``behavior_values[i]``, and ``metadata[i]``
            should be the objective value, behavior values, and metadata for
            ``solutions[i]``.

        Args:
            objective_values ((n_solutions,) array): Each entry of this array
                contains the objective function evaluation of a solution.
            behavior_values ((n_solutions, behavior_dm) array): Each row of
                this array contains a solution's coordinates in behavior space.
            metadata ((n_solutions,) array): Each entry of this array contains
                an object holding metadata for a solution.
            emitter_kwargs (dict or list of dict): kwargs to pass to the
                emitters' :meth:`~ribs.emitters.EmitterBase.tell` method. If one
                dict is passed in, its kwargs are passed to all the emitters. If
                a list of dicts is passed in, each dict is passed to each
                emitter (e.g.  ``dict[0]`` goes to :attr:`emitters` [0]).
                Emitters are in the same order as when the optimizer was
                constructed.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
        """
        if not self._asked:
            raise RuntimeError("tell() was called without calling ask().")
        self._asked = False

        emitter_kwargs = self._process_emitter_kwargs(emitter_kwargs)
        objective_values = np.asarray(objective_values)
        behavior_values = np.asarray(behavior_values)
        metadata = (np.empty(len(self._solutions), dtype=object)
                    if metadata is None else np.asarray(metadata, dtype=object))

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            # Keep track of pos because emitters may have different batch sizes.
            pos = 0
            for emitter, n, kwargs in zip(self._emitters, self._num_emitted,
                                          emitter_kwargs):
                end = pos + n
                emitter.tell(
                    self._solutions[pos:end],
                    objective_values[pos:end],
                    behavior_values[pos:end],
                    metadata[pos:end],
                    **kwargs,
                )
                pos = end
