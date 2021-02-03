"""Provides the Optimizer."""
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

        self._solutions = []

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            for emitter in self._emitters:
                self._solutions.append(emitter.ask())

        self._solutions = np.concatenate(self._solutions, axis=0)
        return self._solutions

    def tell(self, objective_values, behavior_values):
        """Returns objective and behavior values for solutions from :meth:`ask`.

        .. note:: The objective values and behavior values must be in the same
            order as the solutions created by :meth:`ask`; i.e.
            ``objective_values[i]`` and ``behavior_values[i]`` should be the
            objective value and behavior values for ``solutions[i]``.

        Args:
            objective_values ((n_solutions,) array): Each entry of this array
                contains the objective function evaluation of a solution.
            behavior_values ((n_solutions, behavior_dm) array): Each row of
                this array contains a solution's coordinates in behavior space.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
        """
        if not self._asked:
            raise RuntimeError("tell() was called without calling ask().")
        self._asked = False

        objective_values = np.asarray(objective_values)
        behavior_values = np.asarray(behavior_values)

        # Limit OpenBLAS to single thread. This is typically faster than
        # multithreading because our data is too small.
        with threadpool_limits(limits=1, user_api="blas"):
            # Keep track of pos because emitters may have different batch sizes.
            pos = 0
            for emitter in self._emitters:
                end = pos + emitter.batch_size
                emitter.tell(self._solutions[pos:end],
                             objective_values[pos:end],
                             behavior_values[pos:end])
                pos = end
