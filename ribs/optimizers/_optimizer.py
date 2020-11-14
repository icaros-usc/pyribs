"""Provides the Optimizer and corresponding OptimizerConfig."""
import numpy as np

from ribs.config import create_config


class OptimizerConfig:
    """Configuration for the Optimizer.

    Args:
        (none yet)
    """

    def __init__(self):
        pass


class Optimizer:
    """A basic class that composes an archive with multiple emitters.

    To use this class, first create an archive and list of emitters for your
    QD algorithm. Then, construct the Optimizer with these arguments. Finally,
    repeatedly call :meth:`ask` to collect solutions to analyze, and return the
    objective values and behavior values of those solutions **in the same
    order** using :meth:`tell`.

    As all solutions go into the same archive, the  emitters you pass in must
    emit solutions with the same dimension (that is, their ``solution_dim``
    attribute must be the same).

    Args:
        archive (ribs.archives.ArchiveBase): An archive object, selected from
            :mod:`ribs.archives`.
        emitters (list of ribs.archives.EmitterBase): A list of emitter objects,
            such as :class:`ribs.emitters.GaussianEmitter`.
        config (None or dict or OptimizerConfig): Configuration object. If None,
            a default OptimizerConfig is constructed. A dict may also be passed
            in, in which case its arguments will be passed into OptimizerConfig.
    Attributes:
        config (OptimizerConfig): Configuration object.
        archive (ribs.archives.ArchiveBase): See args.
        emitters (list of ribs emitters): See args.
    Raises:
        RuntimeError: The emitters passed in do not have the same solution
            dimensions.
        RuntimeError: There is no emitter passed in.
    """

    def __init__(self, archive, emitters, config=None):
        self.config = create_config(config, OptimizerConfig)

        if len(emitters) == 0:
            raise RuntimeError(
                "You must pass in at least one emitter to the optimizer.")
        self._solution_dim = emitters[0].solution_dim

        for idx, emitter in enumerate(emitters[1:]):
            if emitter.solution_dim != self._solution_dim:
                raise RuntimeError(
                    "All emitters must have the same solution dim, but "
                    f"Emitter {idx} has dimension {emitter.solution_dim}, "
                    f"while Emitter 0 has dimension {self._solution_dim}")

        self.archive = archive
        self.archive.initialize(self._solution_dim)
        self.emitters = emitters

        # Keeps track of whether the Optimizer should be receiving a call to
        # ask() or tell().
        self._asked = False
        # The last set of solutions returned by ask().
        self._solutions = []

    def ask(self):
        """Generates a batch of solutions by calling ask() on all emitters.

        .. note:: The order of the solutions returned from this method is
            important, so do not rearrange them.

        Returns:
            (n_solutions, dim) array: An array of n solutions to evaluate. Each
            row contains a single solution.
        Raises:
            RuntimeError: You attempt to call this method again without first
                calling :meth:`tell`.
        """
        if self._asked:
            raise RuntimeError("You have called ask() twice in a row.")
        self._asked = True

        self._solutions = []
        for emitter in self.emitters:
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
            RuntimeError: You attempt to call this method without first calling
                :meth:`ask`.
        """
        if not self._asked:
            raise RuntimeError("You have called tell() without ask().")
        self._asked = False

        objective_values = np.array(objective_values)
        behavior_values = np.array(behavior_values)

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter in self.emitters:
            end = pos + emitter.batch_size
            emitter.tell(self._solutions[pos:end], objective_values[pos:end],
                         behavior_values[pos:end])
            pos = end
