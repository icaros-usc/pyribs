"""ES that wraps pycma."""

import numpy as np
from threadpoolctl import threadpool_limits

from ribs._utils import readonly
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase


class PyCMAEvolutionStrategy(EvolutionStrategyBase):
    """Wrapper around the pycma :class:`~cma.evolution_strategy.CMAEvolutionStrategy`.

    .. note::

        This optimizer requires the `pycma <https://github.com/CMA-ES/pycma>`_ package,
        which can be installed with ``pip install cma`` or ``conda install cma``.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int or str): Number of solutions to evaluate at a time. This is
            passed directly as ``popsize`` in ``opts``.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        lower_bounds (float or np.ndarray): scalar or (solution_dim,) array indicating
            lower bounds of the solution space. Scalars specify the same bound for the
            entire space, while arrays specify a bound for each dimension. Pass -np.inf
            in the array or scalar to indicated unbounded space.
        upper_bounds (float or np.ndarray): Same as above, but for upper bounds (and
            pass np.inf instead of -np.inf).
        opts (dict): Additional options for pycma. Note that ``popsize``, ``bounds``,
            ``randn``, and ``seed`` are overwritten by us and thus should not be
            provided in this dict. We also make ``verbose`` default to -9, but you can
            also pass in a custom value here.
    """

    def __init__(  # pylint: disable = super-init-not-called
        self,
        sigma0,
        solution_dim,
        batch_size=None,
        seed=None,
        dtype=np.float64,
        lower_bounds=None,
        upper_bounds=None,
        opts=None,
    ):
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype

        self._solutions = None

        self._es = None
        self._opts = opts or {}
        self._opts["popsize"] = batch_size
        self._opts["bounds"] = [lower_bounds, upper_bounds]

        # Default CMAEvolutionStrategy uses global seed. To avoid this, we must
        # provide a randn function tied to our rng -- see:
        # https://github.com/CMA-ES/pycma/issues/221
        self._rng = np.random.default_rng(seed)
        self._opts["randn"] = lambda batch_size, n: self._rng.standard_normal(
            (batch_size, n)
        )
        self._opts["seed"] = np.nan

        self._opts.setdefault("verbose", -9)

    @property
    def batch_size(self):
        """int: Number of solutions per iteration.

        Only valid after a call to :meth:`reset`.
        """
        return self._es.popsize

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        try:
            # We do not want to import at the top because that would require cma
            # to always be installed, as cma would be imported whenever this
            # class is imported.
            # pylint: disable = import-outside-toplevel
            import cma
        except ImportError as e:
            raise ImportError(
                "pycma must be installed -- please run `pip install cma` or "
                "`conda install cma`"
            ) from e

        self._es = cma.CMAEvolutionStrategy(x0, self.sigma0, self._opts)

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Args:
            ranking_values (np.ndarray): Not used.

        Returns:
            Output of cma.CMAEvolutionStrategy.stop
        """
        # Convert (batch_size, 1) array into (batch_size,).
        if ranking_values.ndim == 2 and ranking_values.shape[1] == 1:
            ranking_values = ranking_values[:, 0]

        # If ranking_values are scalar, we use pycma's termination criterion
        # directly.
        if ranking_values.ndim == 1:
            return self._es.stop()

        # Otherwise, we check our own criteria like in CMAEvolutionStrategy,
        # since the objectives passed into tell() were fictitious and thus we
        # need to check ranking_values on our own.

        # Tolerances from pycma CMA-ES.
        if self._es.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self._es.sigma * np.sqrt(np.max(self._es.sm.eigenspectrum))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (
            len(ranking_values) >= 2
            and np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12
        ):
            return True

        return False

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Args:
            batch_size (int): batch size of the sample. Defaults to ``self.batch_size``.
        """
        # batch_size defaults to popsize in CMA-ES.
        self._solutions = np.asarray(self._es.ask(batch_size))
        return readonly(self._solutions.astype(self.dtype))

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def tell(self, ranking_indices, ranking_values, num_parents):
        # Convert (batch_size, 1) array into (batch_size,).
        if ranking_values.ndim == 2 and ranking_values.shape[1] == 1:
            ranking_values = ranking_values[:, 0]

        if ranking_values.ndim == 1:
            # Directly tell values to ES since these are just 1D. The
            # ranking_values are presented with higher being better, but CMA-ES
            # minimizes, so we need to invert all values.
            self._es.tell(self._solutions, -ranking_values)
        else:
            # This case is when ranking_values is 2D. In this case, we cannot
            # tell the values to the ES because the ES expects scalars. Thus, we
            # make up our own scalars, which is fine because CMA-ES only looks
            # at the ranks of the solutions during search. However, it does look
            # at values in the termination criterion, hence we use custom
            # conditions in check_stop.
            self._es.tell(
                self._solutions[ranking_indices], np.arange(len(ranking_indices))
            )
