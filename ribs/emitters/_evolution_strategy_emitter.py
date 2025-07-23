"""Provides the EvolutionStrategyEmitter."""

import numbers

import numpy as np

from ribs._utils import check_shape, validate_batch
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import _get_es
from ribs.emitters.rankers import _get_ranker


class EvolutionStrategyEmitter(EmitterBase):
    """Adapts a distribution of solutions with an ES.

    This emitter originates in `Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_. The
    multivariate Gaussian solution distribution begins at ``x0`` with standard deviation
    ``sigma0``. Based on how the generated solutions are ranked (see ``ranker``), the ES
    then adapts the mean and covariance of the distribution.

    Args:
        archive (ribs.archives.ArchiveBase): Archive of solutions, e.g.,
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution. Must be 1-dimensional.
        sigma0 (float): Initial step size / standard deviation of the distribution from
            which solutions are sampled.
        ranker (Callable or str): The ranker is a
            :class:`~ribs.emitters.rankers.RankerBase` object that orders the solutions
            after they have been evaluated in the environment. This parameter may be a
            callable (e.g. a class or a lambda function) that takes in no parameters and
            returns an instance of :class:`~ribs.emitters.rankers.RankerBase`, or it may
            be a full or abbreviated ranker name as described in
            :mod:`ribs.emitters.rankers`.
        es (Callable or str): The evolution strategy is an
            :class:`~ribs.emitters.opt.EvolutionStrategyBase` object that is used to
            adapt the distribution from which new solutions are sampled. This parameter
            may be a callable (e.g. a class or a lambda function) that takes in the
            parameters of :class:`~ribs.emitters.opt.EvolutionStrategyBase` along with
            kwargs provided by the ``es_kwargs`` argument, or it may be a full or
            abbreviated optimizer name as described in :mod:`ribs.emitters.opt`.
        es_kwargs (dict): Additional arguments to pass to the evolution strategy
            optimizer. See the evolution-strategy-based optimizers in
            :mod:`ribs.emitters.opt` for the arguments allowed by each optimizer.
        selection_rule ("mu" or "filter"): Method for selecting parents for the
            evolution strategy. With "mu" selection, the first half of the solutions
            will be selected as parents, while in "filter", any solutions that were
            added to the archive will be selected.
        restart_rule (int, "no_improvement", and "basic"): Method to use when checking
            for restarts. If given an integer, then the emitter will restart after this
            many iterations, where each iteration is a call to :meth:`tell`. With
            "basic", only the default CMA-ES convergence rules will be used, while with
            "no_improvement", the emitter will restart when none of the proposed
            solutions were added to the archive.
        bounds (None or array-like): Bounds of the solution space. As suggested in
            `Biedrzycki 2020
            <https://www.sciencedirect.com/science/article/abs/pii/S2210650219301622>`_,
            solutions are resampled until they fall within these bounds. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to specify
            the bounds for each dim. Each element in this array-like can be None to
            indicate no bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not passed
            in, a batch size will be automatically calculated using the default CMA-ES
            rules.
        seed (int): Value to seed the random number generator. Set to None to avoid a
            fixed seed.
    Raises:
        ValueError: There is an error in x0 or initial_solutions.
        ValueError: There is an error in the bounds configuration.
        ValueError: If ``restart_rule``, ``selection_rule``, or ``ranker`` is invalid.
    """

    def __init__(
        self,
        archive,
        *,
        x0,
        sigma0,
        ranker="2imp",
        es="cma_es",
        es_kwargs=None,
        selection_rule="filter",
        restart_rule="no_improvement",
        bounds=None,
        batch_size=None,
        seed=None,
    ):
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        seed_sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        opt_seed, ranker_seed = seed_sequence.spawn(2)

        self._x0 = np.array(x0, dtype=archive.dtypes["solution"])
        check_shape(self._x0, "x0", archive.solution_dim, "archive.solution_dim")
        self._sigma0 = sigma0

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0

        # Check if the restart_rule is valid, discard check_restart result.
        _ = self._check_restart(0)

        self._opt = _get_es(
            es,
            sigma0=sigma0,
            batch_size=batch_size,
            solution_dim=self._solution_dim,
            seed=opt_seed,
            dtype=self.archive.dtypes["solution"],
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            **(es_kwargs if es_kwargs is not None else {}),
        )
        self._opt.reset(self._x0)

        self._ranker = _get_ranker(ranker, ranker_seed)
        self._ranker.reset(self, archive)

        self._batch_size = self._opt.batch_size

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def restarts(self):
        """int: The number of restarts for this emitter."""
        return self._restarts

    @property
    def itrs(self):
        """int: The number of iterations for this emitter, where each iteration is a
        call to :meth:`tell`."""
        return self._itrs

    def ask(self):
        """Samples new solutions from a multivariate Gaussian.

        The multivariate Gaussian is parameterized by the evolution strategy optimizer
        ``self._opt``.

        Returns:
            (batch_size, :attr:`solution_dim`) array -- a batch of new solutions to
            evaluate.
        """
        return self._opt.ask()

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.

        Raises:
            ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, numbers.Integral):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell(self, solution, objective, measures, add_info, **fields):
        """Gives the emitter results from evaluating solutions.

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`. Then, the ranked solutions are passed to CMA-ES for adaptation.

        This function also checks for restart condition and restarts CMA-ES when needed.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of solutions
                generated by this emitter's :meth:`ask()` method.
            objective (array-like): 1D array containing the objective function value of
                each solution.
            measures (array-like): (batch_size, measure space dimension) array with the
                measure space coordinates of each solution.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each argument
                should be an array with batch_size as the first dimension.
        """
        data, add_info = validate_batch(
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
        )

        # Increase iteration counter.
        self._itrs += 1

        # Count number of new solutions.
        new_sols = add_info["status"].astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(self, self.archive, data, add_info)

        # Select the number of parents.
        num_parents = (
            new_sols if self._selection_rule == "filter" else self._batch_size // 2
        )

        # Update Evolution Strategy.
        self._opt.tell(indices, ranking_values, num_parents)

        # Check for reset.
        if self._opt.check_stop(ranking_values[indices]) or self._check_restart(
            new_sols
        ):
            new_x0 = self.archive.sample_elites(1)["solution"][0]
            self._opt.reset(new_x0)
            self._ranker.reset(self, self.archive)
            self._restarts += 1
