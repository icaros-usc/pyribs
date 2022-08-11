"""Provides the GradientImprovementEmitter."""
import itertools

import numpy as np

from ribs.emitters._dqd_emitter_base import DQDEmitterBase
from ribs.emitters.opt import AdamOpt, CMAEvolutionStrategy, GradientAscentOpt
from ribs.emitters.rankers import _get_ranker


class GradientAborescenceEmitter(DQDEmitterBase):
    """Generates solutions with a gradient arborescence, with coefficients
    parameterized by CMA-ES.

    This emitter originates in `Fontaine 2021
    <https://arxiv.org/abs/2106.03894>`_. It leverages the gradient information
    of the objective and measure functions, generating new solutions around a
    "solution point" using gradient aborescence with coefficients drawn from a
    Gaussian distribution. Based on how the solutions are ranked after being
    inserted into the archive (see ``ranker``), the solution point is updated
    with gradient ascent, and the distribution is updated with CMA-ES.

    Note that unlike non-gradient emitters, GradientAborescenceEmitter requires
    calling :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before calling
    :meth:`ask` and :meth:`tell` to communicate the gradient information to the
    emitter.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        sigma0 (float): Initial step size / standard deviation.
        step_size (float): Step size for the gradient optimizer
        ranker (Callable or str): The ranker is a :class:`RankerBase` object
            that orders the solutions after they have been evaluated in the
            environment. This parameter may be a callable (e.g. a class or a
            lambda function) that takes in no parameters and returns an instance
            of :class:`RankerBase`, or it may be a full or abbreviated ranker
            name as described in :meth:`ribs.emitters.rankers.get_ranker`.
        selection_rule ("mu" or "filter"): Method for selecting parents in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected as parents, while in "filter", any solutions that were
            added to the archive will be selected.
        restart_rule ("no_improvement" or "basic"): Method to use when checking
            for restarts. With "basic", only the default CMA-ES convergence
            rules will be used, while with "no_improvement", the emitter will
            restart when none of the proposed solutions were added to the
            archive.
        grad_opt ("adam" or "gradient_ascent"): Gradient optimizer to use for
            the gradient ascent step of the algorithm. Defaults to `adam`.
        normalize_grad (bool): If true (default), then gradient infomation will
            be normalized. Otherwise, it will not be normalized.
        bounds (None or array-like): Bounds of the solution space. As suggested
            in `Biedrzycki 2020
            <https://www.sciencedirect.com/science/article/abs/pii/S2210650219301622>`_,
            solutions are resampled until they fall within these bounds.  Pass
            None to indicate there are no bounds. Alternatively, pass an
            array-like to specify the bounds for each dim. Each element in this
            array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will be automatically calculated using the
            default CMA-ES rules. Note that `batch_size` **does not** include
            the number of solutions returned by :meth:`ask_dqd`, but also note
            that :meth:`ask_dqd` always returns one solution, i.e. the solution
            point.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: If ``restart_rule`` is invalid.
    """
    # Used to ensure numerical stability when normalizing the gradient
    _epsilon = 1e-8

    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 step_size,
                 ranker="2imp",
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 grad_opt="adam",
                 normalize_grad=True,
                 bounds=None,
                 batch_size=None,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None
        self._grad_coefficients = None
        DQDEmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)

        # Initialize gradient optimizer
        self._grad_opt = None
        if grad_opt == "adam":
            self._grad_opt = AdamOpt(self._x0, step_size)
        elif grad_opt == "gradient_ascent":
            self._grad_opt = GradientAscentOpt(self._x0, step_size)
        else:
            raise ValueError(f"Invalid Gradient Ascent Optimizer {grad_opt}")

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        # We have a coefficient for each measure and an extra coefficient for
        # the objective.
        self._num_coefficients = archive.behavior_dim + 1

        opt_seed = None if seed is None else self._rng.integers(10_000)
        self.opt = CMAEvolutionStrategy(sigma0, batch_size,
                                        self._num_coefficients, "truncation",
                                        opt_seed, self.archive.dtype)
        self.opt.reset(np.zeros(self._num_coefficients))

        self._batch_size = self.opt.batch_size
        self._restarts = 0

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def batch_size_dqd(self):
        """int: Number of solutions to return in :meth:`ask_dqd`.

        This is always 1, as we only return the solution point in
        :meth:`ask_dqd`.
        """
        return 1

    @property
    def restarts(self):
        """int: The number of restarts for this emitter."""
        return self._restarts

    def ask_dqd(self):
        """Samples a new solution from the gradient optimizer.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            a new solution to evaluate.
        """
        return [self._grad_opt.theta]

    def ask(self):
        """Samples new solutions from a gradient aborescence parameterized by a
        multivariate Gaussian distribution.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self.opt``.

        Note that this method returns `batch_size - 1` solution as one solution
        is returned via ask_dqd.

        Returns:
            (batch_size, :attr:`solution_dim`) array -- a batch of new solutions
            to evaluate.
        """
        lower_bounds = np.full(self._num_coefficients,
                               -np.inf,
                               dtype=self._archive.dtype)
        upper_bounds = np.full(self._num_coefficients,
                               np.inf,
                               dtype=self._archive.dtype)
        self._grad_coefficients = self.opt.ask(lower_bounds, upper_bounds)
        noise = np.expand_dims(self._grad_coefficients, axis=2)

        return self._grad_opt.theta + np.sum(
            np.multiply(self._jacobian_batch, noise), axis=1)

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.
        """
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        return False

    def tell_dqd(self,
                 solution_batch,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 status_batch,
                 value_batch,
                 metadata_batch=None):
        """Gives the emitter results from evaluating the gradient of the
        solutions.

        Args:
            solution_batch (numpy.ndarray): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (numpy.ndarray): 1d array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): (batch_size, measure space
                dimension) array with the measure space coordinates of each
                solution.
            jacobian_batch (numpy.ndarray): ``(batch_size, 1 + measure_dim,
                solution_dim)`` array consisting of Jacobian matrices of the
                solutions obtained from :meth:`ask_dqd`. Each matrix should
                consist of the objective gradient of the solution followed by
                the measure gradients.
            status_batch (numpy.ndarray): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1d object array containing a
                metadata object for each solution.
        """
        if self._normalize_grads:
            norms = np.linalg.norm(jacobian_batch, axis=2) + self._epsilon
            norms = np.expand_dims(norms, axis=2)
            jacobian_batch /= norms
        self._jacobian_batch = jacobian_batch

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        """Gives the emitter results from evaluating solutions.

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`.

        Args:
            solution_batch (numpy.ndarray): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (numpy.ndarray): 1d array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): (batch_size, measure space
                dimension) array with the measure space coordinates of each
                solution.
            status_batch (numpy.ndarray): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1d object array containing a
                metadata object for each solution.
        """
        if self._jacobian_batch is None:
            raise RuntimeError("tell() was called without calling tell_dqd().")

        metadata_batch = itertools.repeat(
            None) if metadata_batch is None else metadata_batch

        # Count number of new solutions.
        new_sols = status_batch.astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(
            self, self.archive, self._rng, solution_batch, objective_batch,
            measures_batch, status_batch, value_batch, metadata_batch)

        # Select the number of parents.
        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._batch_size // 2)

        # Update Evolution Strategy.
        self.opt.tell(self._grad_coefficients[indices], num_parents)

        # Calculate a new mean in solution space. These weights are from CMA-ES.
        parents = solution_batch[indices]
        parents = parents[:num_parents]
        weights = (np.log(num_parents + 0.5) -
                   np.log(np.arange(1, num_parents + 1)))
        weights = weights / np.sum(weights)  # Normalize weights
        new_mean = np.sum(parents * np.expand_dims(weights, axis=1), axis=0)

        # Use the mean to calculate a gradient step and step the optimizer.
        gradient_step = new_mean - self._grad_opt.theta
        self._grad_opt.step(gradient_step)

        # Check for reset.
        if (self.opt.check_stop(ranking_values[indices]) or
                self._check_restart(new_sols)):
            new_coeff = self.archive.sample_elites(1).solution_batch[0]
            self._grad_opt.reset(new_coeff)
            self.opt.reset(np.zeros(self._num_coefficients))
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1
