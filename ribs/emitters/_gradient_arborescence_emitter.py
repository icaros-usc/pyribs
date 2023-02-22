"""Provides the GradientArborescenceEmitter."""
import numpy as np

from ribs._utils import check_1d_shape, validate_batch_args
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import _get_es, _get_grad_opt
from ribs.emitters.rankers import _get_ranker


class GradientArborescenceEmitter(EmitterBase):
    """Generates solutions with a gradient arborescence, with coefficients
    parameterized by an ES.

    This emitter originates in `Fontaine 2021
    <https://arxiv.org/abs/2106.03894>`_. It leverages the gradient information
    of the objective and measure functions, generating new solutions around a
    *solution point* :math:`\\boldsymbol{\\theta}` using *gradient
    arborescence*, with coefficients drawn from a Gaussian distribution.
    Essentially, this means that the emitter samples coefficients
    :math:`\\boldsymbol{c_i} \\sim
    \\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})`
    and creates new solutions :math:`\\boldsymbol{\\theta'_i}` according to

    .. math::

        \\boldsymbol{\\theta'_i} \\gets \\boldsymbol{\\theta} +
            c_{i,0} \\boldsymbol{\\nabla} f(\\boldsymbol{\\theta}) +
            \\sum_{j=1}^k c_{i,j}\\boldsymbol{\\nabla}m_j(\\boldsymbol{\\theta})

    Where :math:`k` is the number of measures, and
    :math:`\\boldsymbol{\\nabla} f(\\boldsymbol{\\theta})` and
    :math:`\\boldsymbol{\\nabla} m_j(\\boldsymbol{\\theta})` are the objective
    and measure gradients of the solution point :math:`\\boldsymbol{\\theta}`,
    respectively.

    Based on how the solutions are ranked after being inserted into the archive
    (see ``ranker``), the solution point :math:`\\boldsymbol{\\theta}` is
    updated with gradient ascent, and the coefficient distribution parameters
    :math:`\\boldsymbol{\\mu}` and :math:`\\boldsymbol{\\Sigma}` are updated
    with an ES (the default ES is CMA-ES).

    .. note::

        Unlike non-gradient emitters, GradientArborescenceEmitter requires
        calling :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell` to communicate the gradient
        information to the emitter.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        sigma0 (float): Initial step size / standard deviation.
        lr (float): Learning rate for the gradient optimizer.
        ranker (Callable or str): The ranker is a :class:`RankerBase` object
            that orders the solutions after they have been evaluated in the
            environment. This parameter may be a callable (e.g. a class or a
            lambda function) that takes in no parameters and returns an instance
            of :class:`RankerBase`, or it may be a full or abbreviated ranker
            name as described in :mod:`ribs.emitters.rankers`.
        selection_rule ("mu" or "filter"): Method for selecting parents in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected as parents, while in "filter", any solutions that were
            added to the archive will be selected.
        restart_rule (int, "no_improvement", and "basic"): Method to use when
            checking for restarts. If given an integer, then the emitter will
            restart after this many iterations, where each iteration is a call
            to :meth:`tell`. With "basic", only the default CMA-ES convergence
            rules will be used, while with "no_improvement", the emitter will
            restart when none of the proposed solutions were added to the
            archive.
        grad_opt (Callable or str): Gradient optimizer to use for the gradient
            ascent step of the algorithm. The optimizer is a
            :class:`GradientOptBase` object. This parameter may be a callable
            (e.g. a class or a lambda function) which takes in the ``theta0``
            and ``lr`` arguments, or it may be a full or abbreviated name as
            described in :mod:`ribs.emitters.opt`.
        grad_opt_kwargs (dict): Additional arguments to pass to the gradient
            optimizer. See the gradient-based optimizers in
            :mod:`ribs.emitters.opt` for the arguments allowed by each
            optimizer. Note that we already pass in ``theta0`` and ``lr``.
        es (Callable or str): The evolution strategy is an
            :class:`EvolutionStrategyBase` object that is used to adapt the
            distribution from which new gradient coefficients are sampled. This
            parameter may be a callable (e.g. a class or a lambda function) that
            takes in the parameters of :class:`EvolutionStrategyBase` along with
            kwargs provided by the ``es_kwargs`` argument, or it may be a full
            or abbreviated optimizer name as described in
            :mod:`ribs.emitters.opt`.
        es_kwargs (dict): Additional arguments to pass to the evolution
            strategy optimizer. See the evolution-strategy-based optimizers in
            :mod:`ribs.emitters.opt` for the arguments allowed by each
            optimizer.
        normalize_grad (bool): If true (default), then gradient infomation will
            be normalized. Otherwise, it will not be normalized.
        bounds: This argument may be used for providing solution space bounds in
            the future. This emitter does not currently support solution space
            bounds, as bounding solutions for DQD algorithms such as CMA-MEGA is
            an open problem. Hence, this argument must be set to None.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will be automatically calculated using the
            default CMA-ES rules. Note that `batch_size` **does not** include
            the number of solutions returned by :meth:`ask_dqd`, but also note
            that :meth:`ask_dqd` always returns one solution, i.e. the solution
            point.
        epsilon (float): For numerical stability, we add a small epsilon when
            normalizing gradients in :meth:`tell_dqd` -- refer to the
            implementation `here
            <../_modules/ribs/emitters/_gradient_arborescence_emitter.html#GradientArborescenceEmitter.tell_dqd>`_.
            Pass this parameter to configure that epsilon.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in x0 or initial_solutions.
        ValueError: ``bounds`` is set even though it is not currently supported.
        ValueError: If ``restart_rule``, ``selection_rule``, or ``ranker`` is
            invalid.
    """

    def __init__(self,
                 archive,
                 *,
                 x0,
                 sigma0,
                 lr,
                 ranker="2imp",
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 grad_opt="adam",
                 grad_opt_kwargs=None,
                 es="cma_es",
                 es_kwargs=None,
                 normalize_grad=True,
                 bounds=None,
                 batch_size=None,
                 epsilon=1e-8,
                 seed=None):

        if bounds is not None:
            raise ValueError(
                "`bounds` must be set to None. The GradientArborescenceEmitter "
                "does not currently support solution space bounds, as bounding "
                "solutions for DQD algorithms such as CMA-MEGA is an open "
                "problem.")

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        check_1d_shape(self._x0, "x0", archive.solution_dim,
                       "archive.solution_dim")
        self._sigma0 = sigma0
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None

        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0

        # Check if the restart_rule is valid, discard check_restart result.
        _ = self._check_restart(0)

        # We have a coefficient for each measure and an extra coefficient for
        # the objective.
        self._num_coefficients = archive.measure_dim + 1

        # Initialize gradient optimizer.
        self._grad_opt = _get_grad_opt(
            grad_opt,
            theta0=self._x0,
            lr=lr,
            **(grad_opt_kwargs if grad_opt_kwargs is not None else {}))

        opt_seed = None if seed is None else self._rng.integers(10_000)
        self._opt = _get_es(es,
                            sigma0=sigma0,
                            batch_size=batch_size,
                            solution_dim=self._num_coefficients,
                            seed=opt_seed,
                            dtype=self.archive.dtype,
                            **(es_kwargs if es_kwargs is not None else {}))

        self._opt.reset(np.zeros(self._num_coefficients))

        self._batch_size = self._opt.batch_size
        self._itrs = 0

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

    @property
    def itrs(self):
        """int: The number of iterations for this emitter."""
        return self._itrs

    @property
    def epsilon(self):
        """int: The epsilon added for numerical stability when normalizing
        gradients in :meth:`tell_dqd`."""
        return self._epsilon

    def ask_dqd(self):
        """Samples a new solution from the gradient optimizer.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            a new solution to evaluate.
        """
        return self._grad_opt.theta[None]

    def ask(self):
        """Samples new solutions from a gradient arborescence parameterized by a
        multivariate Gaussian distribution.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self._opt``.

        Note that this method returns `batch_size - 1` solution as one solution
        is returned via ask_dqd.

        Returns:
            (batch_size, :attr:`solution_dim`) array -- a batch of new solutions
            to evaluate.
        """
        coeff_lower_bounds = np.full(
            self._num_coefficients,
            -np.inf,
            dtype=self._archive.dtype,
        )
        coeff_upper_bounds = np.full(
            self._num_coefficients,
            np.inf,
            dtype=self._archive.dtype,
        )
        grad_coeffs = self._opt.ask(
            coeff_lower_bounds,
            coeff_upper_bounds,
        )[:, :, None]
        return (self._grad_opt.theta +
                np.sum(self._jacobian_batch * grad_coeffs, axis=1))

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.
        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

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
            solution_batch (array-like): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's
                :meth:`ask_dqd()` method.
            objective_batch (array-like): 1d array containing the objective
                function value of each solution.
            measures_batch (array-like): (batch_size, measure space dimension)
                array with the measure space coordinates of each solution.
            jacobian_batch (array-like): (batch_size, 1 + measure_dim,
                solution_dim) array consisting of Jacobian matrices of the
                solutions obtained from :meth:`ask_dqd`. Each matrix should
                consist of the objective gradient of the solution followed by
                the measure gradients.
            status_batch (array-like): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls
                to archive's :meth:`add()` method.
            value_batch (array-like): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (array-like): 1d object array containing a metadata
                object for each solution.
        """
        # Preprocessing arguments.
        solution_batch = np.asarray(solution_batch)
        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        status_batch = np.asarray(status_batch)
        value_batch = np.asarray(value_batch)
        batch_size = solution_batch.shape[0]
        metadata_batch = (np.empty(batch_size, dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        # Validate arguments.
        validate_batch_args(archive=self.archive,
                            solution_batch=solution_batch,
                            objective_batch=objective_batch,
                            measures_batch=measures_batch,
                            status_batch=status_batch,
                            value_batch=value_batch,
                            jacobian_batch=jacobian_batch,
                            metadata_batch=metadata_batch)

        if self._normalize_grads:
            norms = (np.linalg.norm(jacobian_batch, axis=2, keepdims=True) +
                     self._epsilon)
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
            solution_batch (array-like): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (array-like): 1d array containing the objective
                function value of each solution.
            measures_batch (array-like): (batch_size, measure space dimension)
                array with the measure space coordinates of each solution.
            status_batch (array-like): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls
                to archive's :meth:`add()` method.
            value_batch (array-like): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (array-like): 1d object array containing a metadata
                object for each solution.
        """
        # Preprocessing arguments.
        solution_batch = np.asarray(solution_batch)
        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        status_batch = np.asarray(status_batch)
        value_batch = np.asarray(value_batch)
        batch_size = solution_batch.shape[0]
        metadata_batch = (np.empty(batch_size, dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        # Validate arguments.
        validate_batch_args(archive=self.archive,
                            solution_batch=solution_batch,
                            objective_batch=objective_batch,
                            measures_batch=measures_batch,
                            status_batch=status_batch,
                            value_batch=value_batch,
                            metadata_batch=metadata_batch)

        if self._jacobian_batch is None:
            raise RuntimeError("tell() was called without calling tell_dqd().")

        # Increase iteration counter.
        self._itrs += 1

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
        self._opt.tell(indices, num_parents)

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
        if (self._opt.check_stop(ranking_values[indices]) or
                self._check_restart(new_sols)):
            new_coeff = self.archive.sample_elites(1).solution_batch[0]
            self._grad_opt.reset(new_coeff)
            self._opt.reset(np.zeros(self._num_coefficients))
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1
