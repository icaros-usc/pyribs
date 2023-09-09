"""Provides the GradientEmitter."""

import numpy as np

from ribs._utils import check_1d_shape, check_batch_shape, validate_batch_args
from ribs.emitters._emitter_base import EmitterBase


class GradientEmitter(EmitterBase):
    """Generates new solutions based on gradients of the objective and measures.

    _extended_summary_

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        initial_solutions (array-like): An (n, solution_dim) array of solutions
            to be used when the archive is empty. If this argument is None, then
            solutions will be sampled from a Gaussian distribution centered at
            ``x0`` with standard deviation ``sigma``.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        sigma_g (float): A step-size for the gradient in the gradient step.
            If measure gradients are used, sigma_g is the standard deviation of
            the Gaussian noise used to sample gradient coefficients.
        line_sigma (float): The theta_2 parameter for a Iso+LineDD operator.
        measure_gradients (bool): Signals if measure gradients will be used.
        normalize_grad (bool): Whether gradients should be normalized
            before steps.
        epsilon (float): For numerical stability, we add a small epsilon when
            normalizing gradients in :meth:`tell_dqd` -- refer to the
            implementation `here
            <../_modules/ribs/emitters/_gradient_arborescence_emitter.html#GradientArborescenceEmitter.tell_dqd>`_.
            Pass this parameter to configure that epsilon.
        operator_type (str): Either 'isotropic' or 'iso_line_dd' to mark the
            operator type for intermediate operations. Defaults to 'isotropic'.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.    """

    def __init__(self,
                 archive,
                 initial_solutions=None,
                 x0=None,
                 sigma0=0.1,
                 sigma_g=0.05,
                 line_sigma=0.0,
                 measure_gradients=False,
                 normalize_grad=False,
                 epsilon=1e-8,
                 operator_type='isotropic',
                 bounds=None,
                 batch_size=64,
                 seed=None):
        EmitterBase.__init__(
            self,
            archive=archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self._initial_solutions = None
        self._x0 = None

        if x0 is None and initial_solutions is None:
            raise ValueError("Either x0 or initial_solutions must be provided.")
        if x0 is not None and initial_solutions is not None:
            raise ValueError(
                "x0 and initial_solutions cannot both be provided.")

        if x0 is not None:
            self._x0 = np.array(x0, dtype=archive.dtype)
            check_1d_shape(self._x0, "x0", archive.solution_dim,
                           "archive.solution_dim")
        elif initial_solutions is not None:
            self._initial_solutions = np.asarray(initial_solutions,
                                                 dtype=archive.dtype)
            check_batch_shape(self._initial_solutions, "initial_solutions",
                              archive.solution_dim, "archive.solution_dim")

        self._rng = np.random.default_rng(seed)
        self._sigma0 = archive.dtype(sigma0) if isinstance(
            sigma0, (float, np.floating)) else np.array(sigma0)
        self._sigma_g = archive.dtype(sigma_g)
        self._line_sigma = line_sigma
        self._use_isolinedd = operator_type != 'isotropic'
        self._measure_gradients = measure_gradients
        self._normalize_grad = normalize_grad
        self._epsilon = epsilon
        self._batch_size = batch_size

        self._jacobian_batch = None
        self._parents = None

    @property
    def initial_solutions(self):
        """numpy.ndarray: The initial solutions which are returned when the
        archive is empty (if x0 is not set)."""
        return self._initial_solutions

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty (if initial_solutions is not
        set)."""
        return self._x0

    @property
    def sigma0(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution."""
        return self._sigma0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def batch_size_dqd(self):
        """int: Number of solutions to return in :meth:`ask_dqd`."""
        return self._batch_size

    @property
    def epsilon(self):
        """int: The epsilon added for numerical stability when normalizing
        gradients in :meth:`tell_dqd`."""
        return self._epsilon

    def ask_dqd(self):
        """Create new solutions by sampling elites from the archive with
        (optional) Gaussian perturbation.

        If the archive is empty and initial_solutions is given, the call to this
        method on the first iteration returns no solutions. Later iterations
        will sample elites from the archive.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        if self.archive.empty and self._initial_solutions is not None:
            return np.empty((0, self.archive.solution_dim))

        if self.archive.empty:
            parents = np.expand_dims(self.x0, axis=0)
        else:
            parents = self.archive.sample_elites(self.batch_size).solution_batch

        if self._use_isolinedd:
            noise = self._rng.normal(
                loc=0.0,
                scale=self.sigma0,
                size=(self.batch_size, self.solution_dim),
            ).astype(self.archive.dtype)

            directions = self.archive.sample_elites(
                self._batch_size).solution_batch - parents

            line_gaussian = self._rng.normal(
                scale=self._line_sigma,
                size=(self._batch_size, 1),
            ).astype(self.archive.dtype)

            sol = parents + line_gaussian * directions + noise
            sol = np.clip(sol, self.lower_bounds, self.upper_bounds)
        else:
            noise = self._rng.normal(
                loc=0.0,
                scale=self.sigma0,
                size=(self.batch_size, self.solution_dim),
            ).astype(self.archive.dtype)

            sol = parents + noise
            sol = np.clip(sol, self.lower_bounds, self.upper_bounds)

        self._parents = sol
        return self._parents

    def ask(self):
        """Samples new solutions from a gradient arborescence parameterized by a
        multivariate Gaussian distribution.

        The multivariate Gaussian is parameterized by sigma_g.

        This method returns ``batch_size`` solutions, even though one solution
        is returned via ``ask_dqd``.

        Returns:
            (:attr:`batch_size`, :attr:`solution_dim`) array -- a batch of new
            solutions to evaluate.
        Raises:
            RuntimeError: This method was called without first passing gradients
                with calls to ask_dqd() and tell_dqd().
        """
        if self.archive.empty and self._initial_solutions is not None:
            return self._initial_solutions

        if self._jacobian_batch is None:
            raise RuntimeError("Please call ask_dqd() and tell_dqd() "
                               "before calling ask().")

        if self._measure_gradients:
            noise = self._rng.normal(
                loc=0.0,
                scale=self._sigma_g,
                size=self._jacobian_batch.shape[:2],
            )
            noise[:, 0] = np.abs(noise[:, 0])
            noise = np.expand_dims(noise, axis=2)
            offsets = np.sum(np.multiply(self._jacobian_batch, noise), axis=1)
            sols = offsets + self._parents
        else:
            # Transform the Jacobian
            if len(self._jacobian_batch.shape) == 3:
                self._jacobian_batch = np.squeeze(self._jacobian_batch[:,
                                                                       0:1, :],
                                                  axis=1)
            sols = self._parents + self._jacobian_batch * self._sigma_g

        return sols

    def tell_dqd(self,
                 solution_batch,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 status_batch,
                 value_batch,
                 metadata_batch=None):
        """Gives the emitter results from evaluating the solutions.

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
        # preprocess + validate args
        solution_batch = np.asarray(solution_batch)
        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        status_batch = np.asarray(status_batch)
        value_batch = np.asarray(value_batch)
        batch_size = solution_batch.shape[0]
        metadata_batch = (np.empty(batch_size, dtype=object) if metadata_batch
                          is None else np.asarray(metadata_batch, dtype=object))

        # Validate arguments.
        validate_batch_args(archive=self.archive,
                            solution_batch=solution_batch,
                            objective_batch=objective_batch,
                            measures_batch=measures_batch,
                            status_batch=status_batch,
                            value_batch=value_batch,
                            jacobian_batch=jacobian_batch,
                            metadata_batch=metadata_batch)

        # normalize gradients + set jacobian
        # jacobian is obtained from evaluating solutions of ask_dqd()
        if self._normalize_grad:
            norms = np.linalg.norm(jacobian_batch, axis=2,
                                   keepdims=True) + self._epsilon
            jacobian_batch /= norms

        self._jacobian_batch = jacobian_batch
