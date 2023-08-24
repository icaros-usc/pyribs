"""Provides the GradientEmitter."""

import numpy as np

from ribs._utils import check_1d_shape, check_batch_shape, validate_batch_args
from ribs.emitters._emitter_base import EmitterBase


class GradientEmitter(EmitterBase):
    """Generates new solutions based on the gradient of the objective and measures.

    _extended_summary_

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        sigma_g (float): A step-size for the gradient in the gradient step. If measure
            gradients are used, sigma_g is the standard deviation of Gaussian noise
            used to sample gradient coefficients.
        line_sigma (float): The theta_2 parameter for a Iso+LineDD operator.
        measure_gradients (bool): Signals if measure gradients will be used.
        normalize_gradients (bool): Sets if gradients should be normalized before steps.
        operator_type (str): Either 'isotropic' or 'iso_line_dd' to mark the operator type 
            for intermediate operations. Defaults to 'isotropic'.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        epsilon (float): For numerical stability, we add a small epsilon when
            normalizing gradients in :meth:`tell_dqd` -- refer to the
            implementation `here
            <../_modules/ribs/emitters/_gradient_arborescence_emitter.html#GradientArborescenceEmitter.tell_dqd>`_.
            Pass this parameter to configure that epsilon.
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
        """Samples a new solution to have its value and gradient evaluated.
        """
        if self.archive.empty and self._initial_solutions is not None:
            return np.empty((0, self.archive.solution_dim))

        # get perturbed solutions from the archive
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

            sol = parents + line_gaussian * directions
            sol = np.minimum(np.maximum(sol + noise, self.lower_bounds),
                             self.upper_bounds)
        else:
            noise = self._rng.normal(
                loc=0.0,
                scale=self.sigma0,
                size=(self.batch_size, self.solution_dim),
            ).astype(self.archive.dtype)

            sol = np.minimum(np.maximum(parents + noise, self.lower_bounds),
                             self.upper_bounds)

        self._parents = sol
        return self._parents

    def tell_dqd(self,
                 solution_batch,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 status_batch,
                 value_batch,
                 metadata_batch=None):
        """Sets the emitter Jacbians from evaluating the gradient of the
        solutions.
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

    def ask(self):
        """Get branched solutions

        _extended_summary_
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
