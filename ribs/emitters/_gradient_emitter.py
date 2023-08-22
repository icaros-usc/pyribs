"""Provides the GradientEmitter."""

import numpy as np

from ribs._utils import check_1d_shape, validate_batch_args
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import _get_es, _get_grad_opt
from ribs.emitters.rankers import _get_ranker


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
                 x0,
                 sigma0=0.1,
                 sigma_g=0.05,
                 line_sigma=0.0,
                 measure_gradients=False,
                 normalize_gradients=False,
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
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = archive.dtype(sigma0) if isinstance(
            sigma0, (float, np.floating)) else np.array(sigma0)
        self._sigma_g = archive.dtype(sigma_g)
        self._line_sigma = line_sigma
        self._use_isolinedd = operator_type != 'isotropic'
        self._measure_gradients = measure_gradients
        self._normalize_gradients = normalize_gradients
        self._epsilon = epsilon
        self._batch_size = batch_size

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
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

    def ask_dqd(self,):
        """Samples a new solution to have its value and gradient evaluated.
        """
        # get solutions not from grad optimizer, but from....
        if self.archive.empty():
            parents = np.expand_dims(self.x0, axis=0)

    def tell_dqd(self,):
        """Sets the emitter Jacbians from evaluating the gradient of the
        solutions.
        """
        # preprocess + validate args

        # normalize gradients + set jacobian
        # jacobian is obtained from evaluating solutions of ask_dqd()

    def ask(self):
        """Get branched solutions

        _extended_summary_
        """

    def tell(self):
        """update optimizer internals using ranking and Jacobian info

        """
