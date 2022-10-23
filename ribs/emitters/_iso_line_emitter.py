"""Provides the IsoLineEmitter."""
import numpy as np

from ribs._utils import check_batch_shape, check_is_1d
from ribs.emitters._emitter_base import EmitterBase


class IsoLineEmitter(EmitterBase):
    """Emits solutions that are nudged towards other archive solutions.

    If the archive is empty and ``self._initial_solutions`` is set, calls to
    :meth:`ask` will return ``self._initial_solutions``. If
    ``self._initial_solutions`` is not set, we draw solutions from an isotropic
    Gaussian distribution centered at ``self.x0`` with standard deviation
    ``self.iso_sigma``. Otherwise, each solution is drawn from a distribution
    centered at a randomly chosen elite with standard deviation
    ``self.iso_sigma``.

    .. math::

        x_i + \\sigma_{iso} \\mathcal{N}(0,\\mathcal{I}) +
            \\sigma_{line}(x_j - x_i)\\mathcal{N}(0,1)

    This emitter is based on the Iso+LineDD operator presented in `Vassiliades
    2018 <https://arxiv.org/abs/1804.03906>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        iso_sigma (float): Scale factor for the isotropic distribution used to
            generate solutions.
        line_sigma (float): Scale factor for the line distribution used when
            generating solutions.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty. Must be 1-dimensional.
            This argument is ignored if ``initial_solutions`` is set.
        initial_solutions (array-like): An (n, solution_dim) array of solutions
            to be used when the archive is empty. If this argument is None, then
            solutions will be sampled from a Gaussian distribution centered at
            ``x0`` with standard deviation ``iso_sigma``.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    """

    def __init__(self,
                 archive,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 x0=None,
                 initial_solutions=None,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size

        self._iso_sigma = archive.dtype(iso_sigma)
        self._line_sigma = archive.dtype(line_sigma)

        if x0 is None and initial_solutions is None:
            raise ValueError("At least one of x0 or initial_solutions must "
                             "be set.")

        self._x0 = np.array(x0, dtype=archive.dtype)
        check_is_1d(self._x0, "x0")

        self._initial_solutions = None
        if initial_solutions is not None:
            self._initial_solutions = np.asarray(initial_solutions,
                                                 dtype=archive.dtype)
            check_batch_shape(self._initial_solutions, "initial_solutions",
                              archive.solution_dim, "solution_dim")

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        return self._x0

    @property
    def iso_sigma(self):
        """float: Scale factor for the isotropic distribution used to
        generate solutions when the archive is not empty."""
        return self._iso_sigma

    @property
    def line_sigma(self):
        """float: Scale factor for the line distribution used when generating
        solutions."""
        return self._line_sigma

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Generates ``batch_size`` solutions.

        If the archive is empty and ``self._initial_solutions`` is set, we
        return ``self._initial_solutions``. If ``self._initial_solutions`` is
        not set, we draw solutions from an isotropic Gaussian distribution
        centered at ``self.x0`` with standard deviation ``self.iso_sigma``.
        Otherwise, each solution is drawn from a distribution centered at
        a randomly chosen elite with standard deviation ``self.iso_sigma``.

        Returns:
            If the archive is not empty, ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` new solutions to evaluate. If the
            archive is empty, we return ``self._initial_solutions``, which
            might not have ``batch_size`` solutions.
        """
        if self.archive.empty and self._initial_solutions is not None:
            return np.clip(self._initial_solutions, self.lower_bounds,
                           self.upper_bounds)

        iso_gaussian = self._rng.normal(
            scale=self._iso_sigma,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        if self.archive.empty:
            solution_batch = np.expand_dims(self._x0, axis=0) + iso_gaussian
        else:
            parents = self.archive.sample_elites(
                self._batch_size).solution_batch
            directions = (
                self.archive.sample_elites(self._batch_size).solution_batch -
                parents)
            line_gaussian = self._rng.normal(
                scale=self._line_sigma,
                size=(self._batch_size, 1),
            ).astype(self.archive.dtype)
            solution_batch = parents + iso_gaussian + line_gaussian * directions

        return np.clip(solution_batch, self.lower_bounds, self.upper_bounds)

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        """Gives the emitter results from evaluating solutions.

        Args:
            solution_batch (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_batch (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): ``(n, <measure space dimension>)``
                array with the measure space coordinates of each solution.
            status_batch (numpy.ndarray): An array of integer statuses
                returned by a series of calls to archive's :meth:`add_single()`
                method or by a single call to archive's :meth:`add()`.
            value_batch  (numpy.ndarray): 1D array of floats returned by a
                series of calls to archive's :meth:`add_single()` method or by a
                single call to archive's :meth:`add()`. For what these floats
                represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1D object array containing a
                metadata object for each solution.
        """
