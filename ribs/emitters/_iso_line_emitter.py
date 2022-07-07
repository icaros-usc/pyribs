"""Provides the IsoLineEmitter."""
import itertools

import numpy as np
from numba import jit

from ribs.emitters._emitter_base import EmitterBase


class IsoLineEmitter(EmitterBase):
    """Emits solutions that are nudged towards other archive solutions.

    If the archive is empty, calls to :meth:`ask` will generate solutions from
    an isotropic Gaussian distribution with mean ``x0`` and standard deviation
    ``sigma0``. Otherwise, to generate each new solution, the emitter selects
    a pair of elites :math:`x_i` and :math:`x_j` and samples from

    .. math::

        x_i + \\sigma_{iso} \\mathcal{N}(0,\\mathcal{I}) +
            \\sigma_{line}(x_j - x_i)\\mathcal{N}(0,1)

    This emitter is based on the Iso+LineDD operator presented in `Vassiliades
    2018 <https://arxiv.org/abs/1804.03906>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        iso_sigma (float): Scale factor for the isotropic distribution used to
            generate solutions when the archive is non-empty.
        line_sigma (float): Scale factor for the line distribution used when
            generating solutions.
        sigma0 (float): Standard deviation of the isotropic distribution used to
            generate solutions when the archive is empty. If this argument is
            None, then ``iso_sigma`` will be used.
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
                 x0,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 sigma0=None,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._iso_sigma = archive.dtype(iso_sigma)
        self._sigma0 = self._iso_sigma if sigma0 is None else archive.dtype(
            sigma0)
        self._line_sigma = archive.dtype(line_sigma)

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
    def sigma0(self):
        """float: Scale factor for the isotropic distribution used to
        generate solutions when the archive is empty."""
        return self._sigma0

    @property
    def line_sigma(self):
        """float: Scale factor for the line distribution used when generating
        solutions."""
        return self._line_sigma

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @staticmethod
    @jit(nopython=True)
    def _ask_solutions_numba(parents, iso_gaussian, line_gaussian, directions):
        """Numba helper for calculating solutions."""
        return parents + iso_gaussian + line_gaussian * directions

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(solutions, lower_bounds, upper_bounds):
        """Numba version of clip."""
        return np.minimum(np.maximum(solutions, lower_bounds), upper_bounds)

    def ask(self):
        """Generates ``batch_size`` solutions.

        If the archive is empty, solutions are drawn from an isotropic Gaussian
        distribution centered at ``self.x0`` with standard deviation
        ``self.sigma0``. Otherwise, each solution is drawn as described in
        this class's docstring with standard deviation ``self.iso_sigma``.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        iso_gaussian = self._rng.normal(
            scale=self._sigma0 if self.archive.empty else self._iso_sigma,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        if self.archive.empty:
            solutions = np.expand_dims(self._x0, axis=0) + iso_gaussian
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

            solutions = self._ask_solutions_numba(np.asarray(parents),
                                                  iso_gaussian, line_gaussian,
                                                  np.asarray(directions))

        return self._ask_clip_helper(solutions, self.lower_bounds,
                                     self.upper_bounds)

    def tell(self,
             solutions,
             objective_values,
             behavior_values,
             status_batch,
             value_batch,
             metadata=None):
        """Inserts entries into the archive.

        This base class implementation (in :class:`~ribs.emitters.EmitterBase`)
        simply inserts entries into the archive by calling
        :meth:`~ribs.archives.ArchiveBase.add`. It is enough for simple emitters
        like :class:`~ribs.emitters.GaussianEmitter`, but more complex emitters
        will almost certainly need to override it.

        Args:
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            status_batch (numpy.ndarray): 1D array of
                :class:`ribs.archive.AddStatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1D array of floats returned by a series
                of calls to archive's :meth:`add()` method. For what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
        """
        metadata = itertools.repeat(None) if metadata is None else metadata
        for sol, obj, beh, meta in zip(solutions, objective_values,
                                       behavior_values, metadata):
            self.archive.add(sol, obj, beh, meta)
