"""Provides the IsoLineEmitter."""

import numpy as np

from ribs.emitters._emitter_base import EmitterBase


class IsoLineEmitter(EmitterBase):
    """Attempts to emit solutions from the same hypervolume as existing elites.

    If the archive is empty, calls to ask() will generate solutions from an
    isotropic Gaussian distribution with mean ``x0`` and standard deviation
    ``iso_sigma``. Otherwise, to generate each new solution, the emitter selects
    a pair of elites :math:`x_i` and :math:`x_j` and samples from

    .. math::

        x_i + \\sigma_{iso} \\mathcal{N}(0,\\mathcal{I}) +
            \\sigma_{line}(x_j - x_i)\\mathcal{N}(0,1)

    This emitter is based on the operator presented in this paper:
    https://arxiv.org/abs/1804.03906

    Args:
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        iso_sigma (float): Scale factor for the isotropic distribution used when
            generating solutions.
        line_sigma (float): Scale factor for the line distribution used when
            generating solutions.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.

            Pass an array-like to specify the bounds for each dim. Each element
            in this array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to send back in the ask() method.
        seed (int): Value to seed the random number generator. Set to None to
            avoid seeding.
    """

    def __init__(self,
                 x0,
                 archive,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._x0 = np.array(x0)
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma

        EmitterBase.__init__(
            self,
            len(self._x0),
            bounds,
            batch_size,
            archive,
            seed,
        )

    @property
    def x0(self):
        """np.ndarray: Center of the Gaussian distribution from which to sample
        solutions when the archive is empty."""
        return self._x0

    @property
    def iso_sigma(self):
        """float: Scale factor for the isotropic distribution used when
        generating solutions."""
        return self._iso_sigma

    @property
    def line_sigma(self):
        """float: Scale factor for the line distribution used when generating
        solutions."""
        return self._line_sigma

    def ask(self):
        """Generates ``self.batch_size`` solutions.

        If the archive is empty, solutions are drawn from an isotropic Gaussian
        distribution centered at ``self.x0`` with standard deviation
        ``self.iso_sigma``. Otherwise, each solution is drawn as described in
        this class's docstring.

        Returns:
            ``(self.batch_size, self.solution_dim)`` array -- contains
            ``batch_size`` new solutions to evaluate.
        """
        iso_gaussian = self._rng.normal(scale=self._iso_sigma,
                                        size=(self.batch_size,
                                              self.solution_dim))

        if self._archive.empty:
            solutions = np.expand_dims(self._x0, axis=0) + iso_gaussian
        else:
            parents = [
                self._archive.get_random_elite()[0]
                for _ in range(self.batch_size)
            ]
            directions = [(self._archive.get_random_elite()[0] - parents[i])
                          for i in range(self.batch_size)]
            line_gaussian = self._rng.normal(scale=self._line_sigma,
                                             size=(self.batch_size, 1))
            solutions = parents + iso_gaussian + line_gaussian * directions

        return np.clip(solutions, self.lower_bounds, self.upper_bounds)
