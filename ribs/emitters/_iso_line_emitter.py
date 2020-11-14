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
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid seeding.
        batch_size (int): Number of solutions to send back in the ask() method.
    Attributes:
        x0 (np.ndarray): See args.
        iso_sigma (float): See args.
        line_sigma (float): See args.
        solution_dim (int): The (1D) dimension of solutions produced by this
            emitter.
        batch_size (int): Number of solutions to generate on each call to ask().
    """

    def __init__(self,
                 x0,
                 archive,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 batch_size=64,
                 seed=None):
        self.x0 = np.array(x0)
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma

        EmitterBase.__init__(self, len(self.x0), batch_size, archive, seed)

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
        if self._archive.is_empty():
            return np.expand_dims(self.x0, axis=0) + self._rng.normal(
                scale=self.iso_sigma, size=(self.batch_size, self.solution_dim))

        parents = [
            self._archive.get_random_elite()[0] for _ in range(self.batch_size)
        ]
        directions = [(self._archive.get_random_elite()[0] - parents[i])
                      for i in range(self.batch_size)]
        iso_gaussian = self._rng.normal(scale=self.iso_sigma,
                                        size=(self.batch_size,
                                              self.solution_dim))
        line_gaussian = self._rng.normal(scale=self.line_sigma,
                                         size=(self.batch_size, 1))
        return parents + iso_gaussian + line_gaussian * directions
