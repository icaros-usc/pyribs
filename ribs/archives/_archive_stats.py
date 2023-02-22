"""Provides ArchiveStats."""
import dataclasses

import numpy as np


@dataclasses.dataclass
class ArchiveStats:
    """Holds statistics about an archive.

    Attributes of type :class:`~numpy.floating` will match the
    :attr:`~ArchiveBase.dtype` of their archive.
    """

    #: Number of elites in the archive.
    num_elites: int

    #: Proportion of cells in the archive that have an elite - always in the
    #: range :math:`[0,1]`.
    coverage: np.floating

    #: QD score, i.e. sum of objective values of all elites in the archive. If
    #: ``qd_score_offset`` was passed in to the archive, this QD score
    #: normalizes the objectives by subtracting the offset from all objective
    #: values before computing the QD score.
    qd_score: np.floating

    #: Normalized QD score, i.e. the QD score divided by the number of cells in
    #: the archive.
    norm_qd_score: np.floating

    #: Maximum objective value of the elites in the archive. None if there are
    #: no elites in the archive.
    #:
    #: .. note::
    #:     If the archive is non-elitist (this occurs when using the archive
    #:     with a learning rate which is not 1.0, as in CMA-MAE), then an elite
    #:     with this objective may no longer exist in the archive because it was
    #:     replaced with an elite with a lower objective value. This can happen
    #:     because in non-elitist archives, new solutions only need to exceed
    #:     the *threshold* of the cell they are being inserted into, not the
    #:     *objective* of the elite currently in the cell. See `#314
    #:     <https://github.com/icaros-usc/pyribs/pull/314>`_ for more info.
    obj_max: np.floating

    #: Mean objective value of the elites in the archive. None if there are no
    #: elites in the archive.
    obj_mean: np.floating
