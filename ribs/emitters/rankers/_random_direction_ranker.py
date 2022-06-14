import numpy as np

from ribs.emitters.rankers._ranker_base import RankerBase


class RandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in
    behavior space.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    We rank the solutions solely based on their projection onto a random
    direction in behavior space.

    To rank the solutions first by whether they were added, and then by
    the projection, refer to
    :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        """Ranks the soutions based on projection onto a direction in behavior
        space.

        Args:
            emitter (ribs.emitters.EmitterBase):
            archive (ribs.archives.ArchiveBase): An archive to use when creating
                and inserting solutions. For instance, this can be
                :class:`ribs.archives.GridArchive`.
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
            add_statuses ():
            add_values ():

        Returns:
            indices: which represent the descending order of the solutions
        """
        ranking_data = []
        for i, (beh, status) in enumerate(zip(behavior_values, add_statuses)):
            projection = np.dot(beh, self._target_behavior_dir)
            added = bool(status)

            ranking_data.append((added, projection, i))
            if added:
                new_sols += 1

        # Sort only by projection.
        ranking_data.sort(reverse=True, key=lambda x: x[1])
        return [d[2] for d in ranking_data]

    def reset(self, archive, emitter):
        """Generates a new random direction in the behavior space.

        The direction is sampled from a standard Gaussian -- since the standard
        Gaussian is isotropic, there is equal probability for any direction. The
        direction is then scaled to the behavior space bounds.
        """

        ranges = archive.upper_bounds - archive.lower_bounds
        behavior_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(behavior_dim)
        self._target_behavior_dir = unscaled_dir * ranges
