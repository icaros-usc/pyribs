from ribs.emitters.selectors._selector_base import SelectorBase


class FilterSelector(SelectorBase):

    def select(self, emitter, archive, solutions, objective_values,
               behavior_values, metadata, add_statuses, add_values):
        """Selects the number of parents that will be used for the evolution strategy

        Selects all the added or improved solutions.

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
            add_statuses (list of ribs.archives.AddStatus):
            add_values ():

        Returns:
            num_parents: the number of top parents to use in the evolution strategy
        """
        new_sols = 0
        for status in add_statuses:
            if bool(status):
                new_sols += 1
        return new_sols
