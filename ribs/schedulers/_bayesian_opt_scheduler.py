"""Provides the BayesianOptimizationScheduler."""

import numpy as np

from ribs.archives._grid_archive import GridArchive
from ribs.emitters._bayesian_opt_emitter import BayesianOptimizationEmitter
from ribs.schedulers._scheduler import Scheduler


class BayesianOptimizationScheduler(Scheduler):
    """Similar to :class:`~Scheduler` but with support for upscaling archives.

    This scheduler should only be used in conjunction with
    :class:`~ribs.emitters.BayesianOptimizationEmitter` and
    :class:`~ribs.archives.GridArchive`.

    Args:
        archive (ribs.archives.GridArchive): An archive object.
        emitters (list of ribs.emitters.BayesianOptimizationEmitter): A list of emitter
            objects.
        result_archive (ribs.archives.ArchiveBase): In some algorithms, such as CMA-MAE,
            the archive does not store all the best-performing solutions. The
            ``result_archive`` is a secondary archive where we can store all the
            best-performing solutions.
        add_mode (str): Indicates how solutions should be added to the archive. The
            default is "batch", which adds all solutions with one call to
            :meth:`~ribs.archives.ArchiveBase.add`. Alternatively, use "single" to add
            the solutions one at a time with
            :meth:`~ribs.archives.ArchiveBase.add_single`. "single" mode is included
            for legacy reasons, as it was the only mode of operation in pyribs 0.4.0 and
            before. We highly recommend using "batch" mode since it is significantly
            faster.

    Raises:
        TypeError: Some emitters are not BayesianOptimizationEmitter.
        ValueError: Not all emitters have the same upscale schedule.
    """

    def __init__(self, archive, emitters, result_archive=None, *, add_mode="batch"):
        super().__init__(archive, emitters, result_archive, add_mode=add_mode)

        # Checks that all emitters are BayesianOptimizationEmitter and have the same
        # upscale schedule.
        this_upscale_schedule = None
        for i, e in enumerate(emitters):
            if not isinstance(e, BayesianOptimizationEmitter):
                raise TypeError(
                    "All emitters must be of type BayesianOptimizationEmitter, "
                    f"but emitter{i} has type {e.__class__.__name__}"
                )

            if i == 0:
                this_upscale_schedule = e.upscale_schedule
            else:
                other_upscale_schedule = e.upscale_schedule
                # pylint: disable=unidiomatic-typecheck
                if (
                    type(this_upscale_schedule) != type(other_upscale_schedule)
                    or this_upscale_schedule.shape != other_upscale_schedule.shape
                    or np.any(this_upscale_schedule != other_upscale_schedule)
                ):
                    raise ValueError(
                        "All emitters must have the same upscale schedule. "
                        "emitter0 has upscale schedule "
                        f"{this_upscale_schedule}, but emitter{i} has upscale "
                        f"schedule {other_upscale_schedule}."
                    )

        # Checks that ``archive`` is a GridArchive
        if not isinstance(archive, GridArchive):
            raise TypeError(
                "Archive type must be GridArchive. Actually got "
                f"{archive.__class__.__name__}"
            )

        if this_upscale_schedule is None:
            self._upscale_schedule = None
        else:
            self._upscale_schedule = this_upscale_schedule.copy()

    @property
    def upscale_schedule(self):
        """The upscale schedules for all the Bayesian optimization emitters.  None if
        emitters do not undergo archive upscaling."""
        return self._upscale_schedule

    @Scheduler.archive.setter
    def archive(self, new_archive):
        self._archive = new_archive

    def ask_dqd(self):
        raise NotImplementedError("BayesianOptimization currently does not support DQD")

    def tell_dqd(self, objective, measures, jacobian, **fields):
        raise NotImplementedError("BayesianOptimization currently does not support DQD")

    def tell(self, objective, measures, **fields):
        """Updates :attr:`emitters` and the :attr:`archive` with new data.

        When **ALL** emitters are ready to upscale, calls
        :meth:`~ribs.archives.GridArchive.retessellate` to upscale the archive.
        Otherwise same as :meth:`~Scheduler.tell`.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        data = self._validate_tell_data(
            {
                "objective": objective,
                "measures": measures,
                **fields,
            }
        )

        add_info = self._add_to_archives(data)

        pos = 0
        this_upscale_res = None
        for i, (emitter, n) in enumerate(zip(self._emitters, self._num_emitted)):
            end = pos + n
            upscale_res = emitter.tell(
                **{
                    name: None if arr is None else arr[pos:end]
                    for name, arr in data.items()
                },
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end

            # Check that all emitters have returned the same upscale resolution.
            if self.upscale_schedule is not None:
                if i == 0:
                    this_upscale_res = upscale_res
                else:
                    if np.any(this_upscale_res != upscale_res):
                        raise ValueError(
                            "Emitters returned different upscale resolutions "
                            "when they should return the same. Emitter0 "
                            f"returned resolution {this_upscale_res}, but "
                            f"emitter{i} returned resolution {upscale_res}"
                        )

        # If the upscale resolution is not None, upscales :attr:`archive` and all
        # emitter archives.
        if this_upscale_res is not None:
            for e in self._emitters:
                e.archive.retessellate(this_upscale_res)
                e.post_upscale_updates()
