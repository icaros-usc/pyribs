"""Provides the Scheduler."""

import numpy as np
from ribs.schedulers._scheduler import Scheduler
from ribs.emitters._bayesian_opt_emitter import BayesianOptimizationEmitter


class BayesianOptimizationScheduler(Scheduler):
    def __init__(
        self, archive, emitters, *, result_archive=None, add_mode="batch"
    ):
        super().__init__(
            archive, emitters, result_archive=result_archive, add_mode=add_mode
        )

        # Checks that all emitters are BayesianOptimizationEmitter and have the
        # same upscale schedule
        this_upscale_schedule = None
        for i, e in enumerate(emitters):
            if not isinstance(e, BayesianOptimizationEmitter):
                raise TypeError(
                    "All emitters must be of type BayesianOptimizationEmitter, "
                    f"but emitter{i} has type {type(e)}"
                )

            if i == 0:
                this_upscale_schedule = e.upscale_schedule
            else:
                other_upscale_schedule = e.upscale_schedule
                if np.any(this_upscale_schedule != other_upscale_schedule):
                    raise ValueError(
                        "All emitters must have the same upscale schedule. "
                        "emitter0 has upscale schedule "
                        f"{this_upscale_schedule}, but emitter{i} has upscale "
                        f"schedule {other_upscale_schedule}."
                    )

        if this_upscale_schedule is None:
            self._upscale_schedule = None
        else:
            self._upscale_schedule = this_upscale_schedule.copy()

    @property
    def upscale_schedule(self):
        return self._upscale_schedule

    @Scheduler.archive.setter
    def archive(self, new_archive):
        self._archive = new_archive

    def tell_dqd(self, objective, measures, jacobian, **fields):
        raise NotImplementedError(
            "BayesianOptimization currently does not support DQD"
        )

    def tell(self, objective, measures, **fields):
        """Updates :attr:`emitters` and the :attr:`archive` with new data. When
        **ALL** emitters are ready to upscale, calls
        :meth:`~ribs.archives.ArchiveBase.retessellate` to upscale the archive.
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
        for i, (emitter, n) in enumerate(
            zip(self._emitters, self._num_emitted)
        ):
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
            if not self.upscale_schedule is None:
                if i == 0:
                    this_upscale_res = upscale_res
                else:
                    if np.any(this_upscale_res != upscale_res):
                        raise ValueError(
                            "Emitters returned different upscale resolutions when "
                            "they should return the same. Emitter0 returned "
                            f"resolution {this_upscale_res}, but emitter{i} "
                            f"returned resolution {upscale_res}"
                        )

        # If the upscale resolution is not None, upscales :attr:`archive` and
        # all emitter archives.
        if not this_upscale_res is None:
            new_archive = self.archive.retessellate(this_upscale_res)
            self.archive = new_archive
            for e in self._emitters:
                e.archive = new_archive
                e._post_upscale_updates()
