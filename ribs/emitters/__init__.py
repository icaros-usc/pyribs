"""Emitters generate new candidate solutions in QD algorithms.

More formally, emitters are algorithms that generate solutions and adapt to
objective, measure, and archive insertion feedback.

The emitters in this module follow a one-layer hierarchy, with all emitters
inheriting from :class:`~ribs.emitters.EmitterBase`.

.. note::
    Emitters provided here take on the data type of the archive passed to their
    constructor. For instance, if an archive has dtype ``np.float64``, then an
    emitter created with that archive will emit solutions with dtype
    ``np.float64``.

.. autosummary::
    :toctree:

    ribs.emitters.EvolutionStrategyEmitter
    ribs.emitters.GradientArborescenceEmitter
    ribs.emitters.GaussianEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.EmitterBase
"""
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters._evolution_strategy_emitter import EvolutionStrategyEmitter
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._gradient_arborescence_emitter import \
    GradientArborescenceEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter

__all__ = [
    "EvolutionStrategyEmitter",
    "GradientArborescenceEmitter",
    "GaussianEmitter",
    "IsoLineEmitter",
    "EmitterBase",
]
