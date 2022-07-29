"""Emitters output new candidate solutions in QD algorithms.

.. note::
    Emitters provided here take on the data type of the archive passed to their
    constructor. For instance, if an archive has dtype ``np.float64``, then an
    emitter created with that archive will emit solutions with dtype
    ``np.float64``.

.. autosummary::
    :toctree:

    ribs.emitters.GradientAborescenceEmitter
    ribs.emitters.EvolutionStrategyEmitter
    ribs.emitters.GaussianEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.DQDEmitterBase
    ribs.emitters.EmitterBase
"""
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters._dqd_emitter_base import DQDEmitterBase
from ribs.emitters._evolution_strategy_emitter import EvolutionStrategyEmitter
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._gradient_aborescence_emitter import \
  GradientAborescenceEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter

__all__ = [
    "GradientAborescenceEmitter",
    "EvolutionStrategyEmitter",
    "GaussianEmitter",
    "IsoLineEmitter",
    "DQDEmitterBase",
    "EmitterBase",
]
