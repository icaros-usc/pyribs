"""Emitters output new candidate solutions in QD algorithms.

A note on data types: The emitters provided here will use the same data type as
the ones in the archive passed in.

.. autosummary::
    :toctree:

    ribs.emitters.GaussianEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.ImprovementEmitter
    ribs.emitters.RandomDirectionEmitter
    ribs.emitters.OptimizingEmitter
    ribs.emitters.EmitterBase
"""
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._improvement_emitter import ImprovementEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter
from ribs.emitters._optimizing_emitter import OptimizingEmitter
from ribs.emitters._random_direction_emitter import RandomDirectionEmitter

__all__ = [
    "GaussianEmitter",
    "IsoLineEmitter",
    "ImprovementEmitter",
    "RandomDirectionEmitter",
    "OptimizingEmitter",
    "EmitterBase",
]
