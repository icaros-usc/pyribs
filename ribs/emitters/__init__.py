"""Emitters output new candidate solutions in QD algorithms.

A note on data types: The emitters provided here will use the same data type as
the ones in the archive passed in.

.. autosummary::
    :toctree:

    ribs.emitters.GaussianEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.EmitterBase
"""
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter

__all__ = [
    "GaussianEmitter",
    "IsoLineEmitter",
    "EmitterBase",
]
