"""Various optimizers which are employed across emitters.

There are gradient-based optimizers which inherit from :class:`GradientOptBase`
and evolution strategies which inherit from :class:`EvolutionStrategyBase`.

.. autosummary::
    :toctree:

    ribs.emitters.opt.GradientOptBase
    ribs.emitters.opt.GradientAscentOpt
    ribs.emitters.opt.AdamOpt
    ribs.emitters.opt.EvolutionStrategyBase
    ribs.emitters.opt.CMAEvolutionStrategy
"""
from ribs.emitters.opt._adam_opt import AdamOpt
from ribs.emitters.opt._cma_es import CMAEvolutionStrategy
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase
from ribs.emitters.opt._gradient_ascent_opt import GradientAscentOpt
from ribs.emitters.opt._gradient_opt_base import GradientOptBase

__all__ = [
    "AdamOpt",
    "GradientAscentOpt",
    "GradientOptBase",
    "CMAEvolutionStrategy",
    "EvolutionStrategyBase",
]
