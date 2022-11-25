"""Various optimizers which are employed across emitters.

There are gradient-based optimizers which inherit from :class:`GradientOptBase`
and evolution strategies which inherit from :class:`EvolutionStrategyBase`. When
specifying optimizers for an emitter, one can pass in the optimizer class
itself, or the string name of the optimizer, or an abbreviated name. The
supported abbreviations are as follows.

For gradient-based optimizers (inheriting from :class:`GradientOptBase`):

* ``adam``: :class:`AdamOpt`
* ``gradient_ascent``: :class:`GradientAscentOpt`

For evolution strategies (inheriting from :class:`EvolutionStrategyBase`):

* ``cma_es``: :class:`CMAEvolutionStrategy`

.. autosummary::
    :toctree:

    ribs.emitters.opt.GradientAscentOpt
    ribs.emitters.opt.AdamOpt
    ribs.emitters.opt.GradientOptBase
    ribs.emitters.opt.CMAEvolutionStrategy
    ribs.emitters.opt.EvolutionStrategyBase
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
