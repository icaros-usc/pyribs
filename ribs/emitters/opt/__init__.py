"""Optimizers for use across emitters.

Optimizers based on evolution strategies inherit from :class:`OptimizerBase`.

.. autosummary::
    :toctree:

    ribs.emitters.opt.CMAEvolutionStrategy
    ribs.emitters.opt.LMMAEvolutionStrategy
    ribs.emitters.opt.OpenAIEvolutionStrategy
    ribs.emitters.opt.SeparableCMAEvolutionStrategy
    ribs.emitters.opt.AdamOpt
    ribs.emitters.opt.GradientAscentOpt
    ribs.emitters.opt.EvolutionStrategyBase
"""
from ribs.emitters.opt._cma_es import CMAEvolutionStrategy
from ribs.emitters.opt._gradients import AdamOpt, GradientAscentOpt
from ribs.emitters.opt._lm_ma_es import LMMAEvolutionStrategy
from ribs.emitters.opt._openai_es import OpenAIEvolutionStrategy
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase
from ribs.emitters.opt._sep_cma_es import SeparableCMAEvolutionStrategy

__all__ = [
    "CMAEvolutionStrategy",
    "LMMAEvolutionStrategy",
    "OpenAIEvolutionStrategy",
    "SeparableCMAEvolutionStrategy",
    "AdamOpt",
    "GradientAscentOpt",
    "EvolutionStrategyBase",
]

_NAME_TO_OPTIMIZER_MAP = {
    "CMAEvolutionStrategy": CMAEvolutionStrategy,
    "SeparableCMAEvolutionStrategy": SeparableCMAEvolutionStrategy,
    "LMMAEvolutionStrategy": LMMAEvolutionStrategy,
    "OpenAIEvolutionStrategy": OpenAIEvolutionStrategy,
    "AdamOpt": AdamOpt,
    "GradientAscentOpt": GradientAscentOpt,
    "cma_es": CMAEvolutionStrategy,
    "sep_cma_es": SeparableCMAEvolutionStrategy,
    "lm_ma_es": LMMAEvolutionStrategy,
    "openai_es": OpenAIEvolutionStrategy,
    "adam": AdamOpt,
    "gradient_ascent": GradientAscentOpt,
}


def _get_optimizer(klass):
    """Returns a optimizer class based on its name.

    Args:
        klass (str): This parameter has to be the full or abbreviated optimizer
            name.

    Returns:
        The corresponding optimizer class.
    """
    # TODO we might want to allow klass to be the actually class of the
    # optimizer, i.e. _get_optimizer(CMAEvolutionStrategy).
    if klass in _NAME_TO_OPTIMIZER_MAP:
        return _NAME_TO_OPTIMIZER_MAP[klass]
    raise ValueError(f"Unknown optimizer '{klass}'.")
