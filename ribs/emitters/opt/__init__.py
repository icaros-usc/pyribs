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
* ``sep_cma_es``: :class:`SeparableCMAEvolutionStrategy`
* ``lm_ma_es``: :class:`LMMAEvolutionStrategy`
* ``openai_es``: :class:`OpenAIEvolutionStrategy`

.. autosummary::
    :toctree:

    ribs.emitters.opt.CMAEvolutionStrategy
    ribs.emitters.opt.LMMAEvolutionStrategy
    ribs.emitters.opt.OpenAIEvolutionStrategy
    ribs.emitters.opt.SeparableCMAEvolutionStrategy
    ribs.emitters.opt.EvolutionStrategyBase
    ribs.emitters.opt.AdamOpt
    ribs.emitters.opt.GradientAscentOpt
    ribs.emitters.opt.GradientOptBase
"""
from ribs.emitters.opt._adam_opt import AdamOpt
from ribs.emitters.opt._cma_es import CMAEvolutionStrategy
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase
from ribs.emitters.opt._gradient_ascent_opt import GradientAscentOpt
from ribs.emitters.opt._gradient_opt_base import GradientOptBase
from ribs.emitters.opt._lm_ma_es import LMMAEvolutionStrategy
from ribs.emitters.opt._openai_es import OpenAIEvolutionStrategy
from ribs.emitters.opt._sep_cma_es import SeparableCMAEvolutionStrategy

__all__ = [
    "CMAEvolutionStrategy",
    "LMMAEvolutionStrategy",
    "OpenAIEvolutionStrategy",
    "SeparableCMAEvolutionStrategy",
    "EvolutionStrategyBase",
    "AdamOpt",
    "GradientAscentOpt",
    "GradientOptBase",
]

_NAME_TO_GRAD_OPT_MAP = {
    "AdamOpt": AdamOpt,
    "GradientAscentOpt": GradientAscentOpt,
    "adam": AdamOpt,
    "gradient_ascent": GradientAscentOpt,
}


def _get_grad_opt(klass, **grad_opt_kwargs):
    """Returns a gradient optimizer class based on its name.

    Args:
        klass: Either a callable or a str for the gradient optimizer.
        grad_opt_kwargs (dict): Additional kwargs for the gradient optimizer.

    Returns:
        The corresponding gradient optimizer class instance.
    """
    if isinstance(klass, str):
        if klass in _NAME_TO_GRAD_OPT_MAP:
            klass = _NAME_TO_GRAD_OPT_MAP[klass]
        else:
            raise ValueError(f"`{klass}` is not the full or abbreviated "
                             "name of a valid gradient optimizer")
    if callable(klass):
        grad_opt = klass(**grad_opt_kwargs)
        if isinstance(grad_opt, GradientOptBase):
            return grad_opt
        raise ValueError(f"Callable `{klass}` did not return an instance "
                         "of GradientOptBase.")
    raise ValueError(f"`{klass}` is neither a callable nor a string")


_NAME_TO_ES_MAP = {
    "CMAEvolutionStrategy": CMAEvolutionStrategy,
    "SeparableCMAEvolutionStrategy": SeparableCMAEvolutionStrategy,
    "LMMAEvolutionStrategy": LMMAEvolutionStrategy,
    "OpenAIEvolutionStrategy": OpenAIEvolutionStrategy,
    "cma_es": CMAEvolutionStrategy,
    "sep_cma_es": SeparableCMAEvolutionStrategy,
    "lm_ma_es": LMMAEvolutionStrategy,
    "openai_es": OpenAIEvolutionStrategy,
}


def _get_es(klass, **es_kwargs):
    """Returns an evolution strategy (ES) class based on its name.

    Args:
        klass: Either a callable or a str for the ES.
        es_kwargs (dict): Additional keyword arguments for the ES.

    Returns:
        The corresponding evolution strategy class instance.
    """
    if isinstance(klass, str):
        if klass in _NAME_TO_ES_MAP:
            klass = _NAME_TO_ES_MAP[klass]
        else:
            raise ValueError(f"`{klass}` is not the full or abbreviated "
                             "name of a valid evolution strategy")
    if callable(klass):
        es = klass(**es_kwargs)
        if isinstance(es, EvolutionStrategyBase):
            return es
        raise ValueError(f"Callable `{klass}` did not return an instance "
                         "of EvolutionStrategyBase.")
    raise ValueError(f"`{klass}` is neither a callable nor a string")
