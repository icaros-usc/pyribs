"""Selectors for use across emitters.

The selectors implemented in this file are intended to be used with emitters.
The ``Selector`` object will define the :meth:`select` method which returns the
num of top-performing candidates that should be taken as parents for the
evolution strategy. It will also define a :meth:`reset` method which resets
the internal state of the object.

.. autosummary::
    :toctree:

    ribs.emitters.selectors.MuSelector
    ribs.emitters.selectors.FilterSelector
    ribs.emitters.selectors.SelectorBase
"""
from abc import ABC, abstractmethod

import numpy as np

from ribs._docstrings import DocstringComponents, core_args

__all__ = [
    "MuSelector",
    "FilterSelector",
    "SelectorBase",
]

_args = DocstringComponents(core_args)

_select_args = f"""
Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
    The number of top-performing solutions to select as parents.
"""

_reset_args = f"""
Args:
{_args.emitter}
{_args.archive}
"""


class SelectorBase(ABC):
    """Base class for selectors.

    Every selector has a :meth:`select` method that returns the number of
    top-performing parents to select from the ranking and a :meth:`reset`
    method that resets the internal state of the selector.

    Child classes are only required to override :meth:`select`.
    """
    # pylint: disable=missing-function-docstring

    @abstractmethod
    def select(self, emitter, archive, rng, solution_batch, objective_batch,
               measures_batch, metadata, add_statuses, add_values):
        pass

    select.__doc__ = f"""
Selects the number of parents that will be used for the evolution strategy.

{_select_args}
    """

    def reset(self, emitter, archive, rng):
        pass

    reset.__doc__ = f"""
Resets the internal state of the selector.

{_reset_args}
   """


class MuSelector(SelectorBase):
    """Implementation of MuSelector.

    This selector will select the top :math:`\\mu` solutions
    as parents.
    """

    def select(self, emitter, archive, rng, solution_batch, objective_batch,
               measures_batch, metadata, add_statuses, add_values):
        return emitter.batch_size // 2

    select.__doc__ = f"""
Selects the top :math:`\\mu` solutions.

{_select_args}
    """


class FilterSelector(SelectorBase):
    """Implementation of FilterSelector.

    This selector will select all the *added* and *improved* solutions
    as parents.
    """

    def select(self, emitter, archive, rng, solution_batch, objective_batch,
               measures_batch, metadata, add_statuses, add_values):
        return np.array(add_statuses).astype(bool).sum()

    select.__doc__ = f"""
Selects all the added and improved solutions.

{_select_args}
    """
