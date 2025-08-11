"""Various operators that are employed across emitters.

Operators take in one or more parent solutions and output new solutions via operations
such as mutation and crossover. When specifying operators for an emitter, one can pass
in the operator class itself, or the string name of the operator, or an abbreviated
name. The supported abbreviations are:

* ``gaussian``: :class:`GaussianOperator`
* ``isoline``: :class:`IsoLineOperator`

.. autosummary::
    :toctree:

    GaussianOperator
    IsoLineOperator
    OperatorBase
"""

from ribs.emitters.operators._gaussian import GaussianOperator
from ribs.emitters.operators._iso_line import IsoLineOperator
from ribs.emitters.operators._operator_base import OperatorBase

__all__ = [
    "OperatorBase",
    "GaussianOperator",
    "IsoLineOperator",
]

_NAME_TO_OP_MAP = {
    "gaussian": GaussianOperator,
    "isoline": IsoLineOperator,
}


def _get_op(operator):
    """Retrieves Matching Operator."""
    if isinstance(operator, str):
        if operator in _NAME_TO_OP_MAP:
            operator = _NAME_TO_OP_MAP[operator]
            return operator
        else:
            raise ValueError(
                f"`{operator}` is not the full or abbreviated name of a valid operator"
            )

    raise ValueError(f"`{operator}` is not a string")
