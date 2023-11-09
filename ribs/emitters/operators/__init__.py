"""Operators Act as Mutation Functions to archive solutions
Used in tandem with emitter classes to alter solutions. Supports
Pymoo external operators
"""

from ribs.emitters.operators._gaussian import GaussianOperator
from ribs.emitters.operators._iso_line import IsoLineOperator
from ribs.emitters.operators._pymoo_gaussian import PymooGaussianOperator

_NAME_TO_OP_MAP = {
    "pymooGaussian": PymooGaussianOperator,
    "Gaussian": GaussianOperator,
    "IsoLine": IsoLineOperator,
}


def _get_op(operator):
    """Retrieves Matching Operator"""
    if isinstance(operator, str):
        if operator in _NAME_TO_OP_MAP:
            operator = _NAME_TO_OP_MAP[operator]
            return operator
        else:
            raise ValueError(f"`{operator}` is not the full or abbreviated "
                             "name of a valid operator")

    raise ValueError(f"`{operator}` is not a string")
