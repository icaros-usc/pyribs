"""Provides GaussianOperator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ribs.emitters.operators._operator_base import OperatorBase
from ribs.typing import Float, Int


class GaussianOperator(OperatorBase):
    """Adds Gaussian noise to solutions.

    Args:
        sigma: Standard deviation of the Gaussian distribution. Note we assume the
            Gaussian is diagonal, so if this argument is an array, it must be 1D.
        seed: Value to seed the random number generator. Set to None to avoid a fixed
            seed.
    """

    def __init__(self, sigma: Float | ArrayLike, seed: Int | None = None) -> None:
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self) -> int:
        """Parent Type to be used by selector."""
        return 1

    def ask(self, parents: ArrayLike) -> np.ndarray:
        """Adds Gaussian noise to parents.

        Args:
            parents: (batch_size, solution_dim) array of solutions to be mutated.

        Returns:
            ``(batch_size, solution_dim)`` array that contains ``batch_size`` mutated
            solutions.
        """
        parents = np.asarray(parents)
        noise = self._rng.normal(
            scale=self._sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(parents.dtype)
        return parents + noise
