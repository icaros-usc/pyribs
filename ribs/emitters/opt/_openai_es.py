"""Implementation of OpenAI ES that can be used across various emitters.

See here for more info: https://arxiv.org/abs/1703.03864
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import ParamSpec

from ribs._utils import arr_readonly
from ribs.emitters.opt._adam_opt import AdamOpt
from ribs.emitters.opt._evolution_strategy_base import (
    BOUNDS_SAMPLING_THRESHOLD,
    BOUNDS_WARNING,
    EvolutionStrategyBase,
)
from ribs.typing import Float, Int

P = ParamSpec("P")


class OpenAIEvolutionStrategy(EvolutionStrategyBase):
    """OpenAI-ES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0: Initial step size.
        batch_size: Number of solutions to evaluate at a time. If None, we calculate a
            default batch size based on solution_dim.
        solution_dim: Size of the solution space.
        seed: Seed for the random number generator.
        dtype: Data type of solutions.
        lower_bounds: scalar or (solution_dim,) array indicating lower bounds of the
            solution space. Scalars specify the same bound for the entire space, while
            arrays specify a bound for each dimension. Pass -np.inf in the array or
            scalar to indicated unbounded space.
        upper_bounds: Same as above, but for upper bounds (and pass np.inf instead of
            -np.inf).
        mirror_sampling: Whether to use mirror sampling when gathering solutions.
            Defaults to True.
        adam_kwargs: Keyword arguments passed to :class:`AdamOpt`.
    """

    def __init__(
        self,
        sigma0: Float,
        solution_dim: Int,
        batch_size: Int | None = None,
        seed: Int | None = None,
        dtype: DTypeLike = np.float64,
        lower_bounds: Float | np.ndarray = -np.inf,
        upper_bounds: Float | np.ndarray = np.inf,
        mirror_sampling: bool = True,
        **adam_kwargs: P.kwargs,
    ) -> None:
        self.batch_size = (
            4 + int(3 * np.log(solution_dim)) if batch_size is None else batch_size
        )
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype

        # Even scalars must be converted into 0-dim arrays so that they work
        # with the bound check in numba.
        self.lower_bounds = np.asarray(lower_bounds, dtype=self.dtype)
        self.upper_bounds = np.asarray(upper_bounds, dtype=self.dtype)

        self._rng = np.random.default_rng(seed)
        self._solutions = None

        if mirror_sampling and not (
            np.all(lower_bounds == -np.inf) and np.all(upper_bounds == np.inf)
        ):
            raise ValueError(
                "Bounds are currently not supported when using "
                "mirror_sampling in OpenAI-ES; see "
                "OpenAIEvolutionStrategy.ask() for more info."
            )

        self.mirror_sampling = mirror_sampling

        # Default batch size should be an even number for mirror sampling.
        if batch_size is None and self.batch_size % 2 != 0:
            self.batch_size += 1

        if self.batch_size <= 1:
            raise ValueError(
                "Batch size of 1 is not supported because rank"
                " normalization does not work with batch size of"
                " 1."
            )

        if self.mirror_sampling and self.batch_size % 2 != 0:
            raise ValueError(
                "If using mirror sampling, batch_size must be an even number."
            )

        # Strategy-specific params -> initialized in reset().
        self.adam_opt = AdamOpt(self.solution_dim, **adam_kwargs)
        self.last_update_ratio = None
        self.noise = None

    def reset(self, x0: np.ndarray) -> None:
        self.adam_opt.reset(x0)
        self.last_update_ratio = np.inf  # Updated at end of tell().
        self.noise = None  # Becomes (batch_size, solution_dim) array in ask().

    def check_stop(self, ranking_values: np.ndarray) -> bool:
        if self.last_update_ratio < 1e-9:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (  # noqa: SIM103
            len(ranking_values) >= 2
            and np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12
        ):
            return True

        return False

    def ask(self, batch_size: Int | None = None) -> np.ndarray:
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim), dtype=self.dtype)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        sampling_itrs = 0
        while len(remaining_indices) > 0:
            if self.mirror_sampling:
                # Note that we sample batch_size // 2 here rather than
                # accounting for len(remaining_indices). This is because we
                # assume we only run this loop once when mirror_sampling is
                # True. It is unclear how to do bounds handling when mirror
                # sampling is involved since the two entries need to be
                # mirrored. For instance, should we throw out both solutions if
                # one is out of bounds?
                noise_half = self._rng.standard_normal(
                    (batch_size // 2, self.solution_dim), dtype=self.dtype
                )
                self.noise = np.concatenate((noise_half, -noise_half))
            else:
                self.noise = self._rng.standard_normal(
                    (len(remaining_indices), self.solution_dim), dtype=self.dtype
                )

            new_solutions = self.adam_opt.theta[None] + self.sigma0 * self.noise
            out_of_bounds = np.logical_or(
                new_solutions < np.expand_dims(self.lower_bounds, axis=0),
                new_solutions > np.expand_dims(self.upper_bounds, axis=0),
            )

            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

            # Warn if we have resampled too many times.
            sampling_itrs += 1
            if sampling_itrs > BOUNDS_SAMPLING_THRESHOLD:
                warnings.warn(BOUNDS_WARNING, stacklevel=2)

        return arr_readonly(self._solutions)

    def tell(
        self, ranking_indices: np.ndarray, ranking_values: np.ndarray, num_parents: Int
    ) -> None:
        # Indices come in decreasing order, so we reverse to get them to
        # increasing order.
        ranks = np.empty(self.batch_size, dtype=np.int32)

        # Assign ranks -- ranks[i] tells the rank of noise[i].
        ranks[ranking_indices[::-1]] = np.arange(self.batch_size)

        # Normalize ranks to [-0.5, 0.5].
        ranks = (ranks / (self.batch_size - 1)) - 0.5

        # Compute the gradient.
        if self.mirror_sampling:
            half_batch = self.batch_size // 2
            gradient = np.sum(
                self.noise[:half_batch]
                * (ranks[:half_batch] - ranks[half_batch:])[:, None],
                axis=0,
            )
            gradient /= half_batch * self.sigma0
        else:
            gradient = np.sum(self.noise * ranks[:, None], axis=0)
            gradient /= self.batch_size * self.sigma0

        # Used to compute last update ratio.
        theta_prev = self.adam_opt.theta.copy()

        self.adam_opt.step(gradient)

        self.last_update_ratio = np.linalg.norm(
            self.adam_opt.theta - theta_prev
        ) / np.linalg.norm(self.adam_opt.theta)
