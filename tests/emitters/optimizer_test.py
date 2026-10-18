"""Tests for the Optimizers."""

from typing import cast

import numpy as np
import pytest

from ribs.emitters.opt import (
    _NAME_TO_ES_MAP,
    _NAME_TO_GRAD_OPT_MAP,
    AdamOpt,
    GradientAscentOpt,
    OpenAIEvolutionStrategy,
    _get_es,
    _get_grad_opt,
)

# Evolution Strategy Tests -- see here for why we sort parameters:
# https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html


@pytest.mark.parametrize(
    "es_name",
    # Exclude since we only test core library in these tests.
    sorted(_NAME_TO_ES_MAP.keys() - {"PyCMAEvolutionStrategy", "pycma_es"}),
)
def test_init_with_get_es(es_name):
    es_kwargs = {
        "sigma0": 1.0,
        "batch_size": 4,
        "solution_dim": 10,
        "dtype": np.float32,
    }
    es = _get_es(es_name, **es_kwargs)

    # Technically, these attributes are not part of the API, but all of our ES's have
    # them.
    assert es.batch_size == es_kwargs["batch_size"]
    assert es.sigma0 == es_kwargs["sigma0"]
    assert es.solution_dim == es_kwargs["solution_dim"]
    assert es.dtype == es_kwargs["dtype"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["bounded", "unbounded", "mirrored"])
def test_openai_es_update_uses_sampled_perturbations(dtype, mode):
    x0 = np.array([0.9, 0.8, 0.7], dtype=dtype)
    sigma = 0.5
    es = OpenAIEvolutionStrategy(
        sigma0=sigma,
        solution_dim=3,
        batch_size=16,
        seed=42,
        dtype=dtype,
        lower_bounds=-1.0 if mode == "bounded" else -np.inf,
        upper_bounds=1.0 if mode == "bounded" else np.inf,
        mirror_sampling=mode == "mirrored",
        beta1=0.0,
        beta2=0.0,
        epsilon=1.0,
        lr=0.1,
    )
    es.reset(x0)
    solutions = es.ask()
    if mode == "bounded":
        assert np.all(solutions >= -1.0)
        assert np.all(solutions <= 1.0)

    objectives = solutions @ np.array([1.0, 2.0, 3.0])
    ranking_indices = np.argsort(-objectives)
    weights = np.empty(len(solutions))
    weights[ranking_indices[::-1]] = np.linspace(-0.5, 0.5, len(solutions))
    # Reconstruct the estimator from evaluated solutions, independently of es.noise.
    gradient = np.sum((solutions - x0) * weights[:, None], axis=0)
    gradient /= len(solutions) * sigma**2
    if mode == "mirrored":
        gradient *= 2  # Preserve the mirrored estimator's existing scaling.
    expected_theta = x0 + 0.1 * gradient / (np.abs(gradient) + 1.0)

    es.tell(ranking_indices, objectives[ranking_indices], num_parents=8)

    np.testing.assert_allclose(es.adam_opt.theta, expected_theta, atol=1e-7)


# Gradient Optimizer Tests


@pytest.mark.parametrize("grad_opt_name", sorted(_NAME_TO_GRAD_OPT_MAP.keys()))
def test_init_with_get_grad_opt(grad_opt_name):
    theta0 = 2.0
    lr = 1

    if grad_opt_name == "adam":
        grad_opt_kwargs = {
            "beta1": 0.1,
            "beta2": 0.1,
            "epsilon": 1,
            "l2_coeff": 1,
        }
    else:
        grad_opt_kwargs = {}

    grad_opt = _get_grad_opt(grad_opt_name, theta0=theta0, lr=lr, **grad_opt_kwargs)

    assert grad_opt.theta == theta0

    # pylint: disable = protected-access
    grad_opt = cast(GradientAscentOpt | AdamOpt, grad_opt)
    assert grad_opt._lr == lr
    if grad_opt_name == "adam":
        grad_opt = cast(AdamOpt, grad_opt)
        assert grad_opt._beta1 == grad_opt_kwargs["beta1"]
        assert grad_opt._beta2 == grad_opt_kwargs["beta2"]
        assert grad_opt._epsilon == grad_opt_kwargs["epsilon"]
        assert grad_opt._l2_coeff == grad_opt_kwargs["l2_coeff"]
