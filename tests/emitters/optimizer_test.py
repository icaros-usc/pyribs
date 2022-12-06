"""Tests for the Optimizers."""
import numpy as np
import pytest

from ribs.emitters.opt import (_NAME_TO_ES_MAP, _NAME_TO_GRAD_OPT_MAP, _get_es,
                               _get_grad_opt)

# Evolution Strategy Tests


@pytest.mark.parametrize("es_name", _NAME_TO_ES_MAP.keys())
def test_init_with_get_es(es_name):
    es_kwargs = {
        "sigma0": 1.0,
        "batch_size": 4,
        "solution_dim": 10,
        "dtype": np.float32,
    }
    es = _get_es(es_name, **es_kwargs)

    assert es.batch_size == es_kwargs["batch_size"]
    assert es.sigma0 == es_kwargs["sigma0"]
    assert es.solution_dim == es_kwargs["solution_dim"]
    assert es.dtype == es_kwargs["dtype"]


# Gradient Optimizer Tests


@pytest.mark.parametrize("grad_opt_name", _NAME_TO_GRAD_OPT_MAP.keys())
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

    grad_opt = _get_grad_opt(grad_opt_name,
                             theta0=theta0,
                             lr=lr,
                             **grad_opt_kwargs)

    assert grad_opt.theta == theta0

    # pylint: disable = protected-access
    assert grad_opt._lr == lr
    if grad_opt_name == "adam":
        assert grad_opt._beta1 == grad_opt_kwargs["beta1"]
        assert grad_opt._beta2 == grad_opt_kwargs["beta2"]
        assert grad_opt._epsilon == grad_opt_kwargs["epsilon"]
        assert grad_opt._l2_coeff == grad_opt_kwargs["l2_coeff"]
