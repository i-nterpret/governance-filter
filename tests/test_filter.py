"""Smoke tests for GovernanceFilter."""
import numpy as np
import pytest

from governance_filter import GovernanceFilter


def test_filter_zeros_high_interference():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((10, 4))
    filt = GovernanceFilter(W, threshold=0.0)
    x_input = np.zeros(10)
    x_hat = np.ones(10)
    out = filt.filter(x_input, x_hat)
    assert np.all(out == 0.0), "threshold=0 with no signal should suppress everything"


def test_filter_passes_clean_signal():
    W = np.eye(5)
    filt = GovernanceFilter(W, threshold=0.3)
    x_input = np.array([1.0, 0.0, 0.5, 0.0, 0.2])
    x_hat = x_input.copy()
    out = filt.filter(x_input, x_hat)
    np.testing.assert_allclose(out, x_hat, atol=1e-10)


def test_stability_index_in_unit_interval():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((20, 5))
    filt = GovernanceFilter(W)
    sigma = filt.stability_index()
    assert 0.0 <= sigma <= 1.0


def test_shape_validation():
    with pytest.raises(ValueError):
        GovernanceFilter(np.zeros(5))
    with pytest.raises(ValueError):
        GovernanceFilter(np.zeros((5, 3)), threshold=1.5)
