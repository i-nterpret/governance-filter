"""Tests for InterferenceGate + the trainer gradient + the dissent route."""
import numpy as np
import pytest

from governance_filter import InterferenceGate, RouteResult, GovernanceFilter
from governance_filter.benchmark import train_toy_autoencoder, generate_sparse_features


# --- core behaviour --------------------------------------------------------

def test_filter_zeros_high_interference():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((10, 4))
    gate = InterferenceGate(W, threshold=0.0)
    out = gate.filter(np.zeros(10), np.ones(10))
    assert np.all(out == 0.0), "threshold=0 with no signal should suppress everything"


def test_filter_passes_clean_signal():
    gate = InterferenceGate(np.eye(5), threshold=0.3)
    x = np.array([1.0, 0.0, 0.5, 0.0, 0.2])
    np.testing.assert_allclose(gate.filter(x, x.copy()), x, atol=1e-10)


def test_deprecated_alias_is_the_class():
    assert GovernanceFilter is InterferenceGate


# --- the crash the v2 code had (n_features = 1) ----------------------------

def test_spectral_gap_ratio_single_feature_no_crash():
    gate = InterferenceGate(np.ones((1, 3)))
    assert gate.spectral_gap_ratio() == 0.0   # guarded, was an IndexError
    assert isinstance(repr(gate), str)          # __repr__ used to crash too


def test_spectral_gap_ratio_in_unit_interval():
    rng = np.random.default_rng(1)
    gate = InterferenceGate(rng.standard_normal((20, 5)))
    assert 0.0 <= gate.spectral_gap_ratio() <= 1.0


def test_stable_rank_distinguishes_bulk():
    # two Grams with the same top-2 gap but different bulk -> different stable rank
    # build W so W W^T has controlled eigenvalues via orthonormal columns scaled
    q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((5, 5)))
    flat = q @ np.diag([1, .5, .5, .5, .5]) @ q.T
    peaked = q @ np.diag([1, .5, .01, .01, .01]) @ q.T
    # feed as "W" whose Gram is these (symmetric PSD sqrt)
    gate_flat = InterferenceGate(np.linalg.cholesky(flat + 1e-9*np.eye(5)))
    gate_peak = InterferenceGate(np.linalg.cholesky(peaked + 1e-9*np.eye(5)))
    assert gate_flat.stable_rank() > gate_peak.stable_rank() + 0.5


# --- project does NOT strictly dominate filter (the corrected docstring) ----

def test_project_can_keep_a_false_activation_filter_removes():
    # one feature, ||w||^2 = 4, tiny absent input amplifies over the act line
    W = np.array([[2.0, 0.0]])          # ||w||^2 = 4
    gate = InterferenceGate(W, threshold=0.3)
    x_input = np.array([0.03])           # "absent" (below a 0.05 act threshold)
    x_hat = np.array([0.20])             # reconstructed, appears
    filtered = gate.filter(x_input, x_hat)   # -> 0.0, false activation removed
    projected = gate.project(x_input, x_hat)  # -> clip(0.03*4=0.12) = 0.12, KEPT
    assert filtered[0] == 0.0
    assert projected[0] > 0.05, "project keeps what filter removed -> not dominant"


# --- the dissent route conserves signal ------------------------------------

def test_route_conserves_signal():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((12, 4))
    gate = InterferenceGate(W, threshold=0.3)
    x_input = rng.random(12)
    x_hat = np.abs(rng.random(12))
    r = gate.route(x_input, x_hat)
    assert isinstance(r, RouteResult)
    np.testing.assert_allclose(r.governed + r.dissent, x_hat, atol=1e-10)
    # dissent is nonzero only on routed features
    assert np.all(r.dissent[~r.routed] == 0.0)
    # governed matches project() on the same inputs
    np.testing.assert_allclose(r.governed, gate.project(x_input, x_hat), atol=1e-12)


# --- the trainer gradient is exact (finite-difference check) ---------------

def test_trainer_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    X = generate_sparse_features(6, 40, 0.3, rng=rng)
    W = rng.standard_normal((6, 3)) * 0.1
    n = X.shape[0]

    def loss(Wm):
        # matches the trainer's normalization: sum over features, /n_samples
        Z = (X @ Wm) @ Wm.T
        Xhat = np.maximum(Z, 0.0)
        return float(np.sum((X - Xhat) ** 2) / n)

    # analytic gradient (the trainer's formula)
    Z = (X @ W) @ W.T
    Xhat = np.maximum(Z, 0.0)
    R = X - Xhat
    M = (Z > 0).astype(float)
    G = -2.0 * (R * M) / n
    grad = X.T @ G @ W + G.T @ X @ W

    # finite differences
    eps = 1e-6
    fd = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            fd[i, j] = (loss(Wp) - loss(Wm)) / (2 * eps)
    assert np.max(np.abs(grad - fd)) < 1e-5


# --- shape validation ------------------------------------------------------

def test_shape_validation():
    with pytest.raises(ValueError):
        InterferenceGate(np.zeros(5))
    with pytest.raises(ValueError):
        InterferenceGate(np.zeros((5, 3)), threshold=1.5)
