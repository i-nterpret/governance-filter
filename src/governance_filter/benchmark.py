"""
Benchmark — Elhage et al. (2022) toy-model reproduction harness.

Trains a linear autoencoder on sparse features, applies the GovernanceFilter
post-hoc, and measures false-activation elimination across sparsity regimes.

Target: 100% false-activation elimination across sparsity in {0.05, 0.10, 0.30}
with reconstruction-error increase in [4%, 31%].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from governance_filter.filter import GovernanceFilter


@dataclass
class BenchmarkResult:
    sparsity: float
    false_activations_ungoverned: float
    false_activations_governed: float
    reconstruction_error_ungoverned: float
    reconstruction_error_governed: float
    missed_activations_ungoverned: float
    missed_activations_governed: float
    eliminated_pct: float
    error_increase_pct: float


def generate_sparse_features(
    n_features: int,
    n_samples: int,
    sparsity: float,
    importance_decay: float = 0.7,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate sparse feature activations with exponentially decaying importances.

    Each feature k is active with probability sparsity, with magnitude scaled
    by importance_decay ** k.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    importances = importance_decay ** np.arange(n_features)
    active = rng.random((n_samples, n_features)) < sparsity
    magnitudes = rng.random((n_samples, n_features))
    return active * magnitudes * importances[None, :]


def train_toy_autoencoder(
    X: np.ndarray,
    n_dim: int,
    n_steps: int = 5000,
    lr: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Train a linear autoencoder W (n_features x n_dim) on X.

    Encoding: h = X @ W.T
    Decoding: X_hat = ReLU(h @ W)

    Minimizes ||X - X_hat||^2. Returns the trained W.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_features = X.shape[1]
    W = rng.standard_normal((n_features, n_dim)) * 0.1

    for _ in range(n_steps):
        h = X @ W.T
        X_hat = np.maximum(h @ W, 0)
        residual = X - X_hat
        active = (X_hat > 0).astype(float)
        grad = -2 * (residual * active).T @ h - 2 * X.T @ (residual * active)
        W -= lr * grad / X.shape[0]

    return W


def run_benchmark(
    sparsity: float,
    n_features: int = 20,
    n_dim: int = 5,
    n_samples: int = 2000,
    threshold: float = 0.3,
    activation_threshold: float = 0.05,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark at the specified sparsity level.

    Returns hallucination/precision metrics ungoverned vs governed.
    """
    rng = np.random.default_rng(seed)
    X = generate_sparse_features(n_features, n_samples, sparsity, rng=rng)
    W = train_toy_autoencoder(X, n_dim, rng=rng)
    filt = GovernanceFilter(W, threshold=threshold)

    h = X @ W.T
    X_hat = np.maximum(h @ W, 0)

    X_governed = np.zeros_like(X_hat)
    for i in range(X.shape[0]):
        X_governed[i] = filt.filter(X[i], X_hat[i])

    truly_active = X > activation_threshold
    appears_active_un = X_hat > activation_threshold
    appears_active_gov = X_governed > activation_threshold

    false_un = np.mean(appears_active_un & ~truly_active)
    false_gov = np.mean(appears_active_gov & ~truly_active)
    missed_un = np.mean(~appears_active_un & truly_active)
    missed_gov = np.mean(~appears_active_gov & truly_active)

    err_un = float(np.mean((X - X_hat) ** 2))
    err_gov = float(np.mean((X - X_governed) ** 2))

    return BenchmarkResult(
        sparsity=sparsity,
        false_activations_ungoverned=float(false_un),
        false_activations_governed=float(false_gov),
        reconstruction_error_ungoverned=err_un,
        reconstruction_error_governed=err_gov,
        missed_activations_ungoverned=float(missed_un),
        missed_activations_governed=float(missed_gov),
        eliminated_pct=100.0 * (1.0 - false_gov / max(false_un, 1e-12)),
        error_increase_pct=100.0 * (err_gov - err_un) / max(err_un, 1e-12),
    )


def run_full_benchmark(
    sparsities: tuple[float, ...] = (0.05, 0.10, 0.30),
    **kwargs,
) -> list[BenchmarkResult]:
    """Run the full benchmark across the standard sparsity regimes."""
    return [run_benchmark(sparsity=s, **kwargs) for s in sparsities]


if __name__ == "__main__":
    print(f"{'Sparsity':>10} {'False (un)':>12} {'False (gov)':>13} "
          f"{'Eliminated':>12} {'Δ Error %':>12}")
    print("-" * 65)
    for r in run_full_benchmark():
        print(f"{r.sparsity:>10.2f} {r.false_activations_ungoverned:>12.4f} "
              f"{r.false_activations_governed:>13.4f} {r.eliminated_pct:>11.1f}% "
              f"{r.error_increase_pct:>+11.1f}%")
