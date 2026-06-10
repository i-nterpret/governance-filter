"""
Benchmark — Elhage et al. (2022) toy-model reproduction harness.  (v2, 2026-06)

v2 CHANGES
  1. TRAINER FIX: the published train_toy_autoencoder transposed the encode/decode
     (X @ W.T where the dims need X @ W) and dropped an @W in the gradient, so it
     did not run. Corrected to a tied-weight linear autoencoder:
         encode  h     = X @ W           (n_samples, n_dim)
         decode  X_hat = ReLU(h @ W.T)   (n_samples, n_features)
         dL/dW   = X.T @ G @ W + G.T @ X @ W,   G = -2(R*M)/n
  2. SURFACES THE SECOND NUMBER: reports true-positive RETENTION beside elimination.
     The published headline reported only elimination ("100%"); the retention cost
     (missed_activations) was computed but never shown. Both are reported now.
  3. COMPARES suppress() vs project() so the cost and its fix are visible side by side.
"""
from __future__ import annotations

import numpy as np

from governance_filter.filter import GovernanceFilter


def generate_sparse_features(n_features, n_samples, sparsity, importance_decay=0.7, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    importances = importance_decay ** np.arange(n_features)
    active = rng.random((n_samples, n_features)) < sparsity
    magnitudes = rng.random((n_samples, n_features))
    return active * magnitudes * importances[None, :]


def train_toy_autoencoder(X, n_dim, n_steps=3000, lr=0.05, rng=None):
    """Tied-weight linear autoencoder.  h = X @ W ; X_hat = ReLU(h @ W.T)."""
    if rng is None:
        rng = np.random.default_rng(0)
    nf, n = X.shape[1], X.shape[0]
    W = rng.standard_normal((nf, n_dim)) * 0.1
    for _ in range(n_steps):
        h = X @ W
        Z = h @ W.T
        X_hat = np.maximum(Z, 0.0)
        R = X - X_hat
        M = (Z > 0).astype(float)
        G = -2.0 * (R * M) / n
        gradW = X.T @ G @ W + G.T @ X @ W
        W -= lr * gradW
    return W


def _metrics(Xg, X, X_hat, act):
    truly = X > act
    appears = Xg > act
    appears_un = X_hat > act
    false_un = float(np.mean(appears_un & ~truly))
    false = float(np.mean(appears & ~truly))
    elim = 100.0 * (1.0 - false / max(false_un, 1e-12))
    tp = int(np.sum(appears & truly))
    fn = int(np.sum(~appears & truly))
    retention = 100.0 * tp / max(tp + fn, 1)
    err = float(np.mean((X - Xg) ** 2))
    return false * 100, elim, retention, err


def run_benchmark(sparsity, n_features=20, n_dim=5, n_samples=4000,
                  threshold=0.3, activation_threshold=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = generate_sparse_features(n_features, n_samples, sparsity, rng=rng)
    W = train_toy_autoencoder(X, n_dim, rng=rng)
    filt = GovernanceFilter(W, threshold=threshold)
    h = X @ W
    X_hat = np.maximum(h @ W.T, 0.0)
    Xs = np.array([filt.filter(X[i], X_hat[i]) for i in range(n_samples)])
    Xp = np.array([filt.project(X[i], X_hat[i]) for i in range(n_samples)])
    return {
        "sparsity": sparsity,
        "ungoverned": _metrics(X_hat, X, X_hat, activation_threshold),
        "suppress": _metrics(Xs, X, X_hat, activation_threshold),
        "project": _metrics(Xp, X, X_hat, activation_threshold),
        "train_err": float(np.mean((X - X_hat) ** 2)),
    }


if __name__ == "__main__":
    rows = [run_benchmark(s) for s in (0.05, 0.10, 0.30)]
    hdr = f"{'spars':>6} {'method':>11} {'false%':>8} {'elim%':>7} {'TP-retain%':>11} {'recon-err':>10}"
    print(hdr); print("-" * len(hdr))
    elim_p, ret_p, elim_s, ret_s = [], [], [], []
    for r in rows:
        fu = r["ungoverned"]
        print(f"{r['sparsity']:>6.2f} {'ungoverned':>11} {fu[0]:>8.3f} {'—':>7} {fu[2]:>11.1f} {fu[3]:>10.5f}")
        for name in ("suppress", "project"):
            f, e, ret, err = r[name]
            print(f"{'':>6} {name:>11} {f:>8.3f} {e:>7.1f} {ret:>11.1f} {err:>10.5f}")
            if name == "project":
                elim_p.append(e); ret_p.append(ret)
            else:
                elim_s.append(e); ret_s.append(ret)
        print()
    print("=" * 56)
    print(f"SUPPRESS (published): elim {np.mean(elim_s):.1f}%  TP-retention {np.mean(ret_s):.1f}%")
    print(f"PROJECT  (v2)       : elim {np.mean(elim_p):.1f}%  TP-retention {np.mean(ret_p):.1f}%")
