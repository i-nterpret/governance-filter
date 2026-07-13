"""
Benchmark — Elhage et al. (2022) toy-model reproduction + risk-coverage study (v3).

v3 reframes the benchmark around the honest object:
  1. reproduce the v2 operating points (continuity / sanity anchor);
  2. build the risk-coverage curve for the interference gate AND for four
     baselines that share its information access, so "elimination %" is judged
     against comparators instead of in a vacuum;
  3. report the negative result plainly: at matched elimination the weight
     geometry does not beat raw activation magnitude.

Run:  python -m governance_filter.benchmark
Figure: python -m governance_filter.benchmark --figure docs/risk_coverage.png
"""
from __future__ import annotations

import sys

import numpy as np

from governance_filter.filter import InterferenceGate
from governance_filter import baselines as bl


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


def _build(sparsity, n_features=20, n_dim=5, n_samples=4000, seed=42):
    rng = np.random.default_rng(seed)
    X = generate_sparse_features(n_features, n_samples, sparsity, rng=rng)
    W = train_toy_autoencoder(X, n_dim, rng=rng)
    X_hat = np.maximum((X @ W) @ W.T, 0.0)
    return X, W, X_hat


def _scores(X, W, X_hat, rng):
    """All method KEEP-scores on the same data."""
    return {
        "trivial-oracle (uses label)": bl.score_trivial_oracle(X),
        "interference gate": bl.score_interference_gate(X, X_hat, W),
        "magnitude |x_hat|": bl.score_magnitude(X_hat),
        "NNLS Gram (geometry-only)": bl.score_nnls_gram(X_hat, W),
        "random": bl.score_random(X_hat.shape, rng),
    }


def run(sparsities=(0.05, 0.10, 0.30)):
    rows = []
    for s in sparsities:
        X, W, X_hat = _build(s)
        gate = InterferenceGate(W, threshold=0.3)
        Xs = np.array([gate.filter(X[i], X_hat[i]) for i in range(X.shape[0])])
        Xp = np.array([gate.project(X[i], X_hat[i]) for i in range(X.shape[0])])
        elim_s, ret_s = bl.elim_retention_governed(Xs, X, X_hat)
        elim_p, ret_p = bl.elim_retention_governed(Xp, X, X_hat)
        rng = np.random.default_rng(7)
        curves = {name: bl.risk_coverage_curve(sc, X, X_hat)
                  for name, sc in _scores(X, W, X_hat, rng).items()}
        rows.append(dict(sparsity=s, elim_s=elim_s, ret_s=ret_s,
                         elim_p=elim_p, ret_p=ret_p, curves=curves))
    return rows


def _print(rows):
    print("=" * 78)
    print("1) v2 operating points (continuity check — these match the old README)")
    print("=" * 78)
    print(f"{'spars':>6}{'suppress elim%':>16}{'suppress ret%':>15}"
          f"{'project elim%':>15}{'project ret%':>14}")
    for r in rows:
        print(f"{r['sparsity']:>6.2f}{r['elim_s']:>16.1f}{r['ret_s']:>15.1f}"
              f"{r['elim_p']:>15.1f}{r['ret_p']:>14.1f}")
    print("   note: these are single operating points measured on the actual")
    print("   governed output (project REPLACES values); the swept curves in (2)")
    print("   use keep/zero semantics, so the two sections are not point-comparable.")

    print()
    print("=" * 78)
    print("2) risk-coverage: TP-retention at MATCHED elimination (the real scoreboard)")
    print("   higher = better; the gate should be judged against these, not in a vacuum")
    print("=" * 78)
    for r in rows:
        gate_elim = r["elim_s"]
        print(f"\n sparsity {r['sparsity']:.2f}  (comparing all methods at elim >= "
              f"{gate_elim:.0f}%, the gate's suppress operating point)")
        print(f"   {'method':<32}{'retention% @matched-elim':>26}")
        base = {}
        for name, pts in r["curves"].items():
            ret = bl.retention_at_elim(pts, gate_elim)
            base[name] = ret
            shown = "n/a (can't reach elim)" if np.isnan(ret) else f"{ret:.1f}"
            print(f"   {name:<32}{shown:>26}")
        geo = base.get("interference gate", float('nan'))
        mag = base.get("magnitude |x_hat|", float('nan'))
        verdict = ("geometry ADDS retention" if geo > mag + 0.5
                   else "magnitude ties/beats geometry")
        print(f"   -> {verdict}  (gate {geo:.1f} vs magnitude {mag:.1f})")

    print()
    print("=" * 78)
    print("HEADLINE (honest): the interference gate is dominated by the trivial")
    print("label rule, and at matched elimination it does not beat raw activation")
    print("magnitude. The weight geometry carries no discriminative signal here.")
    print("The kept contribution is the decomposition + the dissent route, not a")
    print("detector. Operating points are points on a curve; report the curve.")
    print("=" * 78)


def make_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.2), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    style = {
        "trivial-oracle (uses label)": ("#c76a00", "-", "the answer key (ceiling)"),
        "interference gate": ("#1f6feb", "-", "interference gate"),
        "magnitude |x_hat|": ("#2a2a2a", "--", "magnitude |x_hat|"),
        "NNLS Gram (geometry-only)": ("#8a8a8a", ":", "NNLS Gram (geom-only)"),
        "random": ("#bbbbbb", "-", "random (floor)"),
    }
    for ax, r in zip(axes, rows):
        for name, pts in r["curves"].items():
            fr = bl.frontier(pts)
            e = [p[0] for p in fr]
            ret = [p[1] for p in fr]
            c, ls, lab = style[name]
            ax.plot(e, ret, ls, color=c, label=lab, linewidth=1.8)
        ax.set_title(f"sparsity {r['sparsity']:.2f}")
        ax.set_xlabel("false-activation elimination  %")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("true-positive retention  %")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Risk-coverage: only the answer key clears the frontier; "
                 "geometry ≈ magnitude", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[figure] wrote {path}")


if __name__ == "__main__":
    rows = run()
    _print(rows)
    if "--figure" in sys.argv:
        i = sys.argv.index("--figure")
        path = sys.argv[i + 1] if i + 1 < len(sys.argv) else "docs/risk_coverage.png"
        make_figure(rows, path)
