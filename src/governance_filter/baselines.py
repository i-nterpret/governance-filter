"""
Baseline reject-option methods + the risk-coverage machinery (v3).

The point of this module is falsifiability: "the interference gate eliminates
84.3% of false activations" means nothing without comparators. A false
activation is rare, so *random* suppression of a matched budget already scores
high elimination — elimination alone is a weak scoreboard. The honest object is
the **risk-coverage curve**: retention (coverage of true activations) as a
function of false-activation elimination (risk removed), swept over each
method's threshold.

All methods return a per-entry KEEP score (higher = keep). A method is turned
into an operating point by thresholding that score; sweeping the threshold
traces the curve. Labels (true vs. false activation) are used only to *evaluate*
a curve, never inside a score — except TRIVIAL_ORACLE, which is deliberately the
answer key, to expose the ceiling.

Methods
-------
- trivial_oracle       : score = x_input        (uses the label; ceiling)
- magnitude            : score = |x_hat|         (no x_input, no geometry)
- random               : score = uniform noise   (floor)
- interference_gate    : score = -interference_ratio (uses x_input + norms)
- nnls_gram            : score = NNLS Gram-consistency support (geometry-only)
"""
from __future__ import annotations

import numpy as np

ACT = 0.05  # activation threshold that defines "appears" / "truly present"


# --------------------------------------------------------------------------- #
# per-entry KEEP scores (higher score => more likely to be kept)              #
# --------------------------------------------------------------------------- #

def score_magnitude(X_hat: np.ndarray) -> np.ndarray:
    """No x_input, no geometry: the raw activation magnitude."""
    return np.abs(X_hat)


def score_trivial_oracle(X: np.ndarray) -> np.ndarray:
    """The answer key: the true input value. Exposes the ceiling."""
    return np.abs(X)


def score_random(shape, rng) -> np.ndarray:
    """The floor: uniform noise, uncorrelated with anything."""
    return rng.random(shape)


def score_interference_gate(X: np.ndarray, X_hat: np.ndarray, W: np.ndarray,
                            eps: float = 1e-8) -> np.ndarray:
    """The gate's own score as a KEEP score: -(interference ratio).

    Uses x_input and the weight norms (same access as project/filter).
    """
    norms_sq = np.sum(W ** 2, axis=1)
    signal = X * norms_sq
    ratio = np.abs(X_hat - signal) / (np.abs(X_hat) + eps)
    return -ratio


def score_nnls_gram(X_hat: np.ndarray, W: np.ndarray, iters: int = 300) -> np.ndarray:
    """Geometry-only KEEP score: NNLS Gram-consistency support.

    Observe only x_hat and W (no x_input). Solve, per sample,
    ``min_{x>=0} ||G x - x_hat||^2`` with ``G = W W^T`` via projected gradient,
    and use the recovered support x* as the keep score. This is the principled
    "is this reconstruction consistent with some sparse non-negative input?"
    detector. (Empirically it loses to raw magnitude — reported, not hidden.)
    """
    G = W @ W.T
    L = float(np.linalg.eigvalsh(G)[-1]) + 1e-9
    out = np.zeros_like(X_hat)
    for i in range(X_hat.shape[0]):
        b = X_hat[i]
        x = np.zeros_like(b)
        for _ in range(iters):
            x = np.maximum(x - (G @ x - b) / L, 0.0)
        out[i] = x
    return out


# --------------------------------------------------------------------------- #
# turning a KEEP score into (elimination, retention) points                   #
# --------------------------------------------------------------------------- #

def _labels(X: np.ndarray, X_hat: np.ndarray, act: float = ACT):
    truly = X > act
    appears = X_hat > act
    is_false = appears & ~truly
    is_true = appears & truly
    return truly, is_false, is_true


def elim_retention(keep_mask: np.ndarray, X: np.ndarray, X_hat: np.ndarray,
                   act: float = ACT):
    """Given a boolean KEEP mask, return (elimination%, retention%).

    elimination = 1 - (surviving false activations / ungoverned false activations)
    retention   = kept true activations / all truly-present features
    """
    truly, is_false, _ = _labels(X, X_hat, act)
    Xg = np.where(keep_mask, X_hat, 0.0)
    appears_g = Xg > act
    false_ung = float(np.sum(is_false))
    false_g = float(np.sum(appears_g & ~truly))
    elim = 100.0 * (1.0 - false_g / max(false_ung, 1e-12))
    retention = 100.0 * float(np.sum(appears_g & truly)) / max(float(np.sum(truly)), 1.0)
    return elim, retention


def elim_retention_governed(Xg: np.ndarray, X: np.ndarray, X_hat: np.ndarray,
                            act: float = ACT):
    """(elim%, retention%) for a governed output that may REPLACE values.

    Use this for project()/route(), which overwrite entries with clipped signal
    rather than zeroing them — the keep-mask form of elim_retention would
    mis-measure them.
    """
    truly, is_false, _ = _labels(X, X_hat, act)
    appears_g = Xg > act
    false_ung = float(np.sum(is_false))
    false_g = float(np.sum(appears_g & ~truly))
    elim = 100.0 * (1.0 - false_g / max(false_ung, 1e-12))
    retention = 100.0 * float(np.sum(appears_g & truly)) / max(float(np.sum(truly)), 1.0)
    return elim, retention


def risk_coverage_curve(score: np.ndarray, X: np.ndarray, X_hat: np.ndarray,
                        act: float = ACT, n_points: int = 60):
    """Sweep the keep-threshold over `score`; return sorted (elim, retention) points.

    Only entries that *appear* (x_hat > act) can be suppressed to change the
    metrics, so the sweep is over their score quantiles.
    """
    appears = X_hat > act
    vals = score[appears]
    if vals.size == 0:
        return [(0.0, 100.0)]
    qs = np.quantile(vals, np.linspace(0.0, 1.0, n_points))
    pts = []
    for thr in np.unique(qs):
        keep = ~appears | (score >= thr)   # suppress appearing entries below thr
        pts.append(elim_retention(keep, X, X_hat, act))
    pts.append((0.0, elim_retention(np.ones_like(X_hat, bool), X, X_hat, act)[1]))
    pts = sorted(set(pts))
    return pts


def retention_at_elim(pts, target_elim: float) -> float:
    """Best retention among curve points with elimination >= target."""
    ok = [r for (e, r) in pts if e >= target_elim - 1e-9]
    return max(ok) if ok else float("nan")


def frontier(pts):
    """Achievable upper envelope: (elim, best retention at elimination >= elim).

    A non-increasing curve suitable for plotting — avoids the spurious diagonal
    you get by connecting raw swept points in sorted order (retention is not a
    single-valued function of elimination during a threshold sweep).
    """
    es = sorted({e for e, _ in pts})
    return [(e, max(r for (ee, r) in pts if ee >= e - 1e-9)) for e in es]
