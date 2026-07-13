"""
Interference gate for neural-network superposition (v3).

A post-hoc *selective-prediction* filter — classification with a reject option —
over the false activations that superposition interference induces in a
tied-weight linear decode. Given the input feature vector ``x_input`` and the
reconstruction ``x_hat``, it scores each output feature by how much of its
reconstructed value is NOT explained by that feature's own input-supported term,
and routes the interference-dominated features to a reject/dissent channel.

Field placement (read this before the claims):
  - The task is *selective prediction* / classification-with-a-reject-option
    (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017). The honest scoreboard is
    a **risk-coverage curve**, not a single "elimination %" — see benchmark.py.
  - The substrate is the Elhage et al. (2022) toy model of superposition.
  - HONEST SCOPE: the score consumes ``x_input``. In the benchmark a false
    activation is *defined* by ``x_input_k <= activation_threshold``, so the
    score reads a monotone function of the label. This gate is therefore a
    **reconstruction auditor given the input**, NOT an inference-time
    hallucination detector (at inference the intended feature vector is not
    available). Benchmarked honestly against baselines that share that access,
    the weight geometry adds ~nothing over raw activation magnitude — a negative
    result the benchmark reports plainly. What is kept here is the *decomposition*
    (supported signal vs. interference residual) and the **route** that preserves
    the residual instead of discarding it.

Reference: Dunn (2026), extending Elhage et al. (2022). Apache 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RouteResult:
    """Conservative decomposition of a reconstruction into two wires.

    ``governed + dissent == x_hat`` exactly (the gate conserves signal — it
    relocates the interference-dominated part rather than deleting it).

    Attributes
    ----------
    governed : np.ndarray
        The asserted output: input-supported (clipped) signal on routed
        features, the untouched reconstruction elsewhere.
    dissent : np.ndarray
        The reject channel: the interference-dominated residual removed from
        the asserted output on routed features, zero elsewhere.
    routed : np.ndarray
        Boolean mask of the features whose interference ratio exceeded the
        threshold (the ones sent to the dissent wire).
    """
    governed: np.ndarray
    dissent: np.ndarray
    routed: np.ndarray


class InterferenceGate:
    """Reject-option filter for neural-network superposition interference.

    Parameters
    ----------
    W : np.ndarray, shape (n_features, n_dim)
        Weight matrix of the linear encode/decode layer.
    threshold : float, default 0.3
        Route feature k to reject/dissent if
        ``|x_hat_k - signal_k| / |x_hat_k| > threshold``. This is a single
        scalar operating point on the risk-coverage curve — sweep it to trace
        the curve; there is nothing special about 0.3.
    eps : float, default 1e-8
        Numerical stability constant for the ratio denominator.

    Notes
    -----
    Per output feature k, decompose the reconstruction into an input-supported
    diagonal term and the interference residual::

        signal_k       = x_input_k * ||w_k||^2          # diagonal (supported)
        interference_k  = sum_{j != k} x_input_j * <w_j, w_k>
        ratio_k         = |x_hat_k - signal_k| / (|x_hat_k| + eps)

    Requires no retraining and a single scalar threshold. The full Gram matrix
    ``W @ W.T`` is exposed only for diagnostics (``spectral_gap_ratio``,
    ``interference_profile``); the gate itself uses only the diagonal
    ``||w_k||^2`` — so this is a norm-scaled reconstruction-residual gate, not a
    pairwise-Gram operator.
    """

    def __init__(self, W: np.ndarray, threshold: float = 0.3, eps: float = 1e-8):
        if W.ndim != 2:
            raise ValueError(f"W must be 2D (n_features, n_dim); got shape {W.shape}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {threshold}")

        self.W = W
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.feature_norms_sq = np.sum(W ** 2, axis=1)
        self.gram = W @ W.T  # diagnostics only; the gate uses the diagonal

    # -- scoring -------------------------------------------------------------

    def _ratio(self, x_input: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        if x_input.shape != x_hat.shape:
            raise ValueError(
                f"x_input and x_hat must have matching shape; "
                f"got {x_input.shape} and {x_hat.shape}"
            )
        signal = x_input * self.feature_norms_sq
        return np.abs(x_hat - signal) / (np.abs(x_hat) + self.eps)

    def score(self, x_input: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Return the per-feature interference ratio (the reject-option score)."""
        return self._ratio(x_input, x_hat)

    # -- the three operating modes ------------------------------------------

    def filter(self, x_input: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Suppress: zero the features whose interference ratio exceeds threshold."""
        ratio = self._ratio(x_input, x_hat)
        x_governed = x_hat.copy()
        x_governed[ratio > self.threshold] = 0.0
        return x_governed

    def project(self, x_input: np.ndarray, x_hat: np.ndarray,
                margin: float = 1.0) -> np.ndarray:
        """Project: replace a routed feature with its input-supported (clipped) signal.

        For a routed feature, recover ``clip(margin * x_input_k * ||w_k||^2, 0,
        |x_hat_k|)`` instead of zeroing. The clip stops the recovered value from
        exceeding what was actually reconstructed.

        NOTE ON ELIMINATION PARITY (corrected from v2): ``project`` matches
        ``filter``'s false-activation elimination **only when
        ``margin * ||w_k||^2 * x_input_k`` stays below the activation threshold
        for every truly-absent feature** — i.e. at unit-scale weights
        (``||w_k||^2 ~ 1``) with ``margin = 1``. With ``||w_k||^2 > 1`` a small
        but nonzero absent input can be amplified above the activation line, so
        ``project`` can KEEP a false activation that ``filter`` removes. It does
        not "strictly dominate" ``filter`` in general.
        """
        ratio = self._ratio(x_input, x_hat)
        over = ratio > self.threshold
        signal = x_input * self.feature_norms_sq
        x_governed = x_hat.copy()
        x_governed[over] = np.clip(margin * signal[over], 0.0, np.abs(x_hat[over]))
        return x_governed

    def route(self, x_input: np.ndarray, x_hat: np.ndarray,
              margin: float = 1.0) -> RouteResult:
        """Route: the dissent 2nd-wire. Decompose x_hat = governed + dissent.

        Same governed output as ``project`` on routed features, but the removed
        interference residual is returned on a second wire instead of discarded.
        ``governed + dissent == x_hat`` exactly (signal is conserved, not
        deleted). The dissent wire is the auditable reject channel — the
        features the gate would not assert. Nothing here estimates a "missing
        mass"; it is a book-keeping split of the reconstruction.
        """
        ratio = self._ratio(x_input, x_hat)
        over = ratio > self.threshold
        signal = x_input * self.feature_norms_sq
        governed = x_hat.copy()
        governed[over] = np.clip(margin * signal[over], 0.0, np.abs(x_hat[over]))
        dissent = x_hat - governed          # nonzero only on routed features
        return RouteResult(governed=governed, dissent=dissent, routed=over)

    # -- diagnostics (Gram-based; NOT used by the gate) ----------------------

    def spectral_gap_ratio(self) -> float:
        """Return lambda_2 / lambda_1 of the Gram matrix (in [0, 1]).

        A diagnostic of how contracted the feature spectrum is. Replaces v2's
        ``(lambda_2/lambda_1)**3`` — the cube was a monotone cosmetic with no
        derivation and changed no ordering.

        CAVEAT (verified 2026-07): this rises monotonically with packing ratio
        n/d only for RANDOM Gaussian W (a Marchenko-Pastur / Wishart
        concentration fact); on TRAINED toy-model weights it shows no monotone
        trend. Do NOT read it as a predictor of "how much governance is needed".
        """
        if self.W.shape[0] < 2:
            return 0.0
        ev = np.sort(np.linalg.eigvalsh(self.gram))[::-1]
        if ev[0] <= 0:
            return 0.0
        return float(ev[1] / ev[0])

    def stable_rank(self) -> float:
        """Return the stable rank ||W||_F^2 / ||W||_2^2 (effective dimensionality).

        A better-behaved spectral summary than the top-two-eigenvalue ratio: it
        uses the whole spectrum, so it distinguishes [1,.5,.5,.5,.5] from
        [1,.5,.01,.01,.01] (both of which collapse to the same gap ratio).
        """
        ev = np.linalg.eigvalsh(self.gram)
        top = ev[-1]
        if top <= 0:
            return 0.0
        return float(np.sum(ev) / top)

    def interference_profile(self, k: int) -> np.ndarray:
        """Return the Gram row for feature k (which features share its dimensions)."""
        if not 0 <= k < self.W.shape[0]:
            raise IndexError(f"feature index {k} out of range [0, {self.W.shape[0]})")
        return self.gram[k].copy()

    def __repr__(self) -> str:
        n, d = self.W.shape
        return (
            f"InterferenceGate(n_features={n}, n_dim={d}, "
            f"threshold={self.threshold}, gap_ratio={self.spectral_gap_ratio():.3f})"
        )


# Deprecated alias — the "governance" umbrella was retired (the corpus audit
# rejected GOVERN/ROUTE as non-primitives). Kept for one release; use
# InterferenceGate.
GovernanceFilter = InterferenceGate
