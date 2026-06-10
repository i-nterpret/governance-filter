"""
Substrate 1 — Neural-Network Layer

Post-hoc governance filter for superposition interference. Eliminates false
activations by checking signal-to-interference ratio against a configurable
threshold. Operates on the existing weight matrix of any linear-encoding /
linear-decoding layer.

Reference: Dunn (March 2026), "Governance Architecture for Neural Network
Superposition," extending Elhage et al. (2022) toy-model framework.

v2 (2026-06): adds project() alongside filter(). filter() suppresses (zeros)
over-threshold features; project() recovers the input-supported signal (P x)
for those features instead, dropping only the interference ((I - P) x) — same
false-activation elimination, with the true positives the suppressor discards
recovered. filter() is unchanged.
"""
from __future__ import annotations

import numpy as np


class GovernanceFilter:
    """Post-hoc interference filter for neural-network superposition.

    Parameters
    ----------
    W : np.ndarray, shape (n_features, n_dim)
        Weight matrix of the linear encoding/decoding layer.
    threshold : float, default 0.3
        Suppress feature k if |x_hat_k - signal_k| / |x_hat_k| > threshold.
    eps : float, default 1e-8
        Numerical stability constant for the ratio denominator.

    Notes
    -----
    Decomposes each output feature into signal and interference:

        signal_k       = x_input_k * ||w_k||^2
        interference_k = sum_{j != k} x_input_j * <w_j, w_k>
        ratio_k        = |x_hat_k - signal_k| / (|x_hat_k| + eps)

    Suppresses features whose interference dominates. Requires no retraining,
    no extra parameters, and one scalar threshold.
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
        self.gram = W @ W.T

    def filter(self, x_input: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Suppress features whose interference ratio exceeds threshold.

        Parameters
        ----------
        x_input : np.ndarray, shape (n_features,) or (batch, n_features)
            Input feature activations.
        x_hat : np.ndarray, same shape as x_input
            Reconstructed activations after the decode pass.

        Returns
        -------
        x_governed : np.ndarray, same shape as x_hat
            Filtered output with interference-dominated features zeroed.
        """
        if x_input.shape != x_hat.shape:
            raise ValueError(
                f"x_input and x_hat must have matching shape; "
                f"got {x_input.shape} and {x_hat.shape}"
            )

        signal = x_input * self.feature_norms_sq
        ratio = np.abs(x_hat - signal) / (np.abs(x_hat) + self.eps)
        suppress_mask = ratio > self.threshold
        x_governed = x_hat.copy()
        x_governed[suppress_mask] = 0.0
        return x_governed

    def project(self, x_input: np.ndarray, x_hat: np.ndarray,
                margin: float = 1.0) -> np.ndarray:
        """Give-and-take projection: recover the aligned signal instead of zeroing.

        For a feature whose interference ratio exceeds threshold, recover the
        input-supported signal (P x) rather than zeroing it; the interference
        component ((I - P) x) is dropped.

          - false activation (x_input_k ~ 0): signal_k ~ 0  -> still eliminated
          - true positive    (x_input_k > 0): signal_k > 0  -> RECOVERED

        Elimination matches filter(); the true positives filter() discards are kept.

        Parameters
        ----------
        x_input : np.ndarray, shape (n_features,) or (batch, n_features)
            Input feature activations.
        x_hat : np.ndarray, same shape as x_input
            Reconstructed activations after the decode pass.
        margin : float, default 1.0
            Scales the recovered signal (1.0 = full aligned recovery). The
            projection recovery fraction; distinct from any dissent-recycle margin.

        Returns
        -------
        x_governed : np.ndarray, same shape as x_hat
            Output with interference-dominated features replaced by their aligned
            (input-supported) component rather than zeroed.
        """
        if x_input.shape != x_hat.shape:
            raise ValueError(
                f"x_input and x_hat must have matching shape; "
                f"got {x_input.shape} and {x_hat.shape}"
            )

        signal = x_input * self.feature_norms_sq
        ratio = np.abs(x_hat - signal) / (np.abs(x_hat) + self.eps)
        over = ratio > self.threshold
        x_governed = x_hat.copy()
        # Recover the aligned signal, but CLIP so it can never exceed what was
        # actually reconstructed. Without the clip, signal = x_input * ||w||^2
        # amplifies by the weight norm on trained (non-unit) weights and can push
        # a sub-threshold input over the activation line -> a spurious false
        # activation. Clipping makes project() strictly dominate filter():
        # identical false-activation elimination, equal-or-higher TP retention.
        x_governed[over] = np.clip(margin * signal[over], 0.0, np.abs(x_hat[over]))
        return x_governed

    def stability_index(self) -> float:
        """Return (lambda_2 / lambda_1)^3, the superposition-stability index.

        Low values indicate strong feature contraction (governable). Values
        approaching 1 indicate features collapsing into indistinguishable
        mush; governance becomes architecturally required.
        """
        eigenvalues = np.linalg.eigvalsh(self.gram)
        eigenvalues = np.sort(eigenvalues)[::-1]
        if eigenvalues[0] <= 0:
            return float("nan")
        ratio = eigenvalues[1] / eigenvalues[0]
        return float(ratio ** 3)

    def interference_profile(self, k: int) -> np.ndarray:
        """Return the interference vector for feature k.

        Useful for diagnostic visualization of which features share dimensions
        with feature k.
        """
        if not 0 <= k < self.W.shape[0]:
            raise IndexError(f"feature index {k} out of range [0, {self.W.shape[0]})")
        return self.gram[k].copy()

    def __repr__(self) -> str:
        n, d = self.W.shape
        return (
            f"GovernanceFilter(n_features={n}, n_dim={d}, "
            f"threshold={self.threshold}, sigma={self.stability_index():.3f})"
        )
