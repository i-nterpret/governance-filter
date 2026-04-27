"""
Substrate 3 — Theoretical Layer

Chamber-structure utilities. Provides calibrated ambiguity scoring and
chamber-distance metrics that feed into the routing policy (Substrate 2)
and explain the stability index (Substrate 1).

Reference: Dunn (Feb 2026), "Demand Routing as a Structural Layer for
Inference Allocation: Mathematical Framework with Scattering Amplitude
Foundations." Theoretical grounding from Guevara, Strominger, Skinner (2026).
"""
from __future__ import annotations

import numpy as np


def calibrated_ambiguity(classifier_logits: np.ndarray) -> float:
    """Operational ambiguity score from a calibrated classifier.

        alpha(d) = 1 - max_t p_theta(t | x, c)

    Calibration ensures P[t* = argmax | p_theta = p] ~ p, so that
    low-confidence predictions are actionable signals for governance
    escalation rather than uninterpretable noise.

    Parameters
    ----------
    classifier_logits : np.ndarray, shape (n_classes,) or (batch, n_classes)
        Pre-softmax logits.

    Returns
    -------
    alpha : float or np.ndarray
        Ambiguity score(s) in [0, 1].
    """
    z = classifier_logits - np.max(classifier_logits, axis=-1, keepdims=True)
    probs = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    max_probs = np.max(probs, axis=-1)
    if np.isscalar(max_probs) or max_probs.ndim == 0:
        return float(1.0 - max_probs)
    return 1.0 - max_probs


def chamber_distance(classifier_logits: np.ndarray) -> float:
    """Distance from the half-collinear chamber boundary.

    The half-collinear regime corresponds to structured demand: the region
    where complexity collapses to piecewise-constant decisions. This distance
    metric quantifies how far a given demand event lies from that chamber.

    Defined as the gap between the top two calibrated probabilities; small
    values indicate proximity to the chamber boundary (ambiguous region).
    Mirrors the eigenvalue-ratio gap that defines the stability index in
    Substrate 1.

    Returns
    -------
    distance : float in [0, 1]
        1.0 = deep in chamber (deterministic routing); 0.0 = on boundary.
    """
    z = classifier_logits - np.max(classifier_logits, axis=-1, keepdims=True)
    probs = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    sorted_probs = np.sort(probs, axis=-1)
    if sorted_probs.ndim == 1:
        return float(sorted_probs[-1] - sorted_probs[-2])
    return sorted_probs[..., -1] - sorted_probs[..., -2]


def in_chamber(classifier_logits: np.ndarray, threshold: float = 0.4) -> bool:
    """Boolean: is this demand event in the deterministic-routing chamber?

    Routes that satisfy in_chamber(...) = True can be handled by the default
    policy in routing.route_demand. Routes that fail must be escalated.
    """
    return float(chamber_distance(classifier_logits)) >= threshold
