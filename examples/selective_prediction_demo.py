"""
Example (demoted from the v1 "Substrate 3" core): standard softmax-confidence
measures for a selective-prediction / abstention gate.

These are ordinary, named quantities — no scattering-amplitude or "chamber"
grounding is needed and none is claimed:
  - one_minus_msp   = 1 - max softmax prob        (Hendrycks & Gimpel 2017, MSP)
  - softmax_margin  = top1 - top2 softmax prob    (best-vs-second-best / margin;
                                                    Scheffer 2001; Joshi 2009)
  - should_abstain  = margin below a threshold    (a reject-option gate)

Kept as a demo because the *pattern* (a scalar confidence + a threshold + a
reject decision) is the same shape the interference gate uses on neural
features and the scheduler uses on graph classes — a companion pattern, not a
shared equation.
"""
from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def one_minus_msp(logits: np.ndarray):
    """1 - maximum softmax probability (Hendrycks & Gimpel 2017). In [0, 1]."""
    p = _softmax(logits)
    m = np.max(p, axis=-1)
    return float(1.0 - m) if np.ndim(m) == 0 else 1.0 - m


def softmax_margin(logits: np.ndarray):
    """top1 - top2 softmax probability (best-vs-second-best margin). In [0, 1]."""
    p = np.sort(_softmax(logits), axis=-1)
    d = p[..., -1] - p[..., -2]
    return float(d) if np.ndim(d) == 0 else d


def should_abstain(logits: np.ndarray, margin_threshold: float = 0.4) -> bool:
    """Reject-option gate: abstain when the top-two margin is too small."""
    return float(softmax_margin(logits)) < margin_threshold


if __name__ == "__main__":
    confident = np.array([10.0, 1.0, 1.0, 1.0])
    ambiguous = np.array([3.0, 2.8, 1.0, 1.0])
    for name, lg in [("confident", confident), ("ambiguous", ambiguous)]:
        print(f"{name:>10}: 1-MSP={one_minus_msp(lg):.3f}  "
              f"margin={softmax_margin(lg):.3f}  abstain={should_abstain(lg)}")
