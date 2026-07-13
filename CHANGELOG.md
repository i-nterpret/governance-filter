# Changelog

## v0.3.0 — interference gate, risk-coverage study, dissent route

A rewrite that retires an overclaimed framing and keeps the real, small result.

### Retired (the overclaim)
- The "**three-substrate architecture for hallucination elimination**" framing and
  the "**eliminates 100% of false activations**" headline. The gate's score
  consumes `x_input`, and a false activation is *defined* by `x_input <= threshold`,
  so the score reads a monotone function of the label — a trivial label rule
  dominates it, and at matched elimination the weight geometry does not beat a
  plain activation-magnitude threshold. The headline did not survive its own
  benchmark. It is now framed honestly as a **negative result**.
- The unification claim ("same governance equation at three scales"). The three
  pieces were unrelated mechanisms.

### Renamed
- `GovernanceFilter` → `InterferenceGate` (deprecated alias kept for one release).
  The generic "governance" umbrella is dropped in favour of the specific operation.

### Moved to `examples/`
- `routing.py` → `examples/typed_routing_demo.py` (a fixed rule table; a
  confidence-gated cascade / learning-to-defer pattern, not a co-equal substrate).
- `chamber.py` → `examples/selective_prediction_demo.py`; functions renamed to
  their field-native names (`one_minus_msp` = max-softmax-prob; `softmax_margin` =
  top1−top2; `should_abstain`). The scattering-amplitude / "chamber" gloss over a
  softmax margin was cut.

### Added
- `baselines.py` — the comparators the benchmark needs to be falsifiable:
  trivial-oracle (the label; a ceiling), magnitude `|x_hat|`, random (a floor),
  and a geometry-only NNLS Gram-consistency detector; plus `risk_coverage_curve`
  and `frontier`.
- `benchmark.py` reframed around the **risk-coverage curve** (retention vs
  elimination), with a generated figure (`docs/risk_coverage.png`). The old
  gameable "elimination %" headline is gone.
- `InterferenceGate.route()` — the **dissent 2nd-wire**: `x_hat = governed + dissent`.
  The interference-dominated residual is returned for audit rather than discarded.
  (The identity is bookkeeping — `dissent` is defined as the remainder; the point
  is returning it, which `filter`/`project` do not.)
- `stable_rank()` diagnostic; an `n_features == 1` guard (was an `IndexError`).
- Tests: finite-difference gradient check, the n=1 crash, `project`'s
  **non-dominance** over `filter` (a `||w||^2 > 1` counterexample), and the route
  decomposition.

### Fixed
- `project()`'s docstring claimed it "strictly dominates" `filter()`. It does not:
  parity of elimination holds only when `margin * ||w_k||^2 * x_input_k` stays
  below the activation threshold (i.e. unit-scale weights, `margin = 1`).
- README's "weight-matrix Gram-structure" wording — the gate uses only the
  diagonal `||w_k||^2`; the full Gram is diagnostics-only.
- `stability_index (λ₂/λ₁)³` → `spectral_gap_ratio (λ₂/λ₁)` (the cube was a
  monotone cosmetic). Its README "Substrate-3" monotonicity claim is withdrawn:
  the trend is monotone only for random Gaussian W (a Wishart-concentration fact),
  not for trained weights.
- Synthesis DOI is `10.5281/zenodo.20292282`.

### Honest scope (unchanged, now stated)
This is a **reconstruction auditor given the input**, not an inference-time
hallucination detector. At real inference the intended feature vector is not
available as an answer key.
