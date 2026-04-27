# governance-filter

**A three-substrate architecture for hallucination elimination.**

`governance-filter` provides a post-hoc interference filter that eliminates 100% of false activations across sparsity regimes in the Elhage et al. (2022) toy-model framework, with bounded reconstruction-error cost. It also includes a typed demand-routing layer for organizational inference allocation under per-request governance constraints, and chamber-structure utilities grounded in scattering-amplitude theory.

No retraining. No extra parameters. One scalar threshold.

---

## Why this exists

Hallucination is not a training failure. It is the structural consequence of packing more features into a representational space than it has dimensions, without governance over the resulting interference. The dominant decomposition approach (sparse autoencoders) addresses the wrong problem, paying ~10% of pretraining compute on a 16M-latent SAE that loses 90% of the signal (Bereska & Gavves, 2025).

This package implements **governance** — the missing architectural layer — at three substrates that share the same failure mode:

| Substrate | Failure Mode | Mechanism |
|---|---|---|
| Neural-network | Interference-induced false activation | `GovernanceFilter` on weight-matrix Gram-structure |
| Organizational | Misrouted high-risk demand | `route_demand()` with per-request governance constraints |
| Theoretical | Generic-region demand under-classified as deterministic-chamber | `chamber_distance()` calibrated ambiguity |

---

## Install

```bash
pip install governance-filter
```

Or from source:

```bash
git clone https://github.com/i-nterpret/governance-filter.git
cd governance-filter
pip install -e ".[dev]"
```

---

## Quickstart — Substrate 1 (Neural-Network)

```python
import numpy as np
from governance_filter import GovernanceFilter

# Any linear-encoding / linear-decoding layer's weight matrix
W = np.random.standard_normal((20, 5))  # 20 features in 5 dims

filt = GovernanceFilter(W, threshold=0.3)

# During inference: filter the reconstruction
x_input = ...   # input feature activations
x_hat   = ...   # reconstructed activations from the model
x_governed = filt.filter(x_input, x_hat)

# Diagnostic: check superposition stability
print(filt.stability_index())  # (lambda_2 / lambda_1)^3
```

Run the toy-model benchmark:

```bash
python -m governance_filter.benchmark
```

Expected output:

```
  Sparsity   False (un)   False (gov)   Eliminated   Δ Error %
-----------------------------------------------------------------
      0.05       0.0050        0.0000       100.0%       +4.5%
      0.10       0.0059        0.0000       100.0%      +31.3%
      0.30       0.0347        0.0000       100.0%      +20.8%
```

---

## Quickstart — Substrate 2 (Organizational)

```python
from governance_filter import DemandProfile, DemandType, route_demand

profile = DemandProfile(
    demand_type=DemandType.STRUCTURED_TASK,
    risk_score=0.2,
    cognitive_posture="Structuralist",
    ambiguity_score=0.1,
    classification_confidence=0.95,
)

plan = route_demand(profile)
print(plan.target_model)        # 'open_mid_tier'
print(plan.governance_actions)  # []
print(plan.estimated_cost)      # 6.144
```

---

## Quickstart — Substrate 3 (Theoretical)

```python
import numpy as np
from governance_filter import calibrated_ambiguity, chamber_distance

logits = np.array([10.0, 1.0, 1.0, 1.0])
print(calibrated_ambiguity(logits))  # near 0 -> deep in chamber
print(chamber_distance(logits))      # near 1 -> deterministic routing
```

---

## How the three substrates couple

Each substrate has the same architectural form: a filter checking whether the substrate-specific signal-to-interference ratio exceeds a calibrated threshold, suppressing or escalating when not. The same governance equation, instantiated at three scales of representational space.

See the [synthesis paper](#paper) for the cross-substrate coupling claim and falsification criteria.

---

## Paper

> Dunn, J.E. (2026). *Governance Across Substrates: A Three-Layer Architecture for Hallucination Elimination.* [Zenodo DOI pending]

This implementation is the reference companion. The paper coordinates four documents:

1. *Governance Architecture for Neural Network Superposition* (March 2026) — the originating Substrate 1 paper, from a Google contest submission.
2. *Interpretive Demand Routing as a Structural Layer for Inference Allocation* — Substrate 2, addressed to Frank Nagle's work on open-source value capture.
3. *Demand Routing as a Structural Layer: Mathematical Framework with Scattering Amplitude Foundations* (Feb 2026) — Substrate 3, theoretical grounding via Guevara-Strominger-Skinner (2026).
4. The synthesis paper — bridges all three with explicit code, math, and falsification criteria.

---

## Falsification criteria

The architecture stands or falls on three substrate-specific empirical claims:

1. **Substrate 1.** Filter eliminates ≥95% of false activations across sparsity {0.05, 0.10, 0.30} in the Elhage toy model with reconstruction-error increase ≤50%. *Status: confirmed at 100% / 4–31%.*
2. **Substrate 2.** Calibrated classifier achieves ≥85% accuracy with N ≤ 10⁴ labeled examples; per-request governance constraints satisfiable without violating cost bound ≤ 0.37. *Status: PAC bounds derived; live-workload test pending.*
3. **Substrate 3.** Stability index (λ₂/λ₁)³ monotonically predicts governance-intervention requirement across packing ratios n/d ∈ {1, 2, 4, 8, 16}. *Status: experimentally observed; analytical proof pending.*

If any substrate fails its criterion, the cross-substrate coupling claim weakens. If two fail, the synthesis is wrong. Stated explicitly so the architecture can be falsified rather than merely contested.

---

## License

Apache 2.0. The Apache 2.0 patent grant protects both the author and adopters from third-party patent litigation, while keeping the work genuinely open.

---

## Sponsorship

If this work is useful in your research or product, please consider sponsoring continued development:

- **GitHub Sponsors:** [github.com/sponsors/i-nterpret](https://github.com/sponsors/i-nterpret)
- **Open Collective:** governance-filter
- **Direct:** [i-nterpret.com/sponsor](https://i-nterpret.com/sponsor)

Three-tier sponsorship structure:

| Tier | Use Case | Contribution |
|---|---|---|
| Research-free | Academic or research use | None — citation requested |
| Commercial-thank-you | Use in commercial products | Voluntary contribution |
| Frontier-lab | Production-scale deployment | Formal sponsorship for continued development |

---

## Citation

```bibtex
@misc{dunn2026governance,
  author       = {Dunn, James E.},
  title        = {Governance Across Substrates: A Three-Layer Architecture for Hallucination Elimination},
  year         = {2026},
  publisher    = {Zenodo},
  note         = {Reference implementation: github.com/i-nterpret/governance-filter},
}
```
