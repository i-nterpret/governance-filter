"""
Substrate 2 — Organizational Layer

Typed demand routing under per-request governance constraints. Implements
the demand-side analogue of governor-style scheduling: heterogeneous requests
become typed objects that can be routed and audited rather than treated as
a flat stream.

Reference: Dunn (2026), "Interpretive Demand Routing as a Structural Layer
for Inference Allocation" (note for Frank Nagle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DemandType(Enum):
    EXPLORATION = "exploration"
    STRUCTURED_TASK = "structured_task"
    COMPLIANCE_CRITICAL = "compliance_critical"
    AUTOMATION_ROUTINE = "automation_routine"


@dataclass
class DemandProfile:
    """Typed profile extracted from a raw demand event.

    Attributes
    ----------
    demand_type : DemandType
        Categorical demand type from C = {Exploration, StructuredTask,
        ComplianceCritical, AutomationRoutine}.
    risk_score : float
        Downstream-impact risk in [0, 1] (regulated domain, PII, irreversibility).
    cognitive_posture : str
        Required posture (Structuralist, Explorer, Steward, etc.) per
        HLRP #155 posture system.
    ambiguity_score : float
        alpha(d) = 1 - max p_theta(t | x, c), in [0, 1]. From a calibrated
        classifier; see chamber.calibrated_ambiguity().
    classification_confidence : float
        Calibrated max-probability of the classifier output, in [0, 1].
    """

    demand_type: DemandType
    risk_score: float
    cognitive_posture: str
    ambiguity_score: float
    classification_confidence: float


@dataclass
class RoutingPlan:
    """Execution specification produced by the routing policy."""

    target_model: str
    governance_actions: list[str] = field(default_factory=list)
    execution_params: dict = field(default_factory=dict)
    estimated_cost: float = 0.0


# Default per-token cost rates by tier (placeholder; override per deployment)
DEFAULT_RATES = {
    "closed_high_capacity": 0.030,
    "open_mid_tier": 0.003,
    "open_lightweight": 0.001,
}


def _estimate_cost(model: str, params: dict, rates: dict | None = None) -> float:
    """Tier-pegged cost estimate per output token."""
    rate_table = rates if rates is not None else DEFAULT_RATES
    return rate_table.get(model, 0.030) * params.get("max_tokens", 1024)


def route_demand(
    profile: DemandProfile,
    risk_threshold: float = 0.7,
    confidence_threshold: float = 0.7,
    rates: dict | None = None,
) -> RoutingPlan:
    """Constrained-optimal routing under per-request governance coverage.

    Two governance overrides take precedence over default routes:

    1. risk_score > risk_threshold
       -> closed model + human gate + audit trail
    2. classification_confidence < confidence_threshold
       -> closed model + classification review

    Otherwise, route by demand type per the policy in Section 2 of the
    demand-routing framework.
    """
    plan = RoutingPlan(target_model="")

    if profile.risk_score > risk_threshold:
        plan.target_model = "closed_high_capacity"
        plan.governance_actions = ["human_gate", "audit_trail"]
        plan.execution_params = {"temperature": 0.0, "max_tokens": 2048}

    elif profile.classification_confidence < confidence_threshold:
        plan.target_model = "closed_high_capacity"
        plan.governance_actions = ["classification_review"]
        plan.execution_params = {"temperature": 0.3, "max_tokens": 2048}

    elif profile.demand_type == DemandType.EXPLORATION:
        plan.target_model = "closed_high_capacity"
        plan.execution_params = {"temperature": 0.7, "max_tokens": 4096}

    elif profile.demand_type == DemandType.STRUCTURED_TASK:
        plan.target_model = "open_mid_tier"
        plan.execution_params = {"temperature": 0.0, "max_tokens": 2048}

    elif profile.demand_type == DemandType.COMPLIANCE_CRITICAL:
        plan.target_model = "closed_high_capacity"
        plan.governance_actions = [
            "human_gate",
            "audit_trail",
            "elevated_retention",
        ]
        plan.execution_params = {"temperature": 0.0, "max_tokens": 2048}

    elif profile.demand_type == DemandType.AUTOMATION_ROUTINE:
        plan.target_model = "open_lightweight"
        plan.governance_actions = ["drift_monitoring"]
        plan.execution_params = {"temperature": 0.0, "max_tokens": 1024}

    else:
        raise ValueError(f"Unhandled demand_type: {profile.demand_type}")

    plan.estimated_cost = _estimate_cost(plan.target_model, plan.execution_params, rates)
    return plan


def expected_cost_ratio(alpha: float, r: float) -> float:
    """Conditional cost bound from the demand-routing framework.

    E[C_pi*] / E[C_baseline] <= alpha * r + (1 - alpha)

    where alpha = P[d in structured U automation] is the structured-demand
    mass and r = E[C_open | struct] / E[C_closed] is the open-vs-closed cost
    ratio on the structured subset.

    With alpha = 0.7 (Nagle-Yue empirical finding) and r = 0.1, this gives
    cost_ratio <= 0.37, i.e., a 63% cost reduction at the demand-routing
    layer alone.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"r must be in [0, 1]; got {r}")
    return alpha * r + (1.0 - alpha)
