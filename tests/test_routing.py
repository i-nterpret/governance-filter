"""Smoke tests for typed demand routing."""
import numpy as np
import pytest

from governance_filter import (
    DemandProfile,
    DemandType,
    route_demand,
    calibrated_ambiguity,
    chamber_distance,
)
from governance_filter.routing import expected_cost_ratio


def test_high_risk_overrides_to_closed_with_gate():
    profile = DemandProfile(
        demand_type=DemandType.AUTOMATION_ROUTINE,
        risk_score=0.9,
        cognitive_posture="Steward",
        ambiguity_score=0.1,
        classification_confidence=0.95,
    )
    plan = route_demand(profile)
    assert plan.target_model == "closed_high_capacity"
    assert "human_gate" in plan.governance_actions
    assert "audit_trail" in plan.governance_actions


def test_low_confidence_triggers_review():
    profile = DemandProfile(
        demand_type=DemandType.STRUCTURED_TASK,
        risk_score=0.2,
        cognitive_posture="Structuralist",
        ambiguity_score=0.6,
        classification_confidence=0.5,
    )
    plan = route_demand(profile)
    assert "classification_review" in plan.governance_actions


def test_default_structured_routes_to_open():
    profile = DemandProfile(
        demand_type=DemandType.STRUCTURED_TASK,
        risk_score=0.1,
        cognitive_posture="Structuralist",
        ambiguity_score=0.05,
        classification_confidence=0.95,
    )
    plan = route_demand(profile)
    assert plan.target_model == "open_mid_tier"
    assert plan.governance_actions == []


def test_compliance_critical_always_closed_with_full_gov():
    profile = DemandProfile(
        demand_type=DemandType.COMPLIANCE_CRITICAL,
        risk_score=0.4,
        cognitive_posture="Steward",
        ambiguity_score=0.1,
        classification_confidence=0.95,
    )
    plan = route_demand(profile)
    assert plan.target_model == "closed_high_capacity"
    for action in ("human_gate", "audit_trail", "elevated_retention"):
        assert action in plan.governance_actions


def test_calibrated_ambiguity():
    confident = np.array([10.0, 1.0, 1.0, 1.0])
    uncertain = np.array([1.0, 1.0, 1.0, 1.0])
    assert calibrated_ambiguity(confident) < 0.05
    assert calibrated_ambiguity(uncertain) == pytest.approx(0.75, abs=1e-6)


def test_chamber_distance():
    deep = np.array([10.0, 1.0, 1.0, 1.0])
    boundary = np.array([1.0, 1.0, 0.0, 0.0])
    assert chamber_distance(deep) > 0.5
    assert chamber_distance(boundary) < 0.1


def test_expected_cost_ratio_nagle_yue():
    # Empirical anchor: alpha=0.7, r=0.1 -> cost_ratio <= 0.37
    ratio = expected_cost_ratio(alpha=0.7, r=0.1)
    assert ratio == pytest.approx(0.37, abs=1e-6)
