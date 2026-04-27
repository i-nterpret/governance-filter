"""
governance-filter — A three-substrate architecture for hallucination elimination.

Substrate 1 (Neural-Network):     filter.GovernanceFilter
Substrate 2 (Organizational):     routing.route_demand, DemandProfile, RoutingPlan
Substrate 3 (Theoretical):        chamber.calibrated_ambiguity, chamber_distance

Reference: Dunn (2026), "Governance Across Substrates: A Three-Layer Architecture
for Hallucination Elimination." Apache 2.0.
"""

from governance_filter.filter import GovernanceFilter
from governance_filter.routing import (
    DemandType,
    DemandProfile,
    RoutingPlan,
    route_demand,
)
from governance_filter.chamber import calibrated_ambiguity, chamber_distance

__version__ = "0.1.0"
__all__ = [
    "GovernanceFilter",
    "DemandType",
    "DemandProfile",
    "RoutingPlan",
    "route_demand",
    "calibrated_ambiguity",
    "chamber_distance",
]
