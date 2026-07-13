"""
governance-filter — an interference gate for neural-network superposition.

A post-hoc selective-prediction filter (classification with a reject option)
over the false activations that superposition interference induces in a
tied-weight linear decode. The honest scoreboard is a risk-coverage curve
(see governance_filter.benchmark); the kept contribution is the
signal/interference decomposition and the dissent route, not a detector.

Substrate: Elhage et al. (2022), Toy Models of Superposition.

The v1 "three-substrate governance" framing has been retired: the demand-routing
and softmax-confidence utilities that used to ship in the core now live in
examples/ (typed_routing_demo.py, selective_prediction_demo.py). They were
separate mechanisms, not instances of one equation.
"""

from governance_filter.filter import InterferenceGate, RouteResult, GovernanceFilter

__version__ = "0.3.0"
__all__ = [
    "InterferenceGate",
    "RouteResult",
    "GovernanceFilter",  # deprecated alias for InterferenceGate
]
