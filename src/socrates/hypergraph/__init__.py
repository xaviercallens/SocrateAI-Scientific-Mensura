"""Hypergraph rewriting and emergent-space tools (Tier C physics, Tier B code).

Exploratory infrastructure for testing Wolfram-Model-style hypothesis --
that space, time, and physics emerge from discrete rewriting of a
pre-geometric hypergraph -- against actual computation. See
docs/HYPERGRAPH_NOTES.md for the tier triage: what is proven (the algorithms
do what they say), what is measured (dimension, branching), and what remains
speculative interpretation, never load-bearing.
"""

from .core import Hypergraph, ball
from .dimension import DimensionEstimate, dimension_profile, local_dimension, mean_dimension
from .pointcloud import knn_hypergraph
from .polyalgebra import RuleSearchResult, rule_space, solve_for_rule
from .rewriting import (
    MultiwaySystem,
    RewriteRule,
    apply_at,
    evolve,
    find_matches,
    multiway_evolve,
)

__all__ = [
    "DimensionEstimate",
    "Hypergraph",
    "MultiwaySystem",
    "RewriteRule",
    "RuleSearchResult",
    "apply_at",
    "ball",
    "dimension_profile",
    "evolve",
    "find_matches",
    "knn_hypergraph",
    "local_dimension",
    "mean_dimension",
    "multiway_evolve",
    "rule_space",
    "solve_for_rule",
]
