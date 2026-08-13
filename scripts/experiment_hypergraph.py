"""Hypergraph physics: evolution, dimension profiling, and rule search.

Demonstrates the pieces in `socrates.hypergraph` together on small, fast
examples. Every number this script prints has a corresponding known-answer
test in `tests/test_hypergraph.py` -- this script is a demonstration, not
the source of truth for correctness.

Run:  python scripts/experiment_hypergraph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socrates.hypergraph import (  # noqa: E402
    Hypergraph,
    RewriteRule,
    evolve,
    mean_dimension,
    multiway_evolve,
    rule_space,
    solve_for_rule,
)


def demo_evolution_and_dimension() -> None:
    print("\n[1] Evolution and emergent dimension")
    print("-" * 60)

    chain_rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)), name="chain-extension")
    seed = Hypergraph.of((0, 1))
    history = evolve(seed, [chain_rule], steps=6, max_edges=2000)
    print(f"  chain-extension rule: {[h.num_edges for h in history]} edges per step")
    print("  (this rule applies to EVERY matchable edge each generation, so")
    print("   growth is exponential -- a binary-tree-like structure, not a path)")

    final = history[-1]
    dim = mean_dimension(final, samples=5, max_radius=3)
    print(f"  mean dimension of final state (n={final.num_nodes} nodes): {dim:.3f}")


def demo_multiway_branching() -> None:
    print("\n[2] Multiway system: how much does the rule branch?")
    print("-" * 60)

    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)))
    seed = Hypergraph.of((0, 1), (2, 3))
    mw = multiway_evolve(seed, [rule], steps=2, max_states=200)
    print(f"  states discovered: {mw.num_states}")
    print(f"  causal transitions: {len(mw.transitions)}")
    print(f"  mean branching factor: {mw.branching_factor():.3f}")
    print("  (Tier B: this is a raw branchial-graph statistic, not a")
    print("   cohomology or invariant -- see docs/HYPERGRAPH_NOTES.md)")


def demo_rule_search() -> None:
    print("\n[3] Poly-algebraic search: solve for a rule toward a target dimension")
    print("-" * 60)

    seed = Hypergraph.of((0, 1))

    for target in (1.0, 2.0):
        candidates = rule_space(rhs_edge_count=2, arity=2, num_pattern_vars=3)

        def score(hg: Hypergraph, target: float = target) -> float:
            d = mean_dimension(hg, samples=4, max_radius=3)
            return abs(d - target) if d == d else 10.0  # d == d is a NaN check

        results = solve_for_rule(seed, candidates, score, steps=4, top_k=3)
        print(f"\n  target dimension {target}:")
        for r in results:
            print(f"    rhs={r.rule.rhs}  score={r.score:.4f}  n_edges={r.final_state.num_edges}")


if __name__ == "__main__":
    demo_evolution_and_dimension()
    demo_multiway_branching()
    demo_rule_search()
    print("\nDone. See docs/HYPERGRAPH_NOTES.md for the tier triage of what")
    print("this module does and does not establish.")
