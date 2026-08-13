# Hypergraph physics: tier triage

`src/socrates/hypergraph/` implements infrastructure for testing the
Wolfram Physics Project's hypothesis computationally: that space, time, and
known physics emerge from discrete local rewriting of a pre-geometric
hypergraph. This document applies the project's own epistemic discipline
(Tier A: Lean kernel-checked, Tier B: exact computation / algorithms that do
exactly what they claim, Tier C: physical interpretation, never load-bearing)
to both the module and to the four brainstormed formalization concepts that
motivated it.

**The design rule inherited from the rest of this repository applies
unchanged: nothing here is permitted to assert more than its own check
proves.** A dimension estimate is a statistic of a graph. A rule search
result is a ranked list under an explicit scoring function. Neither is a
claim about the physical universe.

---

## What is implemented (Tier B)

| Piece | File | What it actually computes |
|---|---|---|
| Hypergraph representation | `core.py` | An immutable multiset of ordered hyperedges; adjacency graph; ball (graph-distance neighbourhood) |
| Rewriting engine | `rewriting.py` | Subhypergraph pattern matching (backtracking search over consistent pattern-variable assignments); rule application; single-history evolution applying all non-overlapping matches per generation; multiway (branching) evolution over *every* single-match update, deduplicated by canonical relabeling |
| Dimension estimator | `dimension.py` | Volume-growth dimension via a log-log fit of *shell size* (not cumulative volume — see below) vs radius |
| Rule solver | `polyalgebra.py` | Exhaustive enumeration of a bounded rewrite-rule search space, scored by an arbitrary caller-supplied function of the evolved hypergraph |

All four are covered by known-answer tests (`tests/test_hypergraph.py`), not
self-consistency checks: a path graph's dimension is verified to be exactly
1 (not approximately — the shell-based fit removes the offset bias a naive
cumulative-volume fit has), a 4-connected grid's is exactly 2, and the rule
solver's ranking is verified against hand-derived exact edge counts for
several small rule families, not against the module's own other outputs.

### A real numerical lesson, applied here too

The first version of `local_dimension` fit cumulative ball volume directly
and returned dimension ≈ 0.70–0.90 for a path graph at computationally
tractable radii — not because the code was wrong, but because
$|B(r)| = r+1$ for a path is *affine*, not homogeneous, so a log-log fit is
biased toward a lower exponent at any radius small enough to be practical
(the bias vanishes only asymptotically, as $r \to \infty$). Fitting on
*shell sizes* ($dV(r) = |B(r)| - |B(r-1)|$) instead removes the additive
offset entirely — a path's shell is the exact constant 2 (or 1 from an
endpoint), a grid's is exactly $4r$ — so the corrected estimator returns
dimensions of *exactly* 1.0 and 2.0 (not merely close) at small, practical
radii. This was caught by checking the estimator against known lattices
before trusting it on anything hypergraph-evolution produced, which is
precisely the discipline the rest of this repository has needed repeatedly
(see `docs/FINDINGS.md`).

---

## Triage of the four brainstormed concepts

### 1. Poly-Algebraic Calculus (higher-arity algebra) — Tier B implemented, Tier C interpretation

**What is real and implemented:** `rule_space` + `solve_for_rule` is a
literal instance of "solve for an unknown structural bond": a finite,
explicit search space of candidate rewrite rules, evolved and scored against
a caller-supplied target property. This is genuinely the load-bearing piece
— dimension estimation and any future branchial/rulial tooling all consume
a hypergraph or evolution history that only exists because a rule was
specified or found.

**What is not established:** that this is *the* correct generalization of
algebra to hypergraph physics, that it "formally extends category theory
into higher-arity domains," or that it can "map the strong nuclear force."
The search space in `rule_space` is a narrow, hand-chosen parametrization
(fixed-shape connected LHS, bounded arity, bounded pattern-variable pool) —
useful for small, tractable experiments, not a claim to have formalized the
space of all higher-arity rewrite systems. Subhypergraph matching is
NP-hard in general; `find_matches`'s `limit` parameter is a pragmatic cap,
not a completeness guarantee.

### 2. Branchial Cohomology — Tier C only, not attempted

`multiway_evolve` computes a genuine, well-defined combinatorial object (the
branching-state graph of an abstract rewriting system, with states
deduplicated by canonical relabeling and multi-edges recording distinct
causal derivations that reach the same state). `MultiwaySystem.branching_factor()`
is the only quantity extracted from it, and its docstring says explicitly:
*this is not a cohomology, and no "Branchial Constant" is claimed.*

Nothing here defines a chain complex, a boundary operator, or any invariant
over the multiway graph beyond mean out-degree. "Branchial harmony" and a
scaling constant analogous to the golden ratio remain evocative language
without a working mathematical object behind them. This is, honestly, the
weakest-grounded of the four original concepts (see the recommendation
given when this module was commissioned) and is not implemented beyond the
raw graph it would need to operate on.

### 3. Rulial Operators (computational derivatives) — not implemented

No code here computes a "rate of topological deformation per rewrite step"
or attempts to define energy/mass as rewrite-event density. This has real
grounding via existing discrete-exterior-calculus and causal-set literature,
but building it was out of scope for this pass — `evolve()`'s step-indexed
history is the raw material a rulial derivative would need, nothing more.

### 4. Pre-Geometric Dimension Variables — Tier B implemented

`local_dimension` / `mean_dimension` / `dimension_profile` implement exactly
this: dimension as a per-node, per-region statistic that can vary across a
hypergraph and across an evolution history. This is the most directly
useful and best-tested piece of the module. It is a measurement, not a
unification of general relativity with a discrete network — no claim of
that kind is made or checked here.

---

## Known limitations (read before extending)

- **Rule search is exhaustive only over its own bounded parametrization.**
  `rule_space` does not search "all possible rules" — increasing
  `rhs_edge_count` or `num_pattern_vars` grows the space combinatorially
  fast; there is no pruning beyond the caller's `top_k`/scoring choice.
- **`evolve()` can blow up combinatorially.** `max_edges` is a hard stop,
  not a performance guarantee — a rule whose RHS has $k$ edges can multiply
  the edge count by roughly $k$ every generation once all edges are
  independently matchable (verified directly in the test suite, not
  assumed), so `max_edges` can be overshot within a single generation by up
  to that factor before the check fires.
- **Multiway deduplication depends on `Hypergraph.relabeled()`**, which
  canonicalizes by first-appearance order of a *specific* edge sequence, not
  by graph isomorphism. Two isomorphic hypergraphs whose edges were produced
  in a different order will **not** be deduplicated to the same state unless
  they happen to relabel identically — this is stated as a caveat in
  `core.py` and is a real limitation, not a subtle bug: a proper isomorphism
  canonical form (e.g. via `networkx`'s VF2 or a certified graph-canon
  algorithm) would be needed to make multiway deduplication exact in
  general. The current tests exercise cases where relabeling happens to
  coincide (verified empirically, not assumed) rather than claiming general
  correctness.
- **The dimension estimator is validated on regular lattices only.** Its
  behavior on irregular, evolving hypergraphs (the actual object of
  interest) is unverified beyond what the test suite's evolution-based
  cases show; treat `dimension_profile` output on a genuine evolution
  history as exploratory, not as a calibrated instrument, until it has its
  own known-answer tests against a hypergraph whose true emergent dimension
  is independently derivable.

## How to use this

```python
from socrates.hypergraph import Hypergraph, rule_space, solve_for_rule, mean_dimension

seed = Hypergraph.of((0, 1))
candidates = rule_space(rhs_edge_count=2, arity=2, num_pattern_vars=3)

def toward_2d(hg):
    return abs(mean_dimension(hg, samples=5, max_radius=4) - 2.0)

results = solve_for_rule(seed, candidates, toward_2d, steps=4, top_k=5)
for r in results:
    print(r.rule.rhs, r.score)
```

This is a real, runnable search — not a demonstration that any resulting
rule "is" the rule of physical space. Report results the same way the rest
of this repository reports everything: with the gate that produced them
named, and the tier stated.
