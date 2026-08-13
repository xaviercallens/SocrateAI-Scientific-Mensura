"""Hypergraph rewrite rules and evolution (the Wolfram Model update rule).

Tier B: given a rule and a hypergraph, `apply_all_matches`/`evolve` compute a
well-defined deterministic (given a match order) or branching (multiway)
sequence of hypergraphs. What that sequence *means* physically is Tier C and
lives in `docs/HYPERGRAPH_NOTES.md`, never here.

A rule's left-hand side (LHS) is a tuple of pattern edges over *pattern
variables* (negative integers, by convention, to keep them disjoint from the
non-negative node ids used in concrete hypergraphs without needing a separate
type). Matching a pattern edge to a concrete edge requires a *consistent*
assignment of pattern variables to nodes across every pattern edge in the
rule -- this is subhypergraph pattern matching, not merely per-edge matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Edge, Hypergraph, Node

PatternVar = int  # by convention: negative integers


@dataclass(frozen=True)
class RewriteRule:
    """LHS -> RHS over shared pattern variables.

    Pattern variables appearing in the RHS but not the LHS are treated as
    *new* nodes created by the rule (this is how the Wolfram Model grows the
    hypergraph -- e.g. rule {{-1,-2}} -> {{-1,-2},{-2,-3}} adds a new node
    -3 attached to the matched edge).
    """

    lhs: tuple[Edge, ...]
    rhs: tuple[Edge, ...]
    name: str = ""

    def __post_init__(self) -> None:
        lhs_vars = {v for e in self.lhs for v in e}
        if not all(v < 0 for v in lhs_vars):
            raise ValueError("LHS must use only pattern variables (negative integers)")

    @property
    def lhs_vars(self) -> set[PatternVar]:
        return {v for e in self.lhs for v in e}

    @property
    def new_vars(self) -> set[PatternVar]:
        """Pattern variables introduced fresh on the RHS (new nodes)."""
        rhs_vars = {v for e in self.rhs for v in e}
        return rhs_vars - self.lhs_vars


Assignment = dict[PatternVar, Node]


def _match_edge(pattern: Edge, edge: Edge, partial: Assignment) -> Assignment | None:
    """Try to extend `partial` to match `pattern` against a concrete `edge`."""
    if len(pattern) != len(edge):
        return None
    assignment = dict(partial)
    for var, node in zip(pattern, edge, strict=True):
        if var in assignment:
            if assignment[var] != node:
                return None
        else:
            # A pattern var may not be assigned two different nodes, and two
            # distinct pattern vars may not collapse to the same node unless
            # the pattern explicitly repeats the variable (already handled
            # above) -- this enforces injective-on-repeats matching.
            assignment[var] = node
    return assignment


def find_matches(
    rule: RewriteRule, hg: Hypergraph, *, limit: int | None = None
) -> list[Assignment]:
    """All ways to assign hg's nodes to rule.lhs's pattern variables.

    Backtracking search over which concrete edge matches each pattern edge,
    in order. Exponential in the worst case (subhypergraph isomorphism is
    NP-hard in general) -- `limit` caps the number of matches returned, which
    is essential for any rule search over more than a handful of nodes.
    """
    matches: list[Assignment] = []

    def backtrack(i: int, assignment: Assignment) -> None:
        if limit is not None and len(matches) >= limit:
            return
        if i == len(rule.lhs):
            matches.append(assignment)
            return
        pattern_edge = rule.lhs[i]
        for edge in hg.edges:
            extended = _match_edge(pattern_edge, edge, assignment)
            if extended is not None:
                backtrack(i + 1, extended)
                if limit is not None and len(matches) >= limit:
                    return

    backtrack(0, {})
    return matches


def _instantiate(
    rule: RewriteRule, assignment: Assignment, next_node: int
) -> tuple[list[Edge], int]:
    """Concretize rule.rhs under `assignment`, minting fresh nodes for new_vars."""
    full = dict(assignment)
    for var in sorted(rule.new_vars):
        full[var] = next_node
        next_node += 1
    new_edges = [tuple(full[v] for v in edge) for edge in rule.rhs]
    return new_edges, next_node


def _matched_edges(rule: RewriteRule, assignment: Assignment) -> set[Edge]:
    return {tuple(assignment[v] for v in pattern_edge) for pattern_edge in rule.lhs}


def apply_at(rule: RewriteRule, hg: Hypergraph, assignment: Assignment) -> Hypergraph:
    """Apply `rule` at one specific match, replacing the matched edges."""
    matched = _matched_edges(rule, assignment)
    remaining = [e for e in hg.edges if e not in matched or not _consume(matched, e)]
    new_edges, _ = _instantiate(rule, assignment, hg.fresh_node())
    return Hypergraph(tuple(remaining) + tuple(new_edges))


def _consume(matched: set[Edge], edge: Edge) -> bool:
    """Remove one occurrence of `edge` from `matched` if present (multiset semantics)."""
    if edge in matched:
        matched.discard(edge)
        return True
    return False


def _non_overlapping(
    assignments: list[tuple[RewriteRule, Assignment]],
) -> list[tuple[RewriteRule, Assignment]]:
    """Greedily select matches that touch disjoint sets of matched edges.

    The Wolfram Model updates *all* non-overlapping matches simultaneously
    per generation (not just one) -- this is what makes evolution look like
    parallel local updates across the whole hypergraph rather than a single
    point mutation.
    """
    chosen: list[tuple[RewriteRule, Assignment]] = []
    used_edges: set[Edge] = set()
    for rule, assignment in assignments:
        matched = _matched_edges(rule, assignment)
        if matched & used_edges:
            continue
        chosen.append((rule, assignment))
        used_edges |= matched
    return chosen


def evolve(
    hg: Hypergraph,
    rules: list[RewriteRule],
    steps: int,
    *,
    max_edges: int = 5000,
) -> list[Hypergraph]:
    """One deterministic evolution history: apply all non-overlapping matches per step.

    Returns the sequence [hg_0, hg_1, ..., hg_steps] (or shorter if no rule
    matches, or `max_edges` is exceeded -- hypergraph evolution can blow up
    combinatorially, and a silent unbounded run is exactly the kind of
    footgun this project's methodology exists to prevent).
    """
    history = [hg]
    current = hg
    for _ in range(steps):
        candidates: list[tuple[RewriteRule, Assignment]] = []
        for rule in rules:
            for assignment in find_matches(rule, current, limit=64):
                candidates.append((rule, assignment))
        if not candidates:
            break
        chosen = _non_overlapping(candidates)
        matched_edges: set[Edge] = set()
        next_node = current.fresh_node()
        new_edge_batches: list[list[Edge]] = []
        for rule, assignment in chosen:
            matched_edges |= _matched_edges(rule, assignment)
            batch, next_node = _instantiate(rule, assignment, next_node)
            new_edge_batches.append(batch)
        remaining = []
        matched_pool = dict.fromkeys(matched_edges, 0)
        for e in current.edges:
            if e in matched_pool and matched_pool[e] == 0:
                matched_pool[e] = 1
                continue
            remaining.append(e)
        new_edges = tuple(e for batch in new_edge_batches for e in batch)
        current = Hypergraph(tuple(remaining) + new_edges)
        if current.num_edges > max_edges:
            history.append(current)
            break
        history.append(current)
    return history


def multiway_evolve(
    hg: Hypergraph,
    rules: list[RewriteRule],
    steps: int,
    *,
    max_states: int = 2000,
) -> MultiwaySystem:
    """Branch over *every* single-match update (not just non-overlapping batches).

    This is the raw material for a branchial graph: every distinct
    hypergraph reachable within `steps` single-edge-updates, with edges
    recording which state led to which. States are deduplicated by their
    canonical relabeling, so two different update orders that reach the same
    hypergraph merge into one node -- Wolfram's interpretation of this
    merging as "quantum interference" is Tier C and not asserted here; what
    is computed is exactly the multiway graph of an abstract rewriting
    system, a well-defined combinatorial object on its own.
    """
    start_key = hg.relabeled()
    states: dict[Hypergraph, int] = {start_key: 0}
    order: list[Hypergraph] = [start_key]
    causal_edges: list[tuple[int, int]] = []
    frontier = [start_key]

    for _ in range(steps):
        if len(states) >= max_states:
            break
        next_frontier: list[Hypergraph] = []
        for state in frontier:
            for rule in rules:
                for assignment in find_matches(rule, state, limit=16):
                    child = apply_at(rule, state, assignment).relabeled()
                    if child not in states:
                        if len(states) >= max_states:
                            continue
                        states[child] = len(order)
                        order.append(child)
                        next_frontier.append(child)
                    causal_edges.append((states[state], states[child]))
        frontier = next_frontier
        if not frontier:
            break

    return MultiwaySystem(states=order, transitions=causal_edges)


@dataclass(frozen=True)
class MultiwaySystem:
    """The branching state graph of an abstract rewriting system.

    `states[i]` is the i-th distinct hypergraph reached; `transitions` are
    (parent_index, child_index) pairs. Branchial *width* at generation g
    (how many distinct states exist at that generation) is the honest,
    Tier-B-computable proxy this module offers for "how much the multiway
    system branches" -- it is not a cohomology and no invariant/"Branchial
    Constant" is claimed from it. See docs/HYPERGRAPH_NOTES.md.
    """

    states: list[Hypergraph]
    transitions: list[tuple[int, int]]

    @property
    def num_states(self) -> int:
        return len(self.states)

    def branching_factor(self) -> float:
        """Mean out-degree across all states with at least one child."""
        out_degree: dict[int, int] = {}
        for parent, _ in self.transitions:
            out_degree[parent] = out_degree.get(parent, 0) + 1
        if not out_degree:
            return 0.0
        return sum(out_degree.values()) / len(out_degree)
