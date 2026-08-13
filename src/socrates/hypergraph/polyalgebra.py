"""Poly-Algebraic Calculus: solving for an unknown higher-arity rewrite rule.

Tier C framing, Tier B implementation. The physical claim ("this is the
calculus of pre-geometric bonds") is not asserted or needed here; what is
implemented is a literal, checkable instance of the pattern the brief asked
for: classical algebra solves `f(x) = target` for a number x by searching a
well-defined space of numbers. This module solves `evolve(seed, rule) ~ target`
for an unknown rule R, by *enumerating* a bounded space of candidate rules
(a "higher-arity unknown", i.e. an unknown relation rather than an unknown
scalar) and scoring each one against a target structural property.

This is the operationally meaningful reading of "an unknown structural bond
as the variable to solve for": the search space is finite and explicit
(`rule_space`), the scoring function is a plain Python callable, and the
result is a ranked list of candidate rules together with their measured
score -- exactly the kind of computable, falsifiable object this project's
Tier discipline requires before anything is asserted as a result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product

from .core import Hypergraph
from .rewriting import Edge, RewriteRule, evolve


def rule_space(
    *,
    lhs_edge_count: int = 1,
    rhs_edge_count: int,
    arity: int = 2,
    num_pattern_vars: int,
) -> Iterator[RewriteRule]:
    """Enumerate candidate rules with a bounded shape.

    LHS is fixed to the single canonical pattern using variables -1..-arity
    per edge, chained across `lhs_edge_count` edges by reusing variable
    -arity as the connecting point (the standard "connected pattern"
    convention, matching how Wolfram Model rule search spaces are built) --
    this keeps the LHS shape fixed and searches only over what the RHS does
    with it, which is already a rich enough space to be interesting and small
    enough to enumerate exhaustively.

    RHS is every sequence of `rhs_edge_count` edges of `arity`, drawn from a
    pool of `num_pattern_vars` variables (the LHS variables plus any new ones
    up to the pool size), with variable *identity* determined up to renaming
    -- i.e. this enumerates the space of RHS *shapes*, which is the object
    that actually matters structurally.
    """
    if lhs_edge_count < 1:
        raise ValueError("need at least one LHS edge")

    # First edge uses variables -1..-arity. Each subsequent edge shares its
    # first variable with the previous edge's last variable (the standard
    # "connected pattern" chaining) and introduces (arity - 1) fresh
    # variables, so the pattern as a whole is a single connected path.
    next_var = -1
    lhs: list[Edge] = []
    connector: int | None = None
    for _ in range(lhs_edge_count):
        edge_vars = [connector] if connector is not None else []
        while len(edge_vars) < arity:
            edge_vars.append(next_var)
            next_var -= 1
        lhs.append(tuple(edge_vars))
        connector = edge_vars[-1]

    lhs_var_count = -next_var - 1
    if num_pattern_vars < lhs_var_count:
        raise ValueError(
            f"pattern-variable pool ({num_pattern_vars}) must be at least as large "
            f"as the number of variables the LHS itself uses ({lhs_var_count})"
        )

    pool = list(range(-1, -num_pattern_vars - 1, -1))

    seen: set[tuple[Edge, ...]] = set()
    for rhs_tuple in product(product(pool, repeat=arity), repeat=rhs_edge_count):
        rhs = tuple(rhs_tuple)
        if rhs in seen:
            continue
        seen.add(rhs)
        yield RewriteRule(lhs=tuple(lhs), rhs=rhs, name=f"rhs={rhs}")


@dataclass(frozen=True)
class RuleSearchResult:
    rule: RewriteRule
    score: float
    final_state: Hypergraph


def solve_for_rule(
    seed: Hypergraph,
    candidates: Iterator[RewriteRule],
    score: Callable[[Hypergraph], float],
    *,
    steps: int = 4,
    max_edges: int = 400,
    top_k: int = 5,
    lower_is_better: bool = True,
) -> list[RuleSearchResult]:
    """Evolve `seed` under each candidate rule and rank by `score(final_state)`.

    This is the literal "solve for the unknown bond" operation: `candidates`
    is the space being searched (an iterator, so `rule_space` above can be
    swallowed lazily without materializing a huge list), `score` encodes the
    target property (e.g. `lambda hg: abs(mean_dimension(hg) - 2.0)` to search
    for rules producing emergent 2D structure), and the return value is the
    ranked evidence, not an assertion that any of them "is" a physical rule.
    """
    results: list[RuleSearchResult] = []
    for rule in candidates:
        history = evolve(seed, [rule], steps, max_edges=max_edges)
        final = history[-1]
        if final.num_edges == 0:
            continue  # the rule annihilated everything -- not interesting
        try:
            s = score(final)
        except Exception:  # noqa: BLE001 -- a candidate rule can produce a
            # degenerate hypergraph (e.g. disconnected, too small) that some
            # scoring functions cannot evaluate; skip it rather than crash
            # the whole search over one bad candidate.
            continue
        results.append(RuleSearchResult(rule=rule, score=s, final_state=final))

    results.sort(key=lambda r: r.score, reverse=not lower_is_better)
    return results[:top_k]
