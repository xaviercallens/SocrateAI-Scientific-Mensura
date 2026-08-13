"""Known-answer tests for the hypergraph physics module.

Following this repository's standing discipline: correctness is anchored on
known-answer tests, not self-consistency. A path graph has dimension exactly
1 by construction; a grid has dimension 2; a rule that duplicates an edge
produces exponential edge growth. These are checked directly, not inferred
from the module's own other outputs.
"""

from __future__ import annotations

import pytest

from socrates.hypergraph.core import Hypergraph, ball
from socrates.hypergraph.dimension import local_dimension, mean_dimension
from socrates.hypergraph.polyalgebra import rule_space, solve_for_rule
from socrates.hypergraph.rewriting import (
    MultiwaySystem,
    RewriteRule,
    apply_at,
    evolve,
    find_matches,
    multiway_evolve,
)

# ---------------------------------------------------------------- core.py


def test_hypergraph_nodes_and_arities():
    hg = Hypergraph.of((0, 1, 2), (1, 2), (2, 3, 4, 5))
    assert hg.nodes == frozenset({0, 1, 2, 3, 4, 5})
    assert hg.num_edges == 3
    assert hg.arities() == {3: 1, 2: 1, 4: 1}


def test_fresh_node_never_collides():
    hg = Hypergraph.of((0, 5), (5, 2))
    assert hg.fresh_node() not in hg.nodes
    assert hg.fresh_node() == 6


def test_ball_on_a_path_graph_is_exact():
    # 0 - 1 - 2 - 3 - 4, a plain path; ball(0, r) must be exactly {0..r}.
    hg = Hypergraph.of(*[(i, i + 1) for i in range(4)])
    for r in range(5):
        assert ball(hg, 0, r) == set(range(min(r, 4) + 1))


def test_ball_raises_on_unknown_node():
    hg = Hypergraph.of((0, 1))
    with pytest.raises(ValueError, match="not present"):
        ball(hg, 99, 1)


def test_relabeled_is_stable_under_identical_edge_order():
    hg1 = Hypergraph.of((5, 7), (7, 9))
    hg2 = Hypergraph.of((100, 200), (200, 300))
    assert hg1.relabeled() == hg2.relabeled()


# ---------------------------------------------------------------- rewriting.py


def test_find_matches_simple_edge_pattern():
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2),))
    hg = Hypergraph.of((0, 1), (1, 2), (2, 0))
    matches = find_matches(rule, hg)
    assert len(matches) == 3
    assignments = {(m[-1], m[-2]) for m in matches}
    assert assignments == {(0, 1), (1, 2), (2, 0)}


def test_find_matches_two_edge_connected_pattern_requires_shared_variable():
    # pattern {x,y},{y,z} must not match two edges that don't share a node.
    rule = RewriteRule(lhs=((-1, -2), (-2, -3)), rhs=((-1, -3),))
    hg = Hypergraph.of((0, 1), (2, 3))  # disjoint edges, no shared node
    assert find_matches(rule, hg) == []

    hg_connected = Hypergraph.of((0, 1), (1, 2))
    matches = find_matches(rule, hg_connected)
    assert len(matches) == 1
    assert matches[0] == {-1: 0, -2: 1, -3: 2}


def test_apply_at_replaces_matched_edges_and_mints_fresh_nodes():
    # {x,y} -> {x,y},{y,z}: chain-extension rule, mints a fresh node z.
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)))
    hg = Hypergraph.of((0, 1))
    assignment = {-1: 0, -2: 1}
    result = apply_at(rule, hg, assignment)
    assert result.num_edges == 2
    assert (0, 1) in result.edges
    new_edges = [e for e in result.edges if e != (0, 1)]
    assert len(new_edges) == 1
    assert new_edges[0][0] == 1
    assert new_edges[0][1] not in hg.nodes  # genuinely fresh


def test_evolve_chain_extension_doubles_edges_each_generation():
    # evolve() applies ALL non-overlapping matches per step, not just one.
    # Every edge present independently matches {x,y} and is replaced by two
    # edges ({x,y} plus a fresh chain link), and distinct concrete edges
    # never share a matched-edge identity even when they share a node -- so
    # every edge doubles, every generation, deterministically.
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)))
    seed = Hypergraph.of((0, 1))
    history = evolve(seed, [rule], steps=5)
    assert [h.num_edges for h in history] == [1, 2, 4, 8, 16, 32]


def test_evolve_stops_when_no_rule_matches():
    rule = RewriteRule(lhs=((-1, -2, -3),), rhs=((-1, -2, -3), (-3, -1)))
    seed = Hypergraph.of((0, 1))  # arity-2 edge; rule needs arity 3 -- never matches
    history = evolve(seed, [rule], steps=10)
    assert len(history) == 1
    assert history[0] == seed


def test_evolve_respects_max_edges_cap():
    # A rule whose RHS has 3 edges roughly triples the edge count every
    # generation once all edges are independently matchable. The max_edges
    # check runs only AFTER a full generation is applied, so the cap can be
    # overshot within one generation by up to the rule's own growth factor
    # (here 3x) -- evolve() must still halt (not run all 20 steps) rather
    # than let this compound indefinitely.
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-1, -3), (-3, -2)))
    seed = Hypergraph.of((0, 1))
    history = evolve(seed, [rule], steps=20, max_edges=50)
    assert len(history) < 21  # stopped early, did not run all 20 steps
    assert history[-1].num_edges <= 50 * 3


def test_multiway_evolve_single_edge_seed_has_exactly_one_first_step_child():
    # A seed with only one edge has exactly one possible single-edge update,
    # so its multiway graph must have exactly one child at step 1 -- true
    # regardless of what happens in later generations (where the newly
    # created edge makes a second match possible, see the branching test
    # below; step counts beyond 1 are deliberately not asserted here).
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)))
    seed = Hypergraph.of((0, 1))
    mw = multiway_evolve(seed, [rule], steps=1)
    assert isinstance(mw, MultiwaySystem)
    assert mw.num_states == 2
    assert mw.branching_factor() == pytest.approx(1.0)


def test_multiway_evolve_symmetric_matches_merge_via_two_causal_edges():
    # Two disjoint, structurally identical edges -> two distinct match
    # derivations ("update edge (0,1) first" vs "update edge (2,3) first"),
    # but they reach an ISOMORPHIC hypergraph in both cases (one edge
    # extended by a pendant, one edge untouched), so after canonical
    # relabeling they collapse to a single state reached by two parallel
    # causal transitions -- this is exactly the "distinct histories merging"
    # that a genuine multiway system exhibits, verified here empirically
    # rather than assumed.
    rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)))
    seed = Hypergraph.of((0, 1), (2, 3))
    mw = multiway_evolve(seed, [rule], steps=1)
    assert mw.num_states == 2  # the two derivations merge into one child state
    assert mw.transitions == [(0, 1), (0, 1)]  # two parallel causal edges into it
    assert mw.branching_factor() == pytest.approx(2.0)


# ---------------------------------------------------------------- dimension.py


def test_path_graph_has_dimension_one():
    # A path's shell size is the exact constant 2 from an interior node (one
    # new node on each side per radius step) or 1 from an endpoint -- either
    # way a pure r^0 power law, so the shell-based fit is exact, not merely
    # close.
    hg = Hypergraph.of(*[(i, i + 1) for i in range(30)])
    est = local_dimension(hg, source=0, max_radius=6)
    assert est.is_well_fit()
    assert est.dimension == pytest.approx(1.0, abs=1e-9)
    assert est.r_squared == pytest.approx(1.0, abs=1e-9)


def test_grid_graph_has_dimension_two():
    # An explicit 10x10 grid: edges connect each cell to its right and down
    # neighbours. Ball volume around the centre should grow ~ r^2.
    size = 10

    def node(x: int, y: int) -> int:
        return x * size + y

    edges = []
    for x in range(size):
        for y in range(size):
            if x + 1 < size:
                edges.append((node(x, y), node(x + 1, y)))
            if y + 1 < size:
                edges.append((node(x, y), node(x, y + 1)))
    hg = Hypergraph.of(*edges)

    # On a 4-connected grid the ball is an exact L1 diamond of |B(r)| =
    # 2r^2+2r+1 nodes, so the shell size is exactly 4r -- a pure r^1 power
    # law, giving dimension = 1+1 = 2 exactly.
    est = local_dimension(hg, source=node(5, 5), max_radius=4)
    assert est.is_well_fit(threshold=0.98)
    assert est.dimension == pytest.approx(2.0, abs=1e-9)
    assert est.r_squared == pytest.approx(1.0, abs=1e-9)


def test_mean_dimension_matches_local_dimension_on_a_homogeneous_lattice():
    hg = Hypergraph.of(*[(i, i + 1) for i in range(40)])
    m = mean_dimension(hg, samples=5, max_radius=5)
    assert m == pytest.approx(1.0, abs=1e-9)


def test_dimension_of_disconnected_singleton_edge_is_poorly_fit():
    hg = Hypergraph.of((0, 1))
    est = local_dimension(hg, source=0, max_radius=6)
    assert not est.is_well_fit()  # ball saturates after radius 1; not enough data


def test_log_log_fit_r_squared_does_not_collapse_on_constant_shells():
    # Regression test for a bug found by the 10-problem physics benchmark
    # (docs/POLY_ALGEBRAIC_BENCHMARK.md, finding F2): for a perfectly constant
    # shell sequence, ss_tot should be exactly 0, but floating-point rounding
    # in mean_y = sum(log_y)/n can leave it at ~1e-31 instead, taking the
    # `ss_tot > 0` general branch and collapsing r_squared to ~0 by pure
    # numerical noise -- even though the fitted slope stays correct. This hit
    # shell values 6, 17, and 18 at fit-length 6, which is local_dimension's
    # default max_radius, i.e. it broke mean_dimension's default settings for
    # several ordinary lattices (a 400-point circle at k=6 among them).
    #
    # Every value here reproduces a case the benchmark found broken before
    # the fix (verified directly against the pre-fix code, not assumed).
    from socrates.hypergraph.dimension import _log_log_fit

    for shell in (2, 4, 5, 6, 7, 8, 10, 17, 18):
        for n in (3, 6):
            xs = [float(x) for x in range(1, n + 1)]
            ys = [float(shell)] * n
            slope, r_squared = _log_log_fit(xs, ys)
            assert slope == pytest.approx(0.0, abs=1e-9)
            assert r_squared == pytest.approx(1.0, abs=1e-9), (
                f"shell={shell} n={n}: r_squared collapsed to {r_squared}"
            )


def test_circle_at_k6_is_well_fit_at_the_default_max_radius():
    # The concrete end-to-end case the benchmark hit: a 400-point circle at
    # k=6 (a circulant ring lattice, constant shell size 6) must be well-fit
    # at local_dimension's own default max_radius=6, not only at max_radius=5.
    import math

    from socrates.hypergraph.pointcloud import knn_hypergraph

    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 400 for i in range(400))]
    hg = knn_hypergraph(points, k=6)
    est = local_dimension(hg, source=0, max_radius=6)
    assert est.is_well_fit()
    assert est.dimension == pytest.approx(1.0, abs=1e-6)
    assert est.r_squared == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- polyalgebra.py


def test_rule_space_enumerates_exact_expected_count():
    # arity=2, num_pattern_vars=2, rhs_edge_count=1: pool has 2 vars, each
    # RHS edge has 2 slots -> 2*2 = 4 possible edges -> 4 candidate rules.
    rules = list(rule_space(rhs_edge_count=1, arity=2, num_pattern_vars=2))
    assert len(rules) == 4
    for r in rules:
        assert r.lhs == ((-1, -2),)
        assert len(r.rhs) == 1


def test_rule_space_two_edge_lhs_chains_correctly():
    rules = list(rule_space(lhs_edge_count=2, rhs_edge_count=1, arity=2, num_pattern_vars=3))
    # LHS should be a connected 2-edge path: (-1,-2), (-2,-3).
    assert all(r.lhs == ((-1, -2), (-2, -3)) for r in rules)


def test_rule_space_rejects_insufficient_pattern_var_pool():
    with pytest.raises(ValueError, match="pattern-variable pool"):
        list(rule_space(lhs_edge_count=2, rhs_edge_count=1, arity=2, num_pattern_vars=2))


def test_solve_for_rule_prefers_non_growing_rules_over_a_growing_one():
    # Hand-verified dynamics, not inferred from the code under test:
    #
    # With a 2-variable pool and rhs_edge_count=1, no candidate rule can ever
    # introduce a fresh node (new_vars = rhs_vars - lhs_vars = {} whenever
    # the pool equals the LHS variables), and the single matched edge is
    # always replaced by exactly one new edge -- so every one of these 4
    # rules leaves the hypergraph at exactly 1 edge, forever, regardless of
    # how many steps are taken (each of the 4 possible single-edge RHS
    # shapes -- identity, reversal, or a self-loop on either endpoint --
    # keeps re-matching itself trivially every generation).
    #
    # A genuine chain-extension rule {x,y} -> {x,y},{y,z}, in contrast,
    # doubles the edge count every generation once evolved under evolve()'s
    # full-parallel-match semantics (verified separately above): after
    # `steps` generations, exactly 2**steps edges.
    #
    # So scoring by |final_edge_count - 1| must rank all four flat rules
    # tied for first (score 0) strictly ahead of the growing rule.
    seed = Hypergraph.of((0, 1))
    flat_candidates = list(rule_space(rhs_edge_count=1, arity=2, num_pattern_vars=2))
    growing_rule = RewriteRule(lhs=((-1, -2),), rhs=((-1, -2), (-2, -3)), name="chain-extension")
    candidates = iter([*flat_candidates, growing_rule])

    steps = 3
    results = solve_for_rule(
        seed,
        candidates,
        lambda hg: abs(hg.num_edges - 1),
        steps=steps,
        top_k=len(flat_candidates) + 1,
        lower_is_better=True,
    )

    assert len(results) == len(flat_candidates) + 1
    flat_results = [r for r in results if r.rule.name != "chain-extension"]
    growing_result = next(r for r in results if r.rule.name == "chain-extension")

    assert len(flat_results) == 4
    for r in flat_results:
        assert r.score == 0
        assert r.final_state.num_edges == 1

    assert growing_result.final_state.num_edges == 2**steps
    assert growing_result.score == 2**steps - 1
    # The flat rules must all outrank (sort before) the growing one.
    assert results.index(growing_result) == len(results) - 1


def test_solve_for_rule_respects_top_k_and_ordering():
    seed = Hypergraph.of((0, 1))
    candidates = rule_space(rhs_edge_count=1, arity=2, num_pattern_vars=2)
    results = solve_for_rule(
        seed, candidates, lambda hg: float(hg.num_edges), steps=2, top_k=2, lower_is_better=True
    )
    assert len(results) <= 2
    if len(results) == 2:
        assert results[0].score <= results[1].score
