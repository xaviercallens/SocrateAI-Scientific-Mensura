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


def test_adjacency_is_cached_and_correct():
    # Performance fix: .adjacency() used to rebuild an O(edges) dict from
    # scratch on every call, which compounded with local_dimension calling
    # ball() once per radius into the dominant cost of a dimension-vs-n
    # convergence sweep (measured: 136s -> 5.0s on a 3200-point sweep after
    # this fix). Two equal-but-distinct Hypergraph instances must return the
    # SAME cached dict object (confirming the cache is keyed by value, via
    # Hypergraph's hash/eq, not by identity) -- and the content must still
    # be correct, not merely fast.
    hg1 = Hypergraph.of((0, 1), (1, 2))
    hg2 = Hypergraph.of((0, 1), (1, 2))
    assert hg1 is not hg2
    assert hg1 == hg2
    assert hg1.adjacency() is hg2.adjacency()
    assert hg1.adjacency() == {0: {1}, 1: {0, 2}, 2: {1}}

    # A structurally different hypergraph must not share the cached entry.
    hg3 = Hypergraph.of((0, 1), (1, 3))
    assert hg3.adjacency() != hg1.adjacency()


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
    # N2: the path's shell sequence is exactly constant, so this r_squared=1.0
    # is the degenerate sentinel, not a genuine multi-value fit -- is_well_fit()
    # alone cannot see this distinction; is_genuinely_well_fit() must not.
    assert est.degenerate is True
    assert est.is_well_fit() and not est.is_genuinely_well_fit()


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
    # N2: the grid's shell sequence (4, 8, 12, 16) genuinely varies -- this
    # r_squared=1.0 is a real fit, not the degenerate sentinel.
    assert est.degenerate is False
    assert est.is_genuinely_well_fit()


def test_mean_dimension_matches_local_dimension_on_a_homogeneous_lattice():
    hg = Hypergraph.of(*[(i, i + 1) for i in range(40)])
    m = mean_dimension(hg, samples=5, max_radius=5)
    assert m == pytest.approx(1.0, abs=1e-9)


def test_degenerate_fraction_distinguishes_path_from_grid():
    # N2: a path's shell sequence is constant everywhere sampled -> fully
    # degenerate. A grid's shell sequence genuinely varies (4,8,12,16) ->
    # not degenerate. This is exactly the distinction finding F1 of
    # docs/POLY_ALGEBRAIC_BENCHMARK.md showed is missing from a bare
    # r_squared reading.
    from socrates.hypergraph.dimension import degenerate_fraction

    path = Hypergraph.of(*[(i, i + 1) for i in range(40)])
    assert degenerate_fraction(path, samples=8, max_radius=5) == pytest.approx(1.0)

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
    grid = Hypergraph.of(*edges)
    assert degenerate_fraction(grid, samples=8, max_radius=4) == pytest.approx(0.0)


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
            # N8 changed _log_log_fit's return from a 3-tuple to a _FitResult
            # record (it now also carries the near-constant gate's fields);
            # the assertions below are unchanged from the original N1 test.
            fit = _log_log_fit(xs, ys)
            assert fit.slope == pytest.approx(0.0, abs=1e-9)
            assert fit.r_squared == pytest.approx(1.0, abs=1e-9), (
                f"shell={shell} n={n}: r_squared collapsed to {fit.r_squared}"
            )
            assert fit.degenerate is True  # N2: constant shell sequence must self-report as such
            # ...and the exactly-constant case stays in the `degenerate` branch,
            # not the new near-constant one -- N8 subsumes nothing of N1.
            assert fit.near_degenerate is False


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


# ---- N8 / R2-F4: near-constant (not exactly-constant) shell sequences --------
#
# docs/POLY_ALGEBRAIC_BENCHMARK.md §9.5 (finding R2-F4), §10.3 and next step N8.
# N1 fixed *exactly* constant shell sequences. Immediately adjacent sat a
# distinct defect: a shell sequence that is nearly but not exactly constant has
# genuinely tiny ss_tot, so N1's relative-tolerance branch does not fire, and
# R^2 -- a fraction-of-variance-explained -- collapses on its own merits
# because there is almost no variance to explain. Every number asserted below
# was measured against the PRE-fix code first, not assumed.


def _layered_hypergraph(shells: tuple[int, ...]) -> Hypergraph:
    """A hypergraph whose BFS shell sequence from node 0 is exactly `shells`.

    Each layer's nodes attach to a single spine node in the previous layer,
    so no shortcut edge can merge layers and the BFS frontier sizes are
    exactly the requested counts. This is §10.3's "synthetic graph built to
    have exactly those shells", made deterministic and dependency-free.
    """
    edges: list[tuple[int, int]] = []
    prev_layer = [0]
    next_id = 1
    for size in shells:
        layer = list(range(next_id, next_id + size))
        next_id += size
        for node in layer:
            edges.append((prev_layer[0], node))
        prev_layer = layer
    return Hypergraph.of(*edges)


def test_r2f4_table_reproduces_before_the_fit_gate_is_applied():
    # §9.5's table, plus §10.3's anchor row, as known answers. The point of
    # this test is that the FIX DOES NOT MOVE R^2: r_squared stays the honest
    # variance-explained number for every row. Only the accept/reject gate
    # changes, and it changes via the separate near_degenerate flag.
    from socrates.hypergraph.dimension import _log_log_fit

    # shells -> (dimension, r_squared) exactly as the benchmark reported them
    rows = {
        (6, 6, 6, 6, 6, 6): (1.0000, 1.0000),  # exactly constant -> N1's branch
        (6, 6, 6, 6, 6, 7): (1.0488, 0.2642),
        (6, 5, 6, 6, 6, 6): (1.0335, 0.0889),
        (5, 6, 6, 6, 6, 6): (1.0911, 0.6572),
        (7, 5, 5, 6, 6, 6): (0.9563, 0.0505),
        (4, 8, 4, 8, 4, 8): (1.1836, 0.1027),
        (6, 5, 5, 6, 6, 6): (1.0333, 0.0550),  # §10.3's anchor: problem 01, n=200
    }
    for shells, (expected_dim, expected_r2) in rows.items():
        xs = [float(r) for r in range(1, len(shells) + 1)]
        fit = _log_log_fit(xs, [float(s) for s in shells])
        assert fit.slope + 1.0 == pytest.approx(expected_dim, abs=5e-5), shells
        assert fit.r_squared == pytest.approx(expected_r2, abs=5e-5), shells


def test_near_constant_shell_sequence_is_kept_but_flagged():
    # The whole of §9.5's near-constant band must now be ACCEPTED (their
    # dimension values are all within 0.09 of the truth) while carrying
    # near_degenerate=True so no caller mistakes the low R^2 for a real
    # fit-quality reading.
    from socrates.hypergraph.dimension import _log_log_fit

    for shells in [
        (6, 6, 6, 6, 6, 7),
        (6, 5, 6, 6, 6, 6),
        (5, 6, 6, 6, 6, 6),
        (7, 5, 5, 6, 6, 6),
        (6, 5, 5, 6, 6, 6),
    ]:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        fit = _log_log_fit(xs, [float(s) for s in shells])
        assert fit.near_degenerate is True, f"{shells}: not recognised as near-constant"
        assert fit.degenerate is False, f"{shells}: must not be confused with N1's exact case"
        assert fit.r_squared < 0.9, f"{shells}: precondition -- R^2 alone still rejects this"
        # The rigorous guarantee that makes accepting it safe: the dimension
        # cannot be further from 1 than slope_bound, whatever the residuals do.
        assert abs(fit.slope) <= fit.slope_bound + 1e-12, shells
        assert abs(fit.slope) < 0.10, f"{shells}: dimension should be close to 1"


def test_r2f4_anchor_case_problem01_n200_is_no_longer_discarded():
    # §10.3's exact anchor, end to end through local_dimension rather than
    # through the private fit helper: problem 01 at n=200, k=6, max_radius=6
    # has node volumes (7, 12, 17, 23, 29, 35) -- shells (6, 5, 5, 6, 6, 6) --
    # dimension 1.0333 (error 0.033 against the known answer of 1) discarded
    # at R^2 = 0.0550.
    hg = _layered_hypergraph((6, 5, 5, 6, 6, 6))
    est = local_dimension(hg, source=0, max_radius=6)

    assert est.volumes == (7, 12, 17, 23, 29, 35)
    # Reproduce the documented numbers to the digit (both unchanged by the fix).
    assert est.dimension == pytest.approx(1.0333, abs=5e-5)
    assert est.r_squared == pytest.approx(0.0550, abs=5e-5)
    # ...and the actual defect: this node used to be thrown away.
    assert est.near_degenerate is True
    assert est.degenerate is False
    assert est.is_well_fit(threshold=0.9), "R2-F4: a 0.033-error node discarded at R^2=0.055"
    # Its r_squared is still not evidence of fit quality, so the honest
    # accessor must say so -- same contract as N2's degenerate sentinel.
    assert not est.is_genuinely_well_fit(threshold=0.9)
    assert abs(est.dimension - 1.0) <= est.slope_bound


def test_r2f4_negative_control_oscillating_shells_still_rejected():
    # §9.5's own negative control, and the sharpest statement of why R^2 alone
    # is not enough: this garbage sequence scored R^2 = 0.1027, ABOVE the
    # near-perfect (6,5,6,6,6,6) at 0.0889 -- R^2 could not rank them. The
    # spread gate must, and must still reject this one.
    from socrates.hypergraph.dimension import _log_log_fit

    xs = [float(r) for r in range(1, 7)]
    garbage = _log_log_fit(xs, [4.0, 8.0, 4.0, 8.0, 4.0, 8.0])
    good = _log_log_fit(xs, [6.0, 5.0, 6.0, 6.0, 6.0, 6.0])

    assert garbage.r_squared > good.r_squared  # the inversion R2-F4 documented
    assert garbage.near_degenerate is False, "oscillating shells are not near-constant"
    assert garbage.degenerate is False
    assert garbage.r_squared < 0.9  # still judged on its real R^2, and still fails
    assert good.near_degenerate is True
    # The gate ranks them correctly where R^2 inverted them.
    assert good.shell_cv < garbage.shell_cv

    hg = _layered_hypergraph((4, 8, 4, 8, 4, 8))
    est = local_dimension(hg, source=0, max_radius=6)
    assert not est.is_well_fit(threshold=0.9)


def test_r2f4_negative_control_genuine_curvature_still_rejected():
    # Second negative control, this one with real dynamic range AND real
    # curvature (a step, and a saturating growth curve): neither is anywhere
    # near constant, so both keep their honest R^2 and both keep failing. If
    # the near-constant gate ever widened enough to swallow these it would be
    # accepting genuinely bad power-law fits.
    from socrates.hypergraph.dimension import _log_log_fit

    for shells in [(1, 1, 1, 1, 50, 50), (1, 2, 3, 60, 61, 62), (3, 30, 3, 30, 3, 30)]:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        fit = _log_log_fit(xs, [float(s) for s in shells])
        assert fit.near_degenerate is False, f"{shells}: curvature must not be excused"
        assert fit.degenerate is False, shells
        assert fit.r_squared < 0.9, f"{shells}: R^2={fit.r_squared}"

        hg = _layered_hypergraph(shells)
        est = local_dimension(hg, source=0, max_radius=len(shells))
        assert not est.is_well_fit(threshold=0.9), shells


def test_r2f4_does_not_change_genuine_power_law_fits():
    # Control in the other direction: sequences with real dynamic range and a
    # real power law must be untouched -- same dimension, same R^2, and NOT
    # flagged near-constant (so is_genuinely_well_fit still holds for them).
    from socrates.hypergraph.dimension import _log_log_fit

    xs = [float(r) for r in range(1, 7)]
    for shells, expected_dim in [((4, 8, 12, 16, 20, 24), 2.0), ((1, 4, 9, 16, 25, 36), 3.0)]:
        fit = _log_log_fit(xs, [float(s) for s in shells])
        assert fit.slope + 1.0 == pytest.approx(expected_dim, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
        assert fit.near_degenerate is False
        assert fit.degenerate is False


def test_r2f4_end_to_end_problem01_n200_no_longer_returns_nan():
    # The consequence §9.5 recorded: at problem 01's n=200, k=6, max_radius=6
    # EVERY sampled node landed in the near-constant band, so mean_dimension
    # returned nan -- which then broke poly_algebraic_minimum_points' stable
    # convergence chain and moved poly_min_n from 50 to 400 (the mechanism of
    # R2-F3). Point cloud: the unit circle in (x, v) phase space sampled at
    # golden-ratio (Weyl) spaced indices, i.e. problem 01's own construction
    # with the leapfrog integrator replaced by its exact solution (which the
    # script itself verifies agree to 1e-4). Verified to contain the §10.3
    # anchor node, volumes (7, 12, 17, 23, 29, 35), before and after the fix.
    import math

    from socrates.hypergraph.dimension import near_degenerate_fraction
    from socrates.hypergraph.pointcloud import knn_hypergraph

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    n_steps = 31416
    idx = [min(int((i * phi) % 1.0 * n_steps), n_steps) for i in range(200)]
    points = [
        (math.cos(2 * math.pi * j / n_steps), -math.sin(2 * math.pi * j / n_steps)) for j in idx
    ]
    hg = knn_hypergraph(points, k=6, dedupe=True)
    estimates = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]

    # The anchor node is present in this graph, with the documented numbers.
    anchor = [e for e in estimates if e.volumes == (7, 12, 17, 23, 29, 35)]
    assert anchor, "reproduction case not present -- the test no longer tests R2-F4"
    assert anchor[0].dimension == pytest.approx(1.0333, abs=5e-5)
    assert anchor[0].r_squared == pytest.approx(0.0550, abs=5e-5)

    # Pre-fix behaviour, asserted explicitly so the regression is visible:
    # gating on R^2 alone keeps ZERO of the 200 nodes.
    assert sum(1 for e in estimates if e.r_squared >= 0.9) == 0

    # Post-fix: nearly all are kept, and the mean is close to the known answer
    # of exactly 1 rather than nan.
    kept = [e for e in estimates if e.is_well_fit(threshold=0.9)]
    assert len(kept) >= 190
    m = mean_dimension(hg, max_radius=6)
    assert not math.isnan(m)
    assert m == pytest.approx(1.0, abs=0.02)
    # Every kept node's dimension is close to the truth -- the fix is not
    # buying coverage by admitting bad estimates.
    assert max(abs(e.dimension - 1.0) for e in kept) < 0.10
    # And the graph self-reports that its r_squared column is uninformative.
    assert near_degenerate_fraction(hg, max_radius=6) > 0.9


def test_r2f4_slope_bound_is_a_bound_on_distance_from_one_not_from_truth():
    """Pins WHAT `slope_bound` guarantees, and what it does not.

    `slope_bound` rigorously bounds |dimension - 1|. It is an *error* bound
    only where the true local dimension is ~1. Independent verification of N8
    found the counterexample class empirically: on 2D Brownian motion (true
    dimension 2) at k=6, max_radius=3, the near-constant branch admitted 13 of
    200 sampled nodes -- every one of them respecting slope_bound on
    |dimension - 1| while being up to 1.18 away from the truth -- and moved
    the sampled mean from 1.8321 to 1.7381, away from 2. The shell sequences
    below are three of those actual admitted nodes.

    The gate now refuses them on fit-window LENGTH (N8b), so the assertions
    are two-sided: the slope_bound arithmetic on these sequences is unchanged
    and still respects the bound (the bound was never the buggy part), but
    `near_degenerate` is now False because three radii cannot distinguish
    "near-constant" from "noisy" in any structure.
    """
    from socrates.hypergraph.dimension import (
        NEAR_CONSTANT_MIN_FIT_LENGTH,
        NEAR_CONSTANT_SLOPE_BOUND,
        _log_log_fit,
    )

    short_window_admissions = [(10, 11, 10), (11, 12, 12), (7, 8, 8)]
    for shells in short_window_admissions:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        fit = _log_log_fit(xs, [float(s) for s in shells])
        # The bound itself is untouched and still holds -- it was correct all
        # along, it just answered a different question than was claimed.
        assert abs(fit.slope) <= fit.slope_bound + 1e-12, shells
        assert abs(fit.slope) <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12, shells
        assert fit.shell_cv <= 0.15, shells  # would have passed the spread gate
        # The guarantee that is NOT made, visibly does not hold: these arose in
        # a true-dimension-2 cloud, and the reported dimension is ~1.
        assert abs((fit.slope + 1.0) - 2.0) > 0.75, shells
        # ...which is exactly why the length condition must reject them.
        assert len(shells) < NEAR_CONSTANT_MIN_FIT_LENGTH
        assert fit.near_degenerate is False, (
            f"{shells}: a 3-radius window must not be excused as near-constant"
        )

    # The protection is window length, not the thresholds: the same wobble
    # amplitude over six radii is what the anchor case actually relies on, and
    # it is still accepted.
    anchor = _log_log_fit([float(r) for r in range(1, 7)], [6.0, 5.0, 5.0, 6.0, 6.0, 6.0])
    assert anchor.near_degenerate is True
    assert len(anchor.__dataclass_fields__) > 0  # _FitResult, not a bare tuple


def test_n8b_min_fit_length_is_the_threshold_the_sweep_selected():
    """The length condition itself, at the exact boundary, on one wobble.

    Same shell values, same +/-1 wobble amplitude, same CV regime -- only the
    window length varies. Everything below NEAR_CONSTANT_MIN_FIT_LENGTH is
    refused and everything at or above it is accepted, so a future edit to the
    constant cannot silently drift without this failing.
    """
    from socrates.hypergraph.dimension import NEAR_CONSTANT_MIN_FIT_LENGTH, _log_log_fit

    assert NEAR_CONSTANT_MIN_FIT_LENGTH == 5

    base = [6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    for n in range(2, 8):
        shells = base[:n]
        fit = _log_log_fit([float(r) for r in range(1, n + 1)], shells)
        expected = n >= NEAR_CONSTANT_MIN_FIT_LENGTH
        assert fit.near_degenerate is expected, (
            f"length {n}: near_degenerate={fit.near_degenerate}, expected {expected}"
        )
        # For lengths 3..7 the spread conditions ALL pass, so at 3 and 4 the
        # rejection is the length condition acting alone -- it is doing real,
        # independent work rather than agreeing with the spread gate by
        # accident. (At n=2 the bound is 0.263 and the spread gate rejects on
        # its own, so that row proves nothing about length either way.)
        if n >= 3:
            assert fit.shell_cv <= 0.15, n
            assert fit.slope_bound <= 0.25, n


def test_n8b_brownian_max_radius3_no_longer_admits_nodes_far_from_truth():
    """The collateral-damage case, end to end on the real point cloud.

    2D Brownian motion, true dimension 2, k=6, max_radius=3 -- the regime the
    round-2 comparison scripts actually use. Before N8b this configuration
    admitted 13 of 200 sampled nodes through the near-constant branch with
    |dimension - truth| up to 1.18, pulling the sampled mean from 1.8321 to
    1.7381 (away from 2). After N8b it admits none and the mean is back to the
    R^2-only value.
    """
    import numpy as np

    from socrates.hypergraph.pointcloud import knn_hypergraph

    rng = np.random.default_rng(3)
    points = [tuple(p) for p in np.cumsum(rng.normal(size=(1500, 2)), axis=0).tolist()]
    hg = knn_hypergraph(points, k=6, dedupe=True)
    nodes = sorted(hg.nodes)
    nodes = nodes[:: max(1, len(nodes) // 200)][:200]
    estimates = [local_dimension(hg, u, max_radius=3) for u in nodes]
    assert len(estimates) == 200

    # No node in a true-dimension-2 cloud is excused by the near-constant
    # branch at this window length any more.
    admitted = [e for e in estimates if e.near_degenerate]
    assert admitted == [], (
        f"{len(admitted)} spurious near-constant admissions: "
        f"{[(e.volumes, round(e.dimension, 4)) for e in admitted[:5]]}"
    )

    # ...so is_well_fit collapses back onto R^2 alone here, and the sampled
    # mean is the undamaged one rather than the 1.7381 the gate produced.
    r2_only = [e.dimension for e in estimates if e.r_squared >= 0.9]
    kept = [e.dimension for e in estimates if e.is_well_fit(threshold=0.9)]
    assert kept == r2_only
    mean = sum(kept) / len(kept)
    assert mean == pytest.approx(1.8321, abs=5e-4), mean
    # And it is nearer the truth than the damaged value was, which is the
    # whole point -- not merely different from it.
    assert abs(mean - 2.0) < abs(1.7381 - 2.0)


def test_n8b_1d_ring_at_the_default_radius_pays_nothing_for_the_length_gate():
    """The other side of the trade: the length condition must not cost the
    case N8 exists for.

    Problem 01's ring at its documented settings (n=200, k=6, max_radius=6)
    has fit length 6, above the threshold, so acceptance is bit-for-bit what
    it was before N8b: 196 of 200 nodes kept and a mean of 0.9943 against a
    known answer of exactly 1, where R^2 alone keeps zero.
    """
    import math

    from socrates.hypergraph.pointcloud import knn_hypergraph

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    n_steps = 31416
    idx = [min(int((i * phi) % 1.0 * n_steps), n_steps) for i in range(200)]
    points = [
        (math.cos(2 * math.pi * j / n_steps), -math.sin(2 * math.pi * j / n_steps)) for j in idx
    ]
    hg = knn_hypergraph(points, k=6, dedupe=True)
    estimates = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]

    anchor = [e for e in estimates if e.volumes == (7, 12, 17, 23, 29, 35)]
    assert anchor, "reproduction case not present -- the test no longer tests R2-F4"
    assert anchor[0].near_degenerate is True, "N8b must not evict the anchor"
    assert anchor[0].dimension == pytest.approx(1.0333, abs=5e-5)
    assert anchor[0].r_squared == pytest.approx(0.0550, abs=5e-5)

    assert sum(1 for e in estimates if e.r_squared >= 0.9) == 0  # R^2 alone: nothing
    kept = [e for e in estimates if e.is_well_fit(threshold=0.9)]
    assert len(kept) == 196
    assert mean_dimension(hg, max_radius=6) == pytest.approx(0.9943, abs=5e-4)


def test_n8b_length_gate_does_not_reach_the_length6_residue_round2_problem06():
    """WHY A NODE-LOCAL GATE CANNOT FIX ROUND-2 PROBLEM 06.

    These are the real shell counts of the node that does the damage in round-2
    problem 06 (2D Brownian, k=10, compare()'s default max_radius=6, n=400):
    the near-constant *candidate* test admits it with dimension 1.0587 against
    a true dimension of 2.

    Its window is length 6 -- the anchor's own length -- so no minimum-fit-
    length threshold can reach it, and a CV threshold cannot either: the
    culprit's CV (0.1224) is only ~4% above the worst CV the gate is required
    to accept ((7,5,5,6,6,6) at 0.1178). Everything asserted here is still
    true, which is the point: the node-local layer alone never fixed this.

    The fix is graph-level and is pinned separately in
    `test_n8d_problem06_culprit_is_refused_by_the_graph_consensus`.
    """
    from socrates.hypergraph.dimension import (
        NEAR_CONSTANT_MIN_FIT_LENGTH,
        _log_log_fit,
    )

    culprit = [13.0, 17.0, 13.0, 13.0, 17.0, 15.0]
    fit = _log_log_fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], culprit)

    assert len(culprit) >= NEAR_CONSTANT_MIN_FIT_LENGTH  # the length gate cannot help
    assert fit.shell_cv == pytest.approx(0.1224, abs=5e-5)
    assert fit.slope_bound == pytest.approx(0.1833, abs=5e-5)
    assert fit.near_degenerate is True  # <-- a CANDIDATE, refused at graph level
    assert fit.slope + 1.0 == pytest.approx(1.0587, abs=5e-5)
    # slope_bound holds, as advertised -- it just bounds distance from 1, and
    # the truth here is 2, which is why holding it is no protection at all.
    assert abs(fit.slope) <= fit.slope_bound
    assert abs((fit.slope + 1.0) - 2.0) > 0.9

    # The margin a node-local CV threshold would have to thread.
    worst_must_accept = _log_log_fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 5.0, 5.0, 6.0, 6.0, 6.0])
    assert worst_must_accept.near_degenerate is True
    assert worst_must_accept.shell_cv < fit.shell_cv
    assert (fit.shell_cv - worst_must_accept.shell_cv) / worst_must_accept.shell_cv < 0.05


def test_n8d_consensus_is_a_graph_level_decision_with_two_measured_branches():
    """Unit semantics of `near_constant_consensus` (N8d, finding R2-F4).

    A flat shell window is evidence for "~1D" only if the STRUCTURE is a
    low-dynamic-range ~1D object, which is a property of the graph. Two
    branches, both exercised here with the constants as shipped.
    """
    from socrates.hypergraph.dimension import (
        NEAR_CONSTANT_CONSENSUS_FRACTION,
        NEAR_CONSTANT_CONSENSUS_TOLERANCE,
        DimensionEstimate,
        near_constant_consensus,
    )

    assert NEAR_CONSTANT_CONSENSUS_TOLERANCE == 0.25
    assert NEAR_CONSTANT_CONSENSUS_FRACTION == 0.5

    def est(node, dimension, r_squared, near):
        return DimensionEstimate(
            node,
            (1, 2, 3, 4, 5, 6),
            (7, 12, 17, 23, 29, 35),
            dimension,
            r_squared,
            near_degenerate=near,
        )

    # No evidence at all is not consensus.
    assert near_constant_consensus([], threshold=0.9) is False

    # Branch 1 -- some node is confident: the branch may not contradict them.
    agree = [est(0, 1.0, 1.0, False), est(1, 1.05, 0.99, False), est(2, 1.0587, 0.08, True)]
    assert near_constant_consensus(agree, threshold=0.9) is True
    disagree = [est(0, 1.9, 0.99, False), est(1, 1.0587, 0.08, True)]
    assert near_constant_consensus(disagree, threshold=0.9) is False
    # Exactly at the tolerance edge, from both sides.
    assert near_constant_consensus([est(0, 1.25, 0.99, False)], threshold=0.9) is True
    assert near_constant_consensus([est(0, 1.2501, 0.99, False)], threshold=0.9) is False
    # Degenerate (constant-shell) nodes count as confident 1D evidence: their
    # sentinel r_squared is 1.0 and their dimension is exactly 1.
    assert near_constant_consensus([est(0, 1.0, 1.0, False)] * 4, threshold=0.9) is True

    # Branch 2 -- nobody is confident, so the near-constant nodes must be the
    # DOMINANT description of the graph, not a handful of flat windows.
    majority = [est(i, 1.03, 0.05, i < 6) for i in range(10)]
    assert near_constant_consensus(majority, threshold=0.9) is True
    minority = [est(i, 1.03, 0.05, i < 4) for i in range(10)]
    assert near_constant_consensus(minority, threshold=0.9) is False
    exactly_half = [est(i, 1.03, 0.05, i < 5) for i in range(10)]
    assert near_constant_consensus(exactly_half, threshold=0.9) is True

    # The flag alone never accepts a node -- the graph has to consent.
    culprit = est(0, 1.0587, 0.0865, True)
    assert culprit.is_well_fit(0.9, near_constant_consensus=True) is True
    assert culprit.is_well_fit(0.9, near_constant_consensus=False) is False
    # ... but a genuinely well-fit node is untouched by the graph's verdict.
    solid = est(1, 2.0, 0.999, False)
    assert solid.is_well_fit(0.9, near_constant_consensus=False) is True
    assert solid.is_genuinely_well_fit(0.9, near_constant_consensus=False) is True


def test_n8d_anchor_is_kept_by_the_graph_consensus_rule():
    """The documented R2-F4 anchor must survive the graph-level rule intact.

    Problem 01's cloud at n=200, k=6, max_radius=6: NO node passes R^2 at all,
    so the anchor graph takes the fallback branch, where 0.98 of the sampled
    nodes are near-constant -- far above the 0.5 majority required. Acceptance
    is bit-for-bit what it was before N8d: 196 of 200 kept, mean 0.9943 against
    a known answer of exactly 1.
    """
    import math

    from socrates.hypergraph.dimension import near_constant_consensus
    from socrates.hypergraph.pointcloud import knn_hypergraph

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    n_steps = 31416
    idx = [min(int((i * phi) % 1.0 * n_steps), n_steps) for i in range(200)]
    points = [
        (math.cos(2 * math.pi * j / n_steps), -math.sin(2 * math.pi * j / n_steps)) for j in idx
    ]
    hg = knn_hypergraph(points, k=6, dedupe=True)
    estimates = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]

    anchor = [e for e in estimates if e.volumes == (7, 12, 17, 23, 29, 35)]
    assert anchor, "reproduction case not present -- the test no longer tests R2-F4"
    assert anchor[0].near_degenerate is True, "N8d must not evict the anchor"
    assert anchor[0].dimension == pytest.approx(1.0333, abs=5e-5)
    assert anchor[0].r_squared == pytest.approx(0.0550, abs=5e-5)

    assert sum(1 for e in estimates if e.r_squared >= 0.9) == 0  # fallback branch
    fraction = sum(1 for e in estimates if e.near_degenerate) / len(estimates)
    assert fraction == pytest.approx(0.98, abs=5e-3)
    assert near_constant_consensus(estimates, threshold=0.9) is True

    kept = [e for e in estimates if e.is_well_fit(threshold=0.9, near_constant_consensus=True)]
    assert len(kept) == 196
    assert mean_dimension(hg, max_radius=6) == pytest.approx(0.9943, abs=5e-4)


def test_n8d_problem06_culprit_is_refused_by_the_graph_consensus():
    """The round-2 problem-06 regression, closed, on its real production cloud.

    2D Brownian motion (seed 42, 80k unit-variance increments, strided to
    ~25.6k points -- the construction in
    scripts/hypergraph_benchmark/round2/06_brownian_motion.py), at n=400,
    k=10, max_radius=6, true dimension 2, tolerance +/-0.3.

    Only 4 of 40 sampled nodes pass R^2, and they report a mean of 1.8434, so
    the graph's own evidence says "this is not a 1D structure" by a margin of
    0.84 against a tolerance of 0.25. The one near-constant candidate --
    shells (13,17,13,13,17,15), dimension 1.0587 -- is therefore refused, and
    the sampled mean stays at the R^2-only value instead of being dragged to
    1.6864 and out of tolerance.
    """
    import numpy as np

    from socrates.hypergraph.dimension import near_constant_consensus
    from socrates.hypergraph.pointcloud import knn_hypergraph

    rng = np.random.default_rng(42)
    path = np.concatenate(
        [np.zeros((1, 2)), np.cumsum(rng.normal(loc=0.0, scale=1.0, size=(80_000, 2)), axis=0)],
        axis=0,
    )
    points = [tuple(p) for p in path[:: max(1, len(path) // 25_600)]][:400]

    hg = knn_hypergraph(points, k=10, dedupe=True)
    nodes = sorted(hg.nodes)
    sampled = nodes[:: max(1, len(nodes) // 40)][:40]
    estimates = [local_dimension(hg, node, max_radius=6) for node in sampled]

    culprit = [e for e in estimates if e.near_degenerate]
    assert len(culprit) == 1, "the round-2 culprit node is not in this cloud any more"
    assert culprit[0].volumes == (14, 31, 44, 57, 74, 89)  # shells (13,17,13,13,17,15)
    assert culprit[0].dimension == pytest.approx(1.0587, abs=5e-5)
    assert abs(culprit[0].dimension - 2.0) > 0.9  # ... against a truth of 2

    confident = [e for e in estimates if e.r_squared >= 0.9]
    assert len(confident) == 4
    consensus_dimension = sum(e.dimension for e in confident) / len(confident)
    assert consensus_dimension == pytest.approx(1.8434, abs=5e-4)
    assert near_constant_consensus(estimates, threshold=0.9) is False

    mean = mean_dimension(hg, samples=40, max_radius=6)
    assert mean == pytest.approx(1.8434, abs=5e-4)  # not 1.6864
    assert abs(mean - 2.0) <= 0.3, "problem 06 n=400 must be back inside its tolerance"


def test_n8d_broadened_calibration_kepler_orbit_graph_reaches_consensus():
    """Broadened calibration, must-fire side: round-2 problem 03's own cloud.

    The round-2 skeptic's proposed rule was calibrated on ONE positive example.
    This is a second, independent one taken from production: the Kepler
    two-body orbit (e=0.6) at n=400, k=6, max_radius=6, true dimension 1.

    35 of 40 sampled nodes are confident and report a mean of exactly 1.0, so
    the graph endorses the near-constant branch and its 2 candidate nodes are
    pooled -- moving the estimate 1.0000 -> 1.0038, still far inside the
    +/-0.15 tolerance. This is the case that must NOT be broken by a rule
    tuned to reject problem 06.
    """
    import numpy as np

    from socrates.hypergraph.dimension import near_constant_consensus
    from socrates.hypergraph.pointcloud import knn_hypergraph
    from socrates.solvers import leapfrog

    gm, ecc, dt = 1.0, 0.6, 5e-4
    r_peri = 1.0 - ecc

    def force(q: np.ndarray) -> np.ndarray:
        return -gm * q / float(np.linalg.norm(q)) ** 3

    run = leapfrog(
        force,
        [r_peri, 0.0],
        [0.0, float(np.sqrt(gm * (1 + ecc) / r_peri))],
        dt=dt,
        n_steps=int(1.02 * 2 * np.pi / dt),
    )
    points = [(float(p[0]), float(p[1])) for p in run.positions[:-20]][:400]

    hg = knn_hypergraph(points, k=6, dedupe=True)
    nodes = sorted(hg.nodes)
    sampled = nodes[:: max(1, len(nodes) // 40)][:40]
    estimates = [local_dimension(hg, node, max_radius=6) for node in sampled]

    confident = [e for e in estimates if e.r_squared >= 0.9]
    assert len(confident) == 35
    consensus_dimension = sum(e.dimension for e in confident) / len(confident)
    assert consensus_dimension == pytest.approx(1.0, abs=1e-12)
    assert near_constant_consensus(estimates, threshold=0.9) is True

    candidates = [e for e in estimates if e.near_degenerate]
    assert len(candidates) == 2, "this graph no longer exercises the near-constant branch"
    mean = mean_dimension(hg, samples=40, max_radius=6)
    assert mean == pytest.approx(1.0038, abs=5e-4)
    assert mean != pytest.approx(1.0, abs=1e-6)  # the branch really did contribute
    assert abs(mean - 1.0) <= 0.15


def test_n8d_broadened_calibration_driven_pendulum_n32_is_below_consensus():
    """Broadened calibration, the measured COST of requiring a majority.

    Round-2 problem 10's own cloud (driven damped pendulum, mode-locked
    period-1) at n=32, k=6, max_radius=6, true dimension 1. Nothing passes
    R^2, so the fallback branch applies -- and only 4 of the 32 nodes are
    near-constant, a fraction of 0.125, well below the 0.5 majority.

    So the branch does NOT fire and this graph returns nan. That is deliberate:
    calling a structure "low dynamic range throughout" on 4 windows out of 32
    is not consensus, and 0.5 is the only threshold with margin on both sides
    once the calibration set is broadened (2D Rossler at n=100 also has no
    confident node, at a fraction of 0.325). The consequence is that problem
    10's `poly_algebraic_min_n` is 64, which is the value
    docs/POLY_ALGEBRAIC_BENCHMARK.md 10.2 records.
    """
    import math

    from socrates.hypergraph.dimension import near_constant_consensus
    from socrates.hypergraph.pointcloud import knn_hypergraph

    gamma, f_drive, omega_drive = 0.5, 0.5, 0.6
    t_drive = 2.0 * math.pi / omega_drive

    def deriv(theta, omega, t):
        return omega, -gamma * omega - math.sin(theta) + f_drive * math.cos(omega_drive * t)

    def rk4_step(theta, omega, t, dt):
        k1 = deriv(theta, omega, t)
        k2 = deriv(theta + 0.5 * dt * k1[0], omega + 0.5 * dt * k1[1], t + 0.5 * dt)
        k3 = deriv(theta + 0.5 * dt * k2[0], omega + 0.5 * dt * k2[1], t + 0.5 * dt)
        k4 = deriv(theta + dt * k3[0], omega + dt * k3[1], t + dt)
        return (
            theta + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            omega + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        )

    # Transient + verification window, exactly as the round-2 script does, so
    # the cloud starts from the same certified point on the attractor.
    theta, omega, t = 0.2, 0.0, 0.0
    verify_dt = t_drive / 500
    for _ in range(250 * 500):
        theta, omega = rk4_step(theta, omega, t, verify_dt)
        t += verify_dt

    m = 4096
    dt = t_drive / m
    time_ordered = []
    for _ in range(m):
        theta, omega = rk4_step(theta, omega, t, dt)
        t += dt
        wrapped = (theta + math.pi) % (2 * math.pi) - math.pi
        time_ordered.append((wrapped, omega))
    perm = [int(f"{i:012b}"[::-1], 2) for i in range(m)]
    points = [time_ordered[perm[i]] for i in range(32)]

    hg = knn_hypergraph(points, k=6, dedupe=True)
    estimates = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]

    assert sum(1 for e in estimates if e.r_squared >= 0.9) == 0  # fallback branch
    fraction = sum(1 for e in estimates if e.near_degenerate) / len(estimates)
    assert fraction == pytest.approx(0.125, abs=1e-9)
    assert fraction < 0.5
    assert near_constant_consensus(estimates, threshold=0.9) is False

    import math as _math

    assert _math.isnan(mean_dimension(hg, samples=40, max_radius=6))
    # With the graph's consent it would have reported 0.9086 -- inside the
    # +/-0.2 tolerance, which is exactly how the node-local gate silently moved
    # this problem's recorded min_n from 64 to 32.
    would_be = [e.dimension for e in estimates if e.is_well_fit(0.9, near_constant_consensus=True)]
    assert sum(would_be) / len(would_be) == pytest.approx(0.9086, abs=5e-4)


def test_n8d_near_constant_branch_can_never_move_a_pooled_mean_out_of_the_unit_band():
    """Why the 2D benchmark problems are structurally safe, not merely lucky.

    N8d's confident branch fires only when the confident nodes' mean is within
    NEAR_CONSTANT_CONSENSUS_TOLERANCE (T) of 1, and every near-constant node
    satisfies |dimension - 1| <= NEAR_CONSTANT_SLOPE_BOUND (also T) by its own
    slope bound. The pooled mean is a convex combination of those two means, so
    BOTH the R^2-only mean and the pooled mean must land in [1 - T, 1 + T].

    The consequence is the guarantee this round actually needs: a problem whose
    tolerance band misses [1 - T, 1 + T] entirely -- every 2D/3D problem in the
    benchmark suite (05 and 06 at 2.0 +/- 0.3, 08 at 2.05 +/- 0.5, 09 at
    2.01 +/- 0.5) -- cannot have an accept/reject verdict changed by this
    branch in EITHER direction, whatever the cloud. Round-2's problem-06
    regression is therefore closed as a class, not as an instance. The
    remaining exposure is confined to problems whose true dimension is itself
    near 1, where a wrong admission is at most T away from the right answer.

    Randomised over synthetic estimate sets rather than one hand-picked case,
    so the algebra is what is pinned.
    """
    import random

    from socrates.hypergraph.dimension import (
        NEAR_CONSTANT_CONSENSUS_TOLERANCE,
        NEAR_CONSTANT_SLOPE_BOUND,
        DimensionEstimate,
        near_constant_consensus,
    )

    tol = NEAR_CONSTANT_CONSENSUS_TOLERANCE
    rng = random.Random(20260813)
    fired_via_confident = 0

    for _ in range(20_000):
        estimates = []
        for i in range(rng.randint(2, 30)):
            confident = rng.random() < 0.5
            estimates.append(
                DimensionEstimate(
                    i,
                    (1, 2, 3, 4, 5, 6),
                    (7, 12, 17, 23, 29, 35),
                    # confident nodes may report ANY dimension; near-constant
                    # ones are constrained only by their own slope bound.
                    dimension=(
                        rng.uniform(-2.0, 4.0)
                        if confident
                        else 1.0 + rng.uniform(-1.0, 1.0) * NEAR_CONSTANT_SLOPE_BOUND
                    ),
                    r_squared=rng.uniform(0.9, 1.0) if confident else rng.uniform(0.0, 0.89),
                    near_degenerate=not confident,
                )
            )
        consensus = near_constant_consensus(estimates, threshold=0.9)
        confident_dims = [e.dimension for e in estimates if e.r_squared >= 0.9]
        if not (consensus and confident_dims):
            continue
        fired_via_confident += 1

        pooled = [
            e.dimension
            for e in estimates
            if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
        ]
        pre_n8_mean = sum(confident_dims) / len(confident_dims)
        pooled_mean = sum(pooled) / len(pooled)
        assert 1.0 - tol - 1e-12 <= pre_n8_mean <= 1.0 + tol + 1e-12
        assert 1.0 - tol - 1e-12 <= pooled_mean <= 1.0 + tol + 1e-12

    assert fired_via_confident > 1_000, "the branch under test was barely exercised"


def test_n8d_slope_bound_holds_over_an_exhaustive_integer_shell_scan():
    """The premise the band argument above rests on, checked by brute force.

    `near_degenerate` is only sound as an acceptance criterion if it really
    implies |dimension - 1| <= NEAR_CONSTANT_SLOPE_BOUND. Scanned exhaustively
    over every integer shell sequence in [1,7]^6 and [1,9]^5 rather than on the
    documented examples, since those are the lengths the anchor and the round-2
    problem-06 culprit both live at.
    """
    from itertools import product

    from socrates.hypergraph.dimension import (
        NEAR_CONSTANT_SLOPE_BOUND,
        _log_log_fit,
    )

    admitted = 0
    worst = 0.0
    for length, hi in ((6, 7), (5, 9)):
        radii = [float(r) for r in range(1, length + 1)]
        for combo in product(range(1, hi + 1), repeat=length):
            fit = _log_log_fit(radii, [float(c) for c in combo])
            assert abs(fit.slope) <= fit.slope_bound + 1e-12, combo
            if fit.near_degenerate:
                admitted += 1
                worst = max(worst, abs(fit.slope))
    assert admitted > 0
    assert worst <= NEAR_CONSTANT_SLOPE_BOUND, worst


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
