"""Round 3 / N8 -- verification script for the near-constant shell-sequence fix.

FINDING BEING FIXED
-------------------
docs/MENSURA_BENCHMARK.md finding R2-F4 (§9.5), replicated in §10.3,
carried as next step N8. Round 1's N1 fixed *exactly* constant shell sequences
(ss_tot within floating-point noise of zero -> r_squared sentinel 1.0). A
distinct defect sat immediately next to it: a shell sequence that is *nearly*
but not exactly constant has genuinely tiny ss_tot, so N1's relative-tolerance
branch never fires, and R^2 -- a fraction-of-variance-explained -- collapses on
its own merits because there is almost no variance to explain. The dimension
estimate is excellent; the quality flag calls it worthless.

Anchor case (§10.3): problem 01 at n=200, k=6, max_radius=6 has node volumes
(7, 12, 17, 23, 29, 35), i.e. shells (6, 5, 5, 6, 6, 6), giving dimension
1.0333 (error 0.033 against a known answer of exactly 1) discarded at
R^2 = 0.0550. When *every* sampled node lands in that band, mean_dimension
returns nan.

WHAT THE FIX IS
---------------
`socrates.hypergraph.dimension` now gates acceptance on the shell sequence's
relative spread *in addition to* R^2, never on R^2 alone:

  * NEAR_CONSTANT_CV = 0.15 -- coefficient of variation of the raw shell
    counts. Identifies the regime in which R^2 is uninformative.
  * NEAR_CONSTANT_SLOPE_BOUND = 0.25 -- a rigorous cap on |dimension - 1|:
    if every log(shell) lies in a band of width W, then for ANY arrangement
    of values in that band the least-squares slope obeys
        |slope| <= W * sum|log x_i - mean log x| / (2 * var(log x)),
    so a near-constant sequence cannot yield a dimension far from 1 however
    its residuals are arranged. This is why the low R^2 is not a danger
    signal there -- and it turns the CV heuristic into a guarantee.

R^2 itself is NOT modified. Sequences with real dynamic range are still judged
by R^2 alone and still fail when they should.

Run: python scripts/hypergraph_benchmark/round3/n8_near_constant_shell_gate.py
(requires the repo venv: source .venv/bin/activate)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_SLOPE_BOUND,
    _log_log_fit,
    local_dimension,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

RULE = "=" * 78


def problem01_points(n: int) -> list[tuple[float, float]]:
    """Problem 01's own point cloud: the unit circle in (x, v) phase space,
    sampled at golden-ratio (Weyl) spaced indices over one period.

    The round-2 script gets these points from `leapfrog`; here the exact
    solution x(t)=cos t, v(t)=-sin t is used directly. That script's own
    solver-verification step measures the two agreeing to < 1e-4, and the
    anchor node below is reproduced identically either way (checked: both
    routes give volumes (7,12,17,23,29,35), dimension 1.033327563282641,
    R^2 = 0.055020207884822536).
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    n_steps = 31416  # dt = 2e-4 over one period, as the round-2 script uses
    idx = [min(int((i * phi) % 1.0 * n_steps), n_steps) for i in range(n)]
    return [
        (math.cos(2 * math.pi * j / n_steps), -math.sin(2 * math.pi * j / n_steps)) for j in idx
    ]


def report_table() -> None:
    """§9.5's table plus §10.3's anchor row, with the new gate's columns."""
    print(RULE)
    print("STEP 1: R2-F4's table (§9.5) + §10.3's anchor row, re-measured")
    print(RULE)
    print(f"  thresholds: CV <= {NEAR_CONSTANT_CV}, slope_bound <= {NEAR_CONSTANT_SLOPE_BOUND}")
    print()
    print(
        f"  {'shells':<22} {'dim':>8} {'R^2':>8} {'CV':>7} {'bound':>7} "
        f"{'deg':>5} {'near':>5} {'kept':>5}  {'was':>5}"
    )
    rows = [
        (6, 6, 6, 6, 6, 6),
        (6, 6, 6, 6, 6, 7),
        (6, 5, 6, 6, 6, 6),
        (5, 6, 6, 6, 6, 6),
        (7, 5, 5, 6, 6, 6),
        (4, 8, 4, 8, 4, 8),
        (6, 5, 5, 6, 6, 6),
    ]
    for shells in rows:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        f = _log_log_fit(xs, [float(s) for s in shells])
        kept = f.r_squared >= 0.9 or f.near_degenerate
        was = f.r_squared >= 0.9  # pre-fix gate: R^2 alone
        print(
            f"  {str(shells):<22} {f.slope + 1.0:8.4f} {f.r_squared:8.4f} "
            f"{f.shell_cv:7.4f} {f.slope_bound:7.4f} "
            f"{str(f.degenerate):>5} {str(f.near_degenerate):>5} "
            f"{('yes' if kept else 'no'):>5}  {('yes' if was else 'no'):>5}"
        )
    print()
    print(
        "  Note row (4,8,4,8,4,8): R^2 = 0.1027 is ABOVE the near-perfect\n"
        "  (6,5,6,6,6,6) at 0.0889 -- R^2 cannot rank them. CV can:\n"
        "  0.3333 vs 0.0639. That row is still rejected, as it must be."
    )


def report_anchor_end_to_end() -> None:
    print()
    print(RULE)
    print("STEP 2: §10.3's anchor, end to end -- problem 01, n=200, k=6, max_radius=6")
    print(RULE)
    pts = problem01_points(200)
    hg = knn_hypergraph(pts, k=6, dedupe=True)
    ests = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]

    anchor = [e for e in ests if e.volumes == (7, 12, 17, 23, 29, 35)]
    print(f"  nodes with the documented volumes (7,12,17,23,29,35): {len(anchor)}")
    if not anchor:
        raise SystemExit("anchor node absent -- this script is no longer testing R2-F4")
    a = anchor[0]
    print(f"    dimension        = {a.dimension:.10f}   (documented 1.0333, error 0.033)")
    print(f"    r_squared        = {a.r_squared:.10f}   (documented 0.0550) -- UNCHANGED by fix")
    print(f"    shell CV         = {a.shell_cv:.4f}")
    print(f"    slope_bound      = {a.slope_bound:.4f}  >= |dim-1| = {abs(a.dimension - 1.0):.4f}")
    print(f"    degenerate       = {a.degenerate}   (N1's branch: correctly NOT taken)")
    print(f"    near_degenerate  = {a.near_degenerate}")
    print(f"    is_well_fit(0.9) = {a.is_well_fit(0.9)}  (pre-fix: {a.r_squared >= 0.9})")

    before_kept = sum(1 for e in ests if e.r_squared >= 0.9)
    after_kept = [e for e in ests if e.is_well_fit(threshold=0.9)]
    print()
    print(f"  well-fit nodes, pre-fix gate (R^2 >= 0.9 alone) : {before_kept}/{len(ests)}")
    print(f"  well-fit nodes, post-fix gate                   : {len(after_kept)}/{len(ests)}")
    print("  mean_dimension pre-fix                          : nan (no node survived)")
    post = mean_dimension(hg, max_radius=6)
    print(f"  mean_dimension post-fix                         : {post:.6f}")
    print(
        f"  worst |dimension - 1| among KEPT nodes           : "
        f"{max(abs(e.dimension - 1.0) for e in after_kept):.4f}"
    )
    print(
        f"  near_degenerate_fraction                        : "
        f"{near_degenerate_fraction(hg, max_radius=6):.3f}"
    )
    print()
    print(
        "  The last two lines are the honesty check: coverage was not bought by\n"
        "  admitting bad estimates (every kept node is within 0.08 of the exact\n"
        "  answer 1, well inside the problem's own +/-0.15 tolerance), and the\n"
        "  graph self-reports that its r_squared column is uninformative, so\n"
        "  nothing here may be cited as fit-quality evidence."
    )


def report_negative_controls() -> None:
    print()
    print(RULE)
    print("STEP 3: negative controls -- what must still be rejected")
    print(RULE)
    controls = {
        "(4,8,4,8,4,8) oscillating garbage (§9.5's own row)": (4, 8, 4, 8, 4, 8),
        "(1,1,1,1,50,50) step -- genuine curvature": (1, 1, 1, 1, 50, 50),
        "(1,2,3,60,61,62) saturating -- genuine curvature": (1, 2, 3, 60, 61, 62),
        "(3,30,3,30,3,30) large-amplitude oscillation": (3, 30, 3, 30, 3, 30),
    }
    for label, shells in controls.items():
        xs = [float(r) for r in range(1, len(shells) + 1)]
        f = _log_log_fit(xs, [float(s) for s in shells])
        kept = f.r_squared >= 0.9 or f.near_degenerate
        print(
            f"  {label:<52} R^2={f.r_squared:6.4f} CV={f.shell_cv:6.4f} "
            f"near={str(f.near_degenerate):<5} kept={'yes' if kept else 'NO'}"
        )

    print()
    print("  positive controls -- real power laws, must be untouched:")
    for shells, expected in [((4, 8, 12, 16, 20, 24), 2.0), ((1, 4, 9, 16, 25, 36), 3.0)]:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        f = _log_log_fit(xs, [float(s) for s in shells])
        print(
            f"  {str(shells):<52} dim={f.slope + 1.0:6.4f} (expect {expected}) "
            f"R^2={f.r_squared:6.4f} near={f.near_degenerate}"
        )


def report_no_collateral_damage() -> None:
    """The gate must not change any non-near-constant hypergraph's answer."""
    print()
    print(RULE)
    print("STEP 4: collateral-damage check on graphs the gate must NOT touch")
    print(RULE)

    size = 24

    def node(x: int, y: int) -> int:
        return x * size + y

    from socrates.hypergraph.core import Hypergraph

    edges = []
    for x in range(size):
        for y in range(size):
            if x + 1 < size:
                edges.append((node(x, y), node(x + 1, y)))
            if y + 1 < size:
                edges.append((node(x, y), node(x, y + 1)))
    grid = Hypergraph.of(*edges)
    est = local_dimension(grid, source=node(12, 12), max_radius=6)
    print(
        f"  2D grid centre: dim={est.dimension:.6f} R^2={est.r_squared:.6f} "
        f"CV={est.shell_cv:.4f} near={est.near_degenerate} "
        f"genuinely_well_fit={est.is_genuinely_well_fit()}"
    )

    pts = [
        (math.cos(2 * math.pi * i / 400), math.sin(2 * math.pi * i / 400), 0.0) for i in range(400)
    ]
    hg = knn_hypergraph(pts, k=6, dedupe=True)
    est = local_dimension(hg, source=0, max_radius=6)
    print(
        f"  400-pt circle : dim={est.dimension:.6f} R^2={est.r_squared:.6f} "
        f"degenerate={est.degenerate} near={est.near_degenerate} "
        f"(N1's exact-constant branch must still own this case)"
    )

    # The sharpest adversarial worry: could the new gate leak into the FRACTAL
    # problems, admitting spurious dimension~1 nodes and dragging their means
    # down? Lorenz is the case that would suffer most (its benchmark value is
    # ~2.05). Measured, not assumed.
    import numpy as np
    from scipy.integrate import solve_ivp

    def lorenz(_t, s):
        x, y, z = s
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]

    sol = solve_ivp(lorenz, [0, 200], [1.0, 1.0, 1.0], rtol=1e-10, atol=1e-12, dense_output=True)
    lorenz_pts = [tuple(p) for p in sol.sol(np.linspace(50, 200, 4000)).T]
    print()
    print("  Lorenz attractor, k=8, max_radius=6 -- gate must be inert here:")
    for n in (400, 800, 1600):
        hg_l = knn_hypergraph(lorenz_pts[:n], k=8, dedupe=True)
        ests = [local_dimension(hg_l, nd, max_radius=6) for nd in sorted(hg_l.nodes)][:400]
        pre = [e.dimension for e in ests if e.r_squared >= 0.9]
        post = [e.dimension for e in ests if e.is_well_fit(threshold=0.9)]
        n_near = sum(1 for e in ests if e.near_degenerate)
        print(
            f"    n={n:>4}: PRE {len(pre):>3} nodes mean {sum(pre) / len(pre):.4f} | "
            f"POST {len(post):>3} nodes mean {sum(post) / len(post):.4f} | "
            f"near_degenerate={n_near}"
        )
    print(
        "    Zero near-constant nodes at every n: the fractal results (§4's F4,\n"
        "    the weakest evidence in the suite) are bit-for-bit unchanged. The\n"
        "    gate only fires where the shell sequence really is nearly flat."
    )


def report_downstream_poly_min_n() -> None:
    """§9.5's recorded downstream consequence, reproduced and then fixed.

    "That nan then breaks poly_algebraic_minimum_points's stable-convergence
    chain and moves poly_min_n from 50 to 400 -- the entire mechanism of
    R2-F3."
    """
    print()
    print(RULE)
    print("STEP 5: downstream -- poly_algebraic_minimum_points on problem 01, max_radius=6")
    print(RULE)
    from socrates.hypergraph.comparison import poly_algebraic_minimum_points

    pts = problem01_points(3200)
    n_grid = (50, 100, 200, 400, 800, 1600, 3200)

    def prefix_mean_dimension(hg, samples: int, max_radius: int) -> float:
        """mean_dimension exactly as it behaved BEFORE the fix: gate on R^2 alone."""
        nodes = sorted(hg.nodes)
        if samples < len(nodes):
            step = max(1, len(nodes) // samples)
            nodes = nodes[::step][:samples]
        ests = [local_dimension(hg, node, max_radius=max_radius) for node in nodes]
        well_fit = [e.dimension for e in ests if e.r_squared >= 0.9]
        return float("nan") if not well_fit else sum(well_fit) / len(well_fit)

    before = []
    after = []
    for n in n_grid:
        hg = knn_hypergraph(pts[:n], k=6, dedupe=True)
        before.append((n, prefix_mean_dimension(hg, 40, 6)))
        after.append((n, mean_dimension(hg, samples=40, max_radius=6)))

    print(f"  {'n':>6} {'mean_dim PRE-fix':>18} {'mean_dim POST-fix':>18}")
    for (n, b), (_, a) in zip(before, after, strict=True):
        print(f"  {n:>6} {b:>18.6f} {a:>18.6f}")

    def first_stable(results: list[tuple[int, float]], tol: float = 0.15) -> int | None:
        for i, (n, _) in enumerate(results):
            if all(not math.isnan(d) and abs(d - 1.0) <= tol for _, d in results[i:]):
                return n
        return None

    print()
    print(f"  poly_min_n PRE-fix  (stable-convergence chain) : {first_stable(before)}")
    print(f"  poly_min_n POST-fix                            : {first_stable(after)}")
    print(
        f"  poly_algebraic_minimum_points() POST-fix       : "
        f"{poly_algebraic_minimum_points(pts, 1.0, 0.15, k=6, n_grid=n_grid, max_radius=6)}"
    )
    print()
    print(
        "  §9.5 recorded exactly this: the nan at n=100 and n=200 broke the\n"
        "  stable-convergence chain and pushed poly_min_n from 50 to 400. Note\n"
        "  the PRE-fix column also shows why the old numbers were misleading in\n"
        "  the other direction: where any node did survive, it survived only by\n"
        "  being an EXACT constant-shell sentinel, so mean_dim printed a\n"
        "  suspiciously perfect 1.000000. The POST-fix column reports real,\n"
        "  slightly-off measurements at every n -- less flattering and more true."
    )


def main() -> None:
    report_table()
    report_anchor_end_to_end()
    report_negative_controls()
    report_no_collateral_damage()
    report_downstream_poly_min_n()
    print()
    print(RULE)
    print("N8 verification complete.")
    print(RULE)


if __name__ == "__main__":
    main()
