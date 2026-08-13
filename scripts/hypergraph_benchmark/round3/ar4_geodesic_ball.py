"""AutoResearch iteration 4: metric (Euclidean-weighted geodesic) balls instead of
hop-count shells.

THE ONE CHANGE THIS SCRIPT MEASURES
-----------------------------------
`local_dimension` reads its growth law off HOP-COUNT balls: radius r means
"r edges away", and the observable is the shell size |ball(r)| - |ball(r-1)|,
fitted as log(shell) vs log(r) with `dimension = slope + 1`. The candidate
replaces the hop count with the METRIC graph geodesic: every k-NN edge carries
its Euclidean length, radius becomes a continuous distance, and the observable
becomes the cumulative count N(R) fitted as log(rank) vs log(geodesic distance),
with `dimension = slope` (a cumulative metric ball carries no additive offset, so
the shell-vs-volume argument in dimension.py's module docstring -- which is what
forces the `+1` on the hop side -- does not apply).

WHY, AND WHY IT IS THE RIGHT THING TO TRY AT THIS POINT IN THE LOOP.
AR3's exhaustive scan (ar3_fit_dof_gate.py section 2) established that on Lorenz
at n=400 -- the single `nan` that pins problem 08 at min_n=800 -- NO contiguous
fit window over radii 1..10 reaches tolerance with a residual degree of freedom,
because the local hop-slope decays monotonically and there is no plateau. That
scan varied the WINDOW while holding the shell sequences fixed, so it could not
distinguish "this cloud has no scaling region at n=400" from "hop counting cannot
see the scaling region this cloud has". The distinction matters, because hop
counting on a Theiler-windowed k-NN graph is a genuinely poor proxy for geodesic
distance: the window forces every edge to jump between different passes of the
attractor, so edge LENGTHS vary by more than an order of magnitude and a hop is
not a unit of distance. The measured Lorenz n=400 shell sequence
(10, 38, 18, 35, 18, 31) shows the symptom directly -- a period-2 parity
oscillation, which is a graph artifact rather than a geometric one.

Metric geodesics are the standard repair, and they are literature-grounded rather
than invented here: graph distance on a dense enough proximity graph converges to
the manifold geodesic (Bernstein, de Silva, Langford & Tenenbaum 2000, the Isomap
convergence result that pointcloud.py's own module docstring already cites), and
intrinsic-dimension estimation directly off graph geodesic distances is Granata &
Carnevale (2016). The rank-vs-distance readout is the same quantity Levina &
Bickel (2004) take the maximum-likelihood estimate of, so the MLE
  1/m = (1/(J-1)) * sum_{j<J} log(T_J / T_j)
over exactly the same rank window is reported alongside the least-squares slope
as a second readout of the same change (section 3). Note this is NOT the entry on
the DO-NOT-RETRY list: that was Levina-Bickel on RAW EUCLIDEAN k-NN distances at
k=6, whose 1/(k-1) small-k bias is the documented +0.5. Here the distances are
geodesic and the rank window runs to J = fraction * N, so k is in the hundreds
and that particular bias is negligible.

WHAT IS HELD FIXED, so this measures the estimator and not a new knob. The
candidate reuses (a) the identical symmetrized k-NN edge set -- `_build_graph`
below is asserted edge-for-edge equal to `pointcloud.knn_hypergraph` in
`_selfcheck()`; (b) the identical Theiler window ("auto"); (c) the identical
stride-sampled 40 nodes; (d) the identical R^2 >= 0.9 acceptance threshold; and
(e) the identical finite-size saturation fraction, `SATURATION_BALL_FRACTION`
= 0.5, now cutting the rank window at J = floor(0.5 * N) instead of cutting the
hop window at the first ball past half the sample. Nothing in the candidate is
newly calibrated.

SELF-CONTAINED BY CONSTRUCTION. The candidate is implemented here and the library
is NOT modified (see RESULT), so every number below re-runs against the current
source tree. `_selfcheck()` asserts the shared graph construction is bit-for-bit
identical to the library's, which is what licenses reading the difference between
the columns as the estimator change and nothing else.

SECTIONS (select with --sections, default all)
  1  self-check: the candidate's graph is edge-for-edge and node-for-node the
     library's `knn_hypergraph`, on 08/09/02/10 at several n
  2  before/after: per-n traces and `poly_algebraic_min_n` for the library
     shell-growth estimator vs the candidate, on 08 and 09, over the full
     production n_grid at the AR2 best setting (theiler="auto", fraction 0.5)
  3  is any gain a property of the data or of a knob (the R2-F3 test): the
     fraction swept over {0.05, 0.1, 0.2, 0.35, 0.5} x readout in {LS, MLE}
  4  regression spot check: problems 02 and 10 under the candidate, and
     `comparison.compare()` through the UNMODIFIED library on 02 and 10 to
     confirm poly_algebraic_min_n is still 64

RESULT (measured 2026-08-13; the library was left unmodified).
NOT KEPT, on both halves of the keep-rule at once.

  * Lorenz, at the held-fixed fraction 0.5: min_n 800 -> 800, NEUTRAL. The
    blocking cell does change character -- n=400 stops being `nan` and becomes a
    finite 1.385 with 40 of 40 nodes accepted -- but 1.385 is OUTSIDE the
    [1.55, 2.55] tolerance band, so the trace still breaks at n=400. Per-n
    AFTER: 2.673, 1.848, 1.385, 1.910, 1.787, 1.758, 1.827, 1.862 against BEFORE
    2.318, 1.988, nan, 1.930, 2.157, 2.152, 2.156, 2.126 (the BEFORE column
    reproduces the recorded trace exactly, which is the run-to-run control).
  * Rossler, same setting: min_n 100 -> 6400, a REGRESSION that destroys the
    recorded win (savings 0.875 -> 0.111 against traditional_min_n 800). n=100
    reads 3.092 and n=3200 reads 1.442, both outside [1.51, 2.51].

Section 3 rules out rescuing this with the fraction. Across
{0.05, 0.1, 0.2, 0.35, 0.5} x {LS, MLE} the best Lorenz result is min_n=200
(LS, fraction 0.2) and the best Rossler result anywhere is min_n=800 -- so NO
setting improves one without regressing the other, and Rossler is worse than its
recorded 100 in all ten. The Lorenz 200 is additionally perched rather than on a
plateau (0.1 -> 1600, 0.2 -> 200, 0.35 -> 800) on a trace that drifts by 1.4
across the grid (3.219, 2.315, 1.749, 2.299, 2.108, 1.785, 1.878, 1.862) and sits
inside the band only because the tolerance is +/-0.5. Banking it would be finding
R2-F3's failure mode -- a win contingent on a knob -- on top of a non-converged
trace.

THE STANDING VALUE OF THE NEGATIVE RESULT. AR3 closed Lorenz n=400 against every
FIT WINDOW, holding the shell sequences fixed. This closes it against a different
DISTANCE: an estimator sharing nothing with the hop-count fit except the edge set
reads 1.222-1.749 there at every fraction and both readouts. The n=400 prefix of
this cloud genuinely has no ~2-D scaling region at accessible scales, so
`nan` is not a hop-counting artifact and Lorenz's min_n=800 is a property of the
data. Section 4(a) is the control that the candidate is not simply broken: on the
1-D clouds 02 and 10 it converges monotonically to 1.005 with no degenerate
branch involved (min_n 64 and 32).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, load_cloud  # noqa: E402

from socrates.hypergraph import pointcloud as pc  # noqa: E402
from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    SATURATION_BALL_FRACTION,
    mean_dimension,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

CHAOTIC = ("08", "09")
SPOT_CHECK = ("02", "10")

# Recorded before-numbers this iteration must reproduce (AR2 best setting).
RECORDED_BEFORE = {"08": 800, "09": 100}


# =============================================================================
# The candidate estimator
# =============================================================================


def _build_graph(points, k: int, theiler_window):
    """The library's k-NN construction, with the coordinate array kept aligned.

    `knn_hypergraph` returns a `Hypergraph` and drops the coordinates, but the
    candidate needs an edge LENGTH per edge, so the construction is repeated here
    against `pointcloud`'s own private helpers -- the same dedupe, the same
    Theiler resolution, the same symmetrization -- rather than reimplemented.
    `_selfcheck()` asserts the resulting edge set is identical to the library's.
    """
    n = len(points)
    times = pc._resolve_time_indices(None, n)
    arr = np.asarray(points, dtype=float)
    tree = cKDTree(arr)
    bbox = pc._bbox_diagonal(arr)
    threshold = 1e-6 * bbox if bbox > 0 else 1e-6
    clusters = pc._duplicate_clusters(tree, n, threshold)
    if clusters:
        arr, points, times = pc._drop_duplicate_clusters(arr, clusters, n, times)
        n = len(points)
        tree = cKDTree(arr)
    if k >= n:
        raise ValueError(f"k ({k}) must be less than the number of points ({n})")
    window = pc._resolve_theiler_window(theiler_window, arr, times)
    if window <= 1:
        _, idx = tree.query(arr, k=k + 1)
        neighbours = [[int(j) for j in row if int(j) != i][:k] for i, row in enumerate(idx)]
    else:
        neighbours = pc._windowed_neighbours(tree, arr, times, k, window)
    edges = set()
    for i, row in enumerate(neighbours):
        for j in row:
            edges.add((i, j) if i < j else (j, i))
    return arr, tuple(sorted(edges))


def _log_log_slope(xs, ys):
    """Least-squares slope and R^2 of log(ys) vs log(xs) -- the same fit
    `dimension._log_log_fit` performs, minus the near-constant machinery (which
    is a statement about hop-count shell sequences and has no analogue here)."""
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    var_x = sum((a - mx) ** 2 for a in lx)
    if var_x == 0:
        return 0.0, 0.0
    slope = cov / var_x
    intercept = my - slope * mx
    ss_tot = sum((b - my) ** 2 for b in ly)
    ss_res = sum((b - (slope * a + intercept)) ** 2 for a, b in zip(lx, ly, strict=True))
    if ss_tot <= 1e-24 * max(1.0, sum(b * b for b in ly)):
        return slope, 1.0
    return slope, 1.0 - ss_res / ss_tot


def _sampled(n_nodes: int, samples: int) -> list[int]:
    """`dimension._sampled_nodes`' stride rule, on node ids 0..n-1."""
    nodes = list(range(n_nodes))
    if samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    return nodes


def geodesic_mean_dimension(
    arr,
    edges,
    *,
    samples: int = 40,
    max_ball_fraction: float = SATURATION_BALL_FRACTION,
    readout: str = "ls",
    r_squared_threshold: float = 0.9,
) -> tuple[float, int, int]:
    """The candidate. Returns (mean dimension, accepted nodes, sampled nodes).

    For each sampled source: Dijkstra over the Euclidean-weighted k-NN graph, sort
    the finite positive distances, and read the growth law off the first
    J = floor(max_ball_fraction * N) of them -- the metric-ball form of the same
    finite-size saturation cut `local_dimension` applies to hop balls.

      readout="ls"  -- least squares of log(rank) vs log(distance); the slope IS
                       the dimension (cumulative metric count, no affine offset,
                       so no `+1`). Accepted iff R^2 >= threshold, the same bar
                       `mean_dimension` uses.
      readout="mle" -- Levina & Bickel (2004) over the identical rank window.
                       There is no fit residual to threshold, so every node with a
                       usable window is accepted; that difference is exactly why
                       both are reported rather than one.
    """
    n = len(arr)
    ii = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    jj = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    weights = np.linalg.norm(arr[ii] - arr[jj], axis=1)
    graph = coo_matrix((weights, (ii, jj)), shape=(n, n)).tocsr()

    sources = _sampled(n, samples)
    dist = dijkstra(graph, directed=False, indices=sources)

    j_max = int(max_ball_fraction * n)
    accepted: list[float] = []
    for row in dist:
        d = np.sort(row)
        d = d[np.isfinite(d) & (d > 0.0)]
        upper = min(j_max, len(d))
        if upper < 4:  # fewer than 4 ranks is not a fit
            continue
        radii = [float(x) for x in d[:upper]]
        ranks = [float(j) for j in range(1, upper + 1)]
        if readout == "ls":
            slope, r_squared = _log_log_slope(radii, ranks)
            if r_squared >= r_squared_threshold:
                accepted.append(slope)
        elif readout == "mle":
            t_j = radii[-1]
            total = sum(math.log(t_j / t) for t in radii[:-1] if t > 0.0)
            if total > 0.0:
                accepted.append((upper - 1) / total)
        else:
            raise ValueError(f"unknown readout {readout!r}")
    if not accepted:
        return float("nan"), 0, len(sources)
    return sum(accepted) / len(accepted), len(accepted), len(sources)


def _min_n(trace, true_dimension: float, tolerance: float) -> int | None:
    """`comparison.poly_algebraic_minimum_points`' stable-convergence rule, applied
    to an already-computed (n, estimate) trace."""
    for i, (_, dim) in enumerate(trace):
        if not math.isfinite(dim) or abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in trace[i:]):
            return trace[i][0]
    return None


def _band(spec) -> str:
    lo = spec.true_dimension - spec.tolerance
    hi = spec.true_dimension + spec.tolerance
    return f"[{lo:.2f}, {hi:.2f}]"


def _inside(value: float, spec) -> bool:
    return math.isfinite(value) and abs(value - spec.true_dimension) <= spec.tolerance


# =============================================================================
# Sections
# =============================================================================


def _selfcheck() -> None:
    print("SECTION 1 -- self-check: candidate graph == library graph")
    print("-" * 78)
    ok = True
    for problem in ("08", "09", "02", "10"):
        spec = SPECS[problem]
        points = load_cloud(problem)
        for n in spec.n_grid[:4]:
            if n > len(points) or spec.k >= n:
                continue
            subset = points[:n]
            for window in (0, "auto"):
                try:
                    hg = knn_hypergraph(subset, k=spec.k, dedupe=True, theiler_window=window)
                except (pc.DuplicatePointsError, ValueError) as exc:
                    print(f"  {problem} n={n:<6} window={window!s:<5} library raised: {exc}")
                    continue
                try:
                    arr, edges = _build_graph(subset, spec.k, window)
                except ValueError as exc:
                    print(f"  {problem} n={n:<6} window={window!s:<5} candidate raised: {exc}")
                    continue
                same_edges = tuple(sorted(hg.edges)) == edges
                same_nodes = sorted(hg.nodes) == sorted({u for e in edges for u in e})
                ok = ok and same_edges and same_nodes
                print(
                    f"  {problem} n={n:<6} window={window!s:<5} "
                    f"edges={len(edges):<7} identical={same_edges} nodes_match={same_nodes} "
                    f"coords={arr.shape}"
                )
    print(f"\n  SELF-CHECK {'PASSED' if ok else 'FAILED'}: the two paths build the same graph.")
    assert ok, "candidate graph construction diverged from the library"
    print()


def _section2() -> None:
    print("SECTION 2 -- before/after on 08 and 09 (theiler='auto', fraction 0.5)")
    print("-" * 78)
    print("  BEFORE = library dimension.mean_dimension (hop shells, +1)")
    print("  AFTER  = candidate geodesic_mean_dimension (metric ranks, LS readout)")
    print()
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(f"  {problem} {spec.label}: k={spec.k} max_radius={spec.max_radius} "
              f"true={spec.true_dimension} tol={spec.tolerance} band={_band(spec)}")
        before, after = [], []
        for n in spec.n_grid:
            if n > len(points) or spec.k >= n:
                break
            subset = points[:n]
            try:
                hg = knn_hypergraph(subset, k=spec.k, dedupe=True, theiler_window="auto")
                b = mean_dimension(
                    hg,
                    samples=min(n, spec.samples),
                    max_radius=spec.max_radius,
                    max_ball_fraction=SATURATION_BALL_FRACTION,
                )
            except (pc.DuplicatePointsError, ValueError):
                b = float("nan")
            try:
                arr, edges = _build_graph(subset, spec.k, "auto")
                a, n_acc, n_samp = geodesic_mean_dimension(
                    arr, edges, samples=spec.samples,
                    max_ball_fraction=SATURATION_BALL_FRACTION, readout="ls",
                )
            except ValueError:
                a, n_acc, n_samp = float("nan"), 0, 0
            before.append((n, b))
            after.append((n, a))
            mark_b = "*" if _inside(b, spec) else " "
            mark_a = "*" if _inside(a, spec) else " "
            print(f"    n={n:<6} before={b:8.3f}{mark_b}   after={a:8.3f}{mark_a} "
                  f"(accepted {n_acc}/{n_samp})")
        mb = _min_n(before, spec.true_dimension, spec.tolerance)
        ma = _min_n(after, spec.true_dimension, spec.tolerance)
        print(f"    poly_algebraic_min_n:  BEFORE {mb}   AFTER {ma}   "
              f"(recorded before: {RECORDED_BEFORE[problem]})")
        if mb != RECORDED_BEFORE[problem]:
            print("    !! BEFORE does not reproduce the recorded number -- investigate")
        print()


def _section3() -> None:
    print("SECTION 3 -- is any gain a property of the data or of the fraction knob?")
    print("-" * 78)
    print("  (finding R2-F3's test: a win contingent on a knob is not a win)")
    print()
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(f"  {problem} {spec.label}  band={_band(spec)}")
        graphs: dict[int, tuple] = {}
        for n in spec.n_grid:
            if n > len(points) or spec.k >= n:
                break
            try:
                graphs[n] = _build_graph(points[:n], spec.k, "auto")
            except ValueError:
                graphs[n] = None
        for readout in ("ls", "mle"):
            for fraction in (0.05, 0.1, 0.2, 0.35, 0.5):
                trace = []
                cells = []
                for n, built in graphs.items():
                    if built is None:
                        est = float("nan")
                    else:
                        arr, edges = built
                        est, _, _ = geodesic_mean_dimension(
                            arr, edges, samples=spec.samples,
                            max_ball_fraction=fraction, readout=readout,
                        )
                    trace.append((n, est))
                    inside = _inside(est, spec)
                    cells.append(f"{est:6.3f}{'*' if inside else ' '}")
                got = _min_n(trace, spec.true_dimension, spec.tolerance)
                print(f"    {readout:3s} frac={fraction:<5} min_n={str(got):<6} " + " ".join(cells))
        print()


def _section4() -> None:
    print("SECTION 4 -- regression spot check")
    print("-" * 78)
    print("  (a) the candidate on the 1-D spot-check clouds 02 and 10")
    for problem in SPOT_CHECK:
        spec = SPECS[problem]
        points = load_cloud(problem)
        cells = []
        trace = []
        for n in spec.n_grid:
            if n > len(points) or spec.k >= n:
                break
            try:
                arr, edges = _build_graph(points[:n], spec.k, 0)
                est, _, _ = geodesic_mean_dimension(
                    arr, edges, samples=spec.samples,
                    max_ball_fraction=SATURATION_BALL_FRACTION, readout="ls",
                )
            except ValueError:
                est = float("nan")
            trace.append((n, est))
            inside = _inside(est, spec)
            cells.append(f"{n}:{est:.3f}{'*' if inside else ''}")
        got = _min_n(trace, spec.true_dimension, spec.tolerance)
        print(f"    {problem} {spec.label:<28} min_n={got}   " + "  ".join(cells))

    print()
    print("  (b) comparison.compare() through the UNMODIFIED library (must stay 64 vs 256)")
    for problem in SPOT_CHECK:
        spec = SPECS[problem]
        points = load_cloud(problem)
        for fraction in (1.0, SATURATION_BALL_FRACTION):
            result = compare(
                points,
                spec.true_dimension,
                spec.tolerance,
                k=spec.k,
                n_grid=spec.n_grid,
                max_radius=spec.max_radius,
                max_ball_fraction=fraction,
            )
            savings = result.compute_savings_fraction
            print(
                f"    {problem} frac={fraction:<4} poly_min_n={result.poly_algebraic_min_n} "
                f"trad_min_n={result.traditional_min_n} "
                f"savings={'n/a' if savings is None else f'{savings:.3f}'}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default="1,2,3,4")
    args = parser.parse_args()
    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}
    if "1" in wanted:
        _selfcheck()
    if "2" in wanted:
        _section2()
    if "3" in wanted:
        _section3()
    if "4" in wanted:
        _section4()


if __name__ == "__main__":
    main()
