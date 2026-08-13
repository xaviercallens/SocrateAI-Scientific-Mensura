"""AutoResearch iteration 3: a residual-degrees-of-freedom floor on the R^2 branch.

THE ONE CHANGE THIS SCRIPT MEASURES
-----------------------------------
`DimensionEstimate` records `fit_length` (how many radii actually entered the
log-log fit) and `is_well_fit` gains `min_fit_length` (default 2 = OFF = pre-AR3
code path verbatim, since two radii is the shortest fit that exists at all).
When set to 3, the R^2 branch may accept a node only if its fit window had at
least one residual degree of freedom. `mean_dimension`,
`comparison.poly_algebraic_minimum_points` and `comparison.compare` forward it.

That source change was implemented, measured, and then REVERTED (see RESULT
below), so THIS SCRIPT DOES NOT DEPEND ON IT: `_mean_dimension` below reimplements
`dimension.mean_dimension` plus the candidate floor, and `_selfcheck` asserts the
reimplementation reproduces the library bit-for-bit with the floor off. Everything
here therefore re-runs against the current, unmodified library.

WHY. A least-squares line through k points has k-2 residual degrees of freedom.
At k=2 the residual is identically zero, so `r_squared` is exactly 1.0 for ANY
pair of shell counts -- it is an algebraic identity, not a measurement, and it
passes any R^2 threshold by construction. dimension.py already makes precisely
this argument for the OTHER acceptance branch (NEAR_CONSTANT_MIN_FIT_LENGTH
condition 0) and AR2's own calibration table names the same mechanism as the
reason `max_ball_fraction = 0.35` counts as a regression on problems 02 and 10.
The R^2 branch -- which carries the bulk of the acceptances -- never had the
floor, and AR2's `max_ball_fraction` guard is what made two-radius windows
common enough to matter.

SECTIONS (select with --sections, default all)
  1  the defect: fit-window length behind every ACCEPTED node, at the AR2 best
     setting, on 08/09 and on the 02/10 spot-check clouds
  2  Lorenz n=400, the open question AR2 left: an EXHAUSTIVE scan of every
     contiguous fit window [a,b] over radii 1..10, all nodes, showing that no
     window with a residual degree of freedom puts the estimate in tolerance --
     i.e. the n=400 hole is not a degrees-of-freedom problem and this change
     cannot be expected to close it
  3  before/after compare() on 08 and 09 over min_fit_length in {2 (off), 3},
     at theiler_window="auto" x max_ball_fraction in {1.0, 0.5}, with per-n
     traces
  4  regression spot check: problems 02 and 10 must stay at
     poly_algebraic_min_n = 64, and DEFAULT settings must reproduce the recorded
     numbers on every problem bit-for-bit

Sections 1, 3 and 4 run in a few minutes; section 2 is ~1 minute.

RESULT (measured 2026-08-13, recorded here because the source change was
REVERTED): the floor removes exactly the vacuous acceptances and nothing else.
Lorenz does not move (800 -> 800). Rossler's AR2 headline result moves
100 -> 200, because 40 of its 40 accepted nodes at n=100 are two-radius fits.
The change is correct and costs measured sample efficiency, so under the
AutoResearch keep-rule it was not kept. Section 1 and section 2 stand on their
own and need no source change -- they re-derive the fit window from
`local_dimension`'s returned volumes.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, load_cloud  # noqa: E402

from socrates.hypergraph.baseline import minimum_points_for_target_accuracy  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    SATURATION_BALL_FRACTION,
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402

CHAOTIC = ("08", "09")
SPOT_CHECK = ("02", "10")

MIN_FIT_LENGTH = 3  # = dimension.MIN_R2_FIT_LENGTH; 2 is the off switch.

RECORDED_DEFAULT_MIN_N = {
    "01": 50, "02": 64, "03": 100, "04": 100, "05": 400,
    "06": 400, "07": 100, "08": 800, "09": 1600, "10": 64,
}


def _fit_length(estimate, n_nodes: int, max_ball_fraction: float) -> int:
    """Re-derive the number of radii that entered the fit, from public fields.

    Mirrors `local_dimension`'s window rule exactly (stop at an exactly-zero
    shell or at the first radius whose PREVIOUS ball is past the budget) so this
    script measures the defect without depending on the candidate source change.
    """
    volumes = (1,) + tuple(estimate.volumes)
    budget = max_ball_fraction * n_nodes
    length = 0
    for r in range(1, len(volumes)):
        if volumes[r] - volumes[r - 1] <= 0 or volumes[r - 1] > budget:
            break
        length += 1
    return length


def _mean_dimension(
    hg, *, samples: int, max_radius: int, max_ball_fraction: float, min_fit_length: int
) -> float:
    """`dimension.mean_dimension` with the candidate degrees-of-freedom floor added.

    Reimplemented here rather than called through the library because the source
    change this script measures was REVERTED (it cost sample efficiency; see the
    module docstring). Keeping the candidate rule in the script is what makes the
    measurement re-runnable against the current, unmodified library.

    `_selfcheck()` asserts this reproduces `dimension.mean_dimension` bit-for-bit
    at `min_fit_length = 2` (the off switch), which is what licenses reading the
    `min_fit_length = 3` column as the effect of the floor and nothing else.
    """
    nodes = sorted(hg.nodes)
    if samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    n_nodes = len(hg.nodes)
    estimates = [
        local_dimension(hg, u, max_radius=max_radius, max_ball_fraction=max_ball_fraction)
        for u in nodes
    ]
    consensus = near_constant_consensus(estimates, threshold=0.9)
    accepted = []
    for estimate in estimates:
        long_enough = min_fit_length <= 2 or (
            _fit_length(estimate, n_nodes, max_ball_fraction) >= min_fit_length
        )
        confident = estimate.r_squared >= 0.9 and long_enough
        if confident or (estimate.near_degenerate and consensus):
            accepted.append(estimate.dimension)
    if not accepted:
        return float("nan")
    return sum(accepted) / len(accepted)


def _poly_trace(problem: str, theiler, fraction: float, min_fit_length: int):
    """The per-n dimension trace `poly_algebraic_minimum_points` searches over."""
    spec = SPECS[problem]
    points = load_cloud(problem)
    out: list[tuple[int, float]] = []
    for n in spec.n_grid:
        if n > len(points) or spec.k >= n:
            break
        try:
            hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window=theiler)
        except DuplicatePointsError:
            continue
        except ValueError:
            out.append((n, float("nan")))
            continue
        out.append(
            (
                n,
                _mean_dimension(
                    hg,
                    samples=min(n, spec.samples),
                    max_radius=spec.max_radius,
                    max_ball_fraction=fraction,
                    min_fit_length=min_fit_length,
                ),
            )
        )
    return out


def _poly_min_n(problem: str, theiler, fraction: float, min_fit_length: int) -> int | None:
    """`comparison.poly_algebraic_minimum_points`' stable-convergence rule, verbatim."""
    spec = SPECS[problem]
    results = _poly_trace(problem, theiler, fraction, min_fit_length)
    for i, (_, dim) in enumerate(results):
        if not math.isfinite(dim) or abs(dim - spec.true_dimension) > spec.tolerance:
            continue
        if all(
            math.isfinite(d) and abs(d - spec.true_dimension) <= spec.tolerance
            for _, d in results[i:]
        ):
            return results[i][0]
    return None


def _trad_min_n(problem: str, theiler) -> int | None:
    spec = SPECS[problem]
    return minimum_points_for_target_accuracy(
        load_cloud(problem),
        spec.true_dimension,
        spec.tolerance,
        n_grid=spec.n_grid,
        theiler_window=theiler,
    )


def _selfcheck() -> None:
    """The floor OFF must reproduce the library exactly, on both chaotic clouds."""
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        for n in (200, 800):
            hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window="auto")
            mine = _mean_dimension(
                hg,
                samples=min(n, spec.samples),
                max_radius=spec.max_radius,
                max_ball_fraction=SATURATION_BALL_FRACTION,
                min_fit_length=2,
            )
            theirs = mean_dimension(
                hg,
                samples=min(n, spec.samples),
                max_radius=spec.max_radius,
                max_ball_fraction=SATURATION_BALL_FRACTION,
            )
            assert mine == theirs, (problem, n, mine, theirs)
    print("self-check: min_fit_length=2 reproduces dimension.mean_dimension exactly.\n")


# --- section 1 ---------------------------------------------------------------


def section1() -> None:
    print("=" * 78)
    print("SECTION 1 -- the defect: zero-residual-dof fits behind the accepted nodes")
    print("=" * 78)
    print(
        "For each n, the fit-window LENGTH of every node `mean_dimension` accepted.\n"
        "Length 2 means the log-log fit had 2 points and therefore ZERO residual\n"
        "degrees of freedom: r_squared is exactly 1.0 by algebra, whatever the two\n"
        "shell counts are, so such a node passes any R^2 threshold by construction.\n"
    )
    configs = [
        ("08", "auto", SATURATION_BALL_FRACTION),
        ("09", "auto", SATURATION_BALL_FRACTION),
        ("02", 0, SATURATION_BALL_FRACTION),
        ("10", 0, SATURATION_BALL_FRACTION),
        ("02", 0, 1.0),
        ("10", 0, 1.0),
    ]
    for problem, theiler, fraction in configs:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(
            f"problem {problem} ({spec.label}): k={spec.k}, max_radius={spec.max_radius}, "
            f"theiler={theiler!r}, max_ball_fraction={fraction}"
        )
        for n in spec.n_grid[:5]:
            if n > len(points) or spec.k >= n:
                break
            try:
                hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window=theiler)
            except (DuplicatePointsError, ValueError) as exc:
                print(f"  {n:>6} | (no graph: {type(exc).__name__})")
                continue
            n_nodes = len(hg.nodes)
            nodes = sorted(hg.nodes)
            step = max(1, len(nodes) // spec.samples)
            nodes = nodes[::step][: spec.samples]
            estimates = [
                local_dimension(hg, u, max_radius=spec.max_radius, max_ball_fraction=fraction)
                for u in nodes
            ]
            consensus = near_constant_consensus(estimates, threshold=0.9)
            accepted = [
                e
                for e in estimates
                if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
            ]
            lengths = Counter(_fit_length(e, n_nodes, fraction) for e in accepted)
            vacuous = sum(count for length, count in lengths.items() if length <= 2)
            mean = (
                sum(e.dimension for e in accepted) / len(accepted) if accepted else float("nan")
            )
            print(
                f"  {n:>6} | accepted {len(accepted):>2}/{len(estimates):<2} mean={mean:>7.3f}  "
                f"fit lengths {dict(sorted(lengths.items()))}  "
                f"ZERO-DOF ACCEPTED = {vacuous}"
            )
        print()


# --- section 2 ---------------------------------------------------------------


def _fit(radii: list[float], shells: list[float]) -> tuple[float, float]:
    n = len(radii)
    log_x = [math.log(x) for x in radii]
    log_y = [math.log(y) for y in shells]
    mean_x, mean_y = sum(log_x) / n, sum(log_y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(log_x, log_y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in log_x)
    if var_x == 0:
        return 0.0, 0.0
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((b - mean_y) ** 2 for b in log_y)
    ss_res = sum(
        (b - (slope * a + intercept)) ** 2 for a, b in zip(log_x, log_y, strict=True)
    )
    scale = max(1.0, sum(b * b for b in log_y))
    r_squared = 1.0 if ss_tot <= 1e-24 * scale else 1.0 - ss_res / ss_tot
    return slope + 1.0, r_squared


def section2() -> None:
    print("=" * 78)
    print("SECTION 2 -- Lorenz n=400: every contiguous fit window, all nodes")
    print("=" * 78)
    spec = SPECS["08"]
    lo = spec.true_dimension - spec.tolerance
    print(
        f"AR2 left problem 08 blocked by a single nan at n=400. This scans EVERY\n"
        f"contiguous window [a,b] of radii within 1..10, on all nodes of the n=400\n"
        f"graph, and reports what any fit-window or acceptance rule could possibly\n"
        f"read off it. Tolerance band is [{lo:.2f}, {spec.true_dimension + spec.tolerance:.2f}].\n"
    )
    r_max = 10
    points = load_cloud("08")
    hg = knn_hypergraph(points[:400], k=spec.k, dedupe=True, theiler_window="auto")
    adjacency = hg.adjacency()
    sequences = []
    for source in sorted(hg.nodes):
        volumes = [1]
        visited = {source}
        frontier = {source}
        for _ in range(r_max):
            nxt: set = set()
            for u in frontier:
                nxt |= adjacency.get(u, set()) - visited
            visited |= nxt
            volumes.append(len(visited))
            if not nxt:
                break
            frontier = nxt
        sequences.append([volumes[i] - volumes[i - 1] for i in range(1, len(volumes))])

    print(f"  {len(sequences)} nodes, N={len(hg.nodes)}")
    print("  window  len   min_dim  max_dim  mean_dim   #nodes>=lo   #r2>=0.9")
    for a in range(1, r_max + 1):
        for b in range(a + 1, r_max + 1):
            dims, r_squareds = [], []
            for shells in sequences:
                if b > len(shells):
                    continue
                window = shells[a - 1 : b]
                if min(window) <= 0:
                    continue
                dim, r_squared = _fit(
                    [float(r) for r in range(a, b + 1)], [float(s) for s in window]
                )
                dims.append(dim)
                r_squareds.append(r_squared)
            if len(dims) < 20:
                continue
            print(
                f"  [{a},{b}]{'':>3} {b - a + 1:>3}  {min(dims):8.3f} {max(dims):8.3f} "
                f"{sum(dims) / len(dims):9.3f} {sum(1 for d in dims if d >= lo):>12} "
                f"{sum(1 for r in r_squareds if r >= 0.9):>10}"
            )
    print(
        "\n  Reading. EVERY length-2 window scores r_squared >= 0.9 on all 400 nodes --\n"
        "  that is the zero-residual-dof identity, visible directly, and it is the only\n"
        "  way anything here reaches the tolerance band.\n"
        "\n"
        "  Windows WITH a residual degree of freedom, measured over all 400 nodes:\n"
        "    [1,3]  mean over all nodes 1.586, but only 50/400 reach r_squared 0.9 and\n"
        "           the mean of exactly those 50 is 1.5166 -- OUTSIDE the band.\n"
        "    [1,4]  mean 1.471; 6/400 confident, their mean 1.3717 -- outside.\n"
        "    [1,5]  mean 1.350; 0/400 confident.\n"
        "    [1,6]  mean 1.309; 0/400 confident. (This is the production window.)\n"
        "  Every window not starting at radius 1 is worse still (means 0.48-1.51),\n"
        "  because the local slope only decays with radius on this graph -- there is no\n"
        "  plateau anywhere, i.e. no scaling region.\n"
        "\n"
        "  So no fit-window rule and no R^2-acceptance rule can put Lorenz n=400 in\n"
        f"  tolerance [{lo:.2f}, {spec.true_dimension + spec.tolerance:.2f}] non-vacuously."
        " nan is the honest answer at n=400, and\n"
        "  poly_algebraic_min_n = 800 is the honest number for problem 08.\n"
    )


# --- section 3 ---------------------------------------------------------------


def section3() -> None:
    print("=" * 78)
    print("SECTION 3 -- before/after on the two chaotic attractors")
    print("=" * 78)
    print(
        "Theiler window 'auto' throughout (the AR1 corrected setting). The poly side\n"
        "is this script's own aggregator, self-checked against the library at\n"
        "min_fit_length=2; the traditional side is baseline.minimum_points_for_target_\n"
        "accuracy, untouched by this change.\n"
    )
    _selfcheck()
    for problem in CHAOTIC:
        spec = SPECS[problem]
        print(
            f"problem {problem} ({spec.label}): k={spec.k}, max_radius={spec.max_radius}, "
            f"true={spec.true_dimension}, tol={spec.tolerance}"
        )
        trad_n = _trad_min_n(problem, "auto")
        for fraction in (1.0, SATURATION_BALL_FRACTION):
            for min_fit_length in (2, MIN_FIT_LENGTH):
                trace = _poly_trace(problem, "auto", fraction, min_fit_length)
                poly_n = _poly_min_n(problem, "auto", fraction, min_fit_length)
                cells = []
                for n, dim in trace:
                    if not math.isfinite(dim):
                        cells.append(f"{n}:nan")
                    else:
                        mark = "*" if abs(dim - spec.true_dimension) <= spec.tolerance else " "
                        cells.append(f"{n}:{dim:.3f}{mark}")
                wins = poly_n is not None and trad_n is not None and poly_n < trad_n
                savings = (
                    None if poly_n is None or trad_n is None else 1.0 - poly_n / trad_n
                )
                label = "OFF" if min_fit_length <= 2 else str(min_fit_length)
                print(
                    f"  frac={fraction:<4} min_fit_length={label:<3} -> "
                    f"poly_min_n={str(poly_n):>5}  trad_min_n={str(trad_n):>5}  "
                    f"wins={wins}  savings={savings}"
                )
                print("      trace: " + " ".join(cells))
        print()


# --- section 4 ---------------------------------------------------------------


def section4() -> None:
    print("=" * 78)
    print("SECTION 4 -- regression spot check")
    print("=" * 78)
    print("(a) DEFAULT settings must reproduce the recorded numbers on every problem.")
    ok = True
    for problem, recorded in RECORDED_DEFAULT_MIN_N.items():
        got = _poly_min_n(problem, 0, 1.0, 2)
        flag = "OK" if got == recorded else "REGRESSED"
        ok &= got == recorded
        print(f"  problem {problem}: recorded {recorded:>5}  got {str(got):>5}   {flag}")
    print(f"  -> defaults unchanged: {ok}\n")

    print("(b) problems 02 and 10 must stay at poly_algebraic_min_n = 64.")
    for problem in SPOT_CHECK:
        trad_n = _trad_min_n(problem, 0)
        for fraction in (1.0, SATURATION_BALL_FRACTION):
            for min_fit_length in (2, MIN_FIT_LENGTH):
                poly_n = _poly_min_n(problem, 0, fraction, min_fit_length)
                wins = poly_n is not None and trad_n is not None and poly_n < trad_n
                savings = (
                    None if poly_n is None or trad_n is None else 1.0 - poly_n / trad_n
                )
                label = "OFF" if min_fit_length <= 2 else str(min_fit_length)
                print(
                    f"  problem {problem} frac={fraction:<4} min_fit_length={label:<3}: "
                    f"poly={str(poly_n):>5} trad={str(trad_n):>5} "
                    f"wins={wins} savings={savings}"
                )
    print()


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", default="1,2,3,4")
    args = parser.parse_args()
    print(f"MIN_R2_FIT_LENGTH candidate = {MIN_FIT_LENGTH} (2 = off)")
    print(f"SATURATION_BALL_FRACTION    = {SATURATION_BALL_FRACTION}\n")
    for token in args.sections.split(","):
        SECTIONS[int(token.strip())]()


if __name__ == "__main__":
    main()
