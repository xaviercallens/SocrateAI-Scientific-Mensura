"""Round 3 remeasurement of problem 02 (nonlinear pendulum) against current code.

WHY THIS SCRIPT EXISTS
----------------------
Problem 02's round-2 result (poly_algebraic_min_n=64 vs traditional_min_n=256,
compute savings 0.750) was recorded before N8 (near-constant shell gate), H1
(Theiler exclusion window) and H3 (criterion (c), asymptotic accuracy) landed.
This script re-derives it against the current, unmodified library and adds the
criterion-(c) measurement that did not exist in round 2.

POINT CLOUD: REUSED, NOT REBUILT
--------------------------------
The point cloud is produced by importing
scripts/hypergraph_benchmark/round2/02_nonlinear_pendulum.py as a module and
calling its own `verify_solver()`, `generate_single_period_cloud()` and
`bit_reversal_permutation()` -- byte-for-byte the round-2 construction, with no
re-derivation and no reimplementation here. Section 1 additionally asserts the
resulting cloud is element-for-element identical to the disk-cached cloud that
every round-3 script uses (`n8d_clouds.load_cloud("02")`), so this script and
the rest of round 3 are demonstrably measuring the same object.

Construction, in one sentence: exactly ONE period of theta'' = -sin(theta),
theta(0)=2.0, thetadot(0)=0, integrated with leapfrog at dt = exact_elliptic_
period / 4096, sampled time-uniformly (so the physical density variation near
the turning points is intact, never arc-length-resampled), then bit-reversal
permuted so that `points[:n]` at every power-of-two n is an evenly-time-spaced
sample of the WHOLE closed orbit rather than a truncated arc.

PARAMETERS: MODULE DEFAULTS, NO DEVIATION
-----------------------------------------
k=6, max_radius=6, samples=40 -- all module defaults, and identical to round 2.
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.4 (finding R2-F3) traced a false round-2
win to an undisclosed non-default max_radius; there is no deviation here to
disclose. `max_ball_fraction` is left at its default 1.0 (off) for the headline
numbers, with the 0.5 setting reported alongside as a sensitivity check only.

WHAT H1 MEANS FOR A SINGLE-PERIOD CLOSED ORBIT (read before section 3)
---------------------------------------------------------------------
The Theiler window exists to stop temporally adjacent samples of a *chaotic*
trajectory from being counted as independent spatial neighbours. On this cloud
that argument does not transfer, and section 3 measures the consequence rather
than asserting it: the target here IS a closed 1-D curve traversed once, so the
temporally-adjacent pairs are not a redundant sampling artifact -- they are the
curve. Excluding them at small r removes precisely the pairs that carry the
1-D scaling. Section 3 therefore reports the window's effect as a disclosed
sensitivity, and the headline criterion-(a) number stays at the module default
theiler_window=0, which is also the setting every recorded round-2 number for
this problem was produced under.

The window is applied with `time_indices=perm` (perm = the bit-reversal
permutation), i.e. real trajectory indices in units of the finest dt, NOT the
storage order of the permuted array -- passing the default t_i = i on a
bit-reversal-permuted cloud would apply the correction to a meaningless
"adjacency". Note the documented caveat on `theiler_window_from_autocorrelation`:
each prefix is evenly sampled but with stride 4096/n, so "auto" returns a window
in units of prefix samples which is then applied as a time-index difference.
Section 3 therefore reports "auto" AND explicit integer windows measured on the
finest sampling, so the reader can see the whole curve.

THE F1 DEGENERACY CAVEAT TRAVELS WITH EVERY NUMBER BELOW
--------------------------------------------------------
Section 2 measures degenerate_fraction = 1.000 at every n from 64 to 4096: this
cloud is a single traversal of a closed curve, so its k-NN graph is a ring
lattice and the shell sequence is EXACTLY constant, which routes every sampled
node into `dimension.py`'s degenerate branch and forces the answer 1.0000. The
criterion-(a) win is therefore a statement about HOW FEW POINTS EACH METHOD
NEEDS TO IDENTIFY A RING (poly n=64, correlation sum n=256), not a claim that
the shell-growth estimator is more accurate here -- exactly the narrow reading
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.7 already records for this problem. It
is not vacuous in the "would say 1.0 for anything" sense: Sec. 10.3's control
shows poly returning 1.32-2.47 (degenerate fraction 0.000) on 2-D and 3-D
clouds at these same n. Section 4 shows criterion (c) refusing this same cloud
outright on that degeneracy, which is the criterion working as designed.

RUN
    python scripts/hypergraph_benchmark/round3/02_nonlinear_pendulum.py
    (sections 1-5, ~4 min; --sections to select)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.comparison import (  # noqa: E402
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_MAX_SENTINEL_FRACTION,
    compare,
    compare_accuracy_at_max_n,
)
from socrates.hypergraph.dimension import (  # noqa: E402
    degenerate_fraction,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402

ROUND2 = REPO / "scripts" / "hypergraph_benchmark" / "round2"

# --- Problem 02's production configuration, copied from the round-2 script ---
TRUE_DIMENSION = 1.0
TOLERANCE = 0.15
N_GRID: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)
K = 6  # module default
MAX_RADIUS = 6  # module default
SAMPLES = 40  # module default
M = N_GRID[-1]


def _import_round2_02():
    """Import the round-2 script unchanged (it is main-guarded)."""
    path = ROUND2 / "02_nonlinear_pendulum.py"
    spec = importlib.util.spec_from_file_location("round2_p02", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# SECTION 1 -- cloud provenance
# =============================================================================
def section1() -> dict[str, object]:
    print("=" * 78)
    print("SECTION 1 -- point cloud provenance (round-2 construction, reused unchanged)")
    print("=" * 78)

    m = _import_round2_02()
    print(f"  imported: {ROUND2 / '02_nonlinear_pendulum.py'}")
    print(f"  round-2 module constants: THETA0={m.THETA0} K={m.K} MAX_RADIUS={m.MAX_RADIUS} "
          f"N_GRID={m.N_GRID}")
    assert (m.K, m.MAX_RADIUS, m.N_GRID) == (K, MAX_RADIUS, N_GRID), "config drift vs round 2"

    ver = m.verify_solver()
    print("\n  --- solver gates (round-2 code, re-run) ---")
    print(f"    exact elliptic period 4*K(m): {ver['exact_period']:.10f}")
    print(f"    measured period:              {ver['measured_period']:.10f}")
    print(f"    period relative error:        {ver['period_relative_error']:.3e} "
          f"(gate < {m.PERIOD_REL_ERR_GATE:.0e})")
    print(f"    one-period return error:      {ver['return_error']:.3e} "
          f"(gate < {m.RETURN_ERR_GATE:.0e})")
    print(f"    SOLVER VERIFIED: {ver['passed']}")
    if not ver["passed"]:
        raise RuntimeError("solver verification failed; refusing to trust the point cloud")

    cloud = m.generate_single_period_cloud()
    if not cloud["cloud_closure_passed"]:
        raise RuntimeError("single-period cloud did not close cleanly")
    points = cloud["bitrev_ordered"]
    print("\n  --- single-period cloud ---")
    print(f"    M = {M} points, dt = {cloud['dt']:.6e}, "
          f"closure error {cloud['return_error']:.3e} "
          f"(gate < {m.CLOUD_RETURN_ERR_GATE:.0e}) -> {cloud['cloud_closure_passed']}")
    print("    bit-reversal prefix-coverage property: VERIFIED inside the round-2 code")

    perm = m.bit_reversal_permutation(M.bit_length() - 1)
    m.check_bitreversal_prefix_coverage(perm, M)
    time_ordered = cloud["time_ordered"]
    assert all(points[i] == time_ordered[perm[i]] for i in range(M)), (
        "perm is not the permutation actually used to build bitrev_ordered"
    )
    print("    time_indices = perm: VERIFIED (points[i] == time_ordered[perm[i]] for all i)")

    # Cross-check against the cloud every other round-3 script uses.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import n8d_clouds  # noqa: PLC0415

    cached = n8d_clouds.load_cloud("02")
    same = len(cached) == len(points) and all(
        tuple(a) == tuple(b) for a, b in zip(cached, points, strict=True)
    )
    print(f"    identical to n8d_clouds.load_cloud('02'): {same}  "
          f"({len(cached)} vs {len(points)} points)")
    assert same, "round-3 cached cloud differs from the round-2 construction"

    print("\n  SECTION 1 PASSED: the cloud below is the round-2 cloud, unmodified.\n")
    return {"points": points, "time_indices": perm, "time_ordered": time_ordered}


# =============================================================================
# SECTION 2 -- criterion (a) at module defaults
# =============================================================================
def _per_n_table(points, *, theiler_window=0, time_indices=None, max_ball_fraction=1.0):
    rows = []
    for n in N_GRID:
        if n > len(points) or n <= K:
            continue
        sub = points[:n]
        ti = None if time_indices is None else time_indices[:n]
        try:
            hg = knn_hypergraph(sub, k=K, dedupe=True, theiler_window=theiler_window,
                                time_indices=ti)
            poly = mean_dimension(hg, samples=min(n, SAMPLES), max_radius=MAX_RADIUS,
                                  max_ball_fraction=max_ball_fraction)
            deg = degenerate_fraction(hg, samples=min(n, SAMPLES), max_radius=MAX_RADIUS,
                                      max_ball_fraction=max_ball_fraction)
            near = near_degenerate_fraction(hg, samples=min(n, SAMPLES), max_radius=MAX_RADIUS,
                                            max_ball_fraction=max_ball_fraction)
        except (DuplicatePointsError, ValueError) as exc:
            poly, deg, near = float("nan"), float("nan"), float("nan")
            print(f"    n={n}: poly side unavailable ({type(exc).__name__}: {exc})")
        try:
            est = correlation_dimension(sub, theiler_window=theiler_window, time_indices=ti)
            trad, r2, win = est.dimension, est.r_squared, est.theiler_window
        except ValueError as exc:
            trad, r2, win = float("nan"), 0.0, -1
            print(f"    n={n}: traditional side unavailable ({exc})")
        rows.append((n, poly, deg + near, trad, r2, win))
    print(f"    {'n':>6} {'poly':>9} {'sentinel':>9} {'trad':>9} {'trad R^2':>9} {'W':>5}"
          f"   in-tol(poly/trad)")
    for n, poly, sent, trad, r2, win in rows:
        pin = "Y" if abs(poly - TRUE_DIMENSION) <= TOLERANCE else "n"
        tin = "Y" if abs(trad - TRUE_DIMENSION) <= TOLERANCE else "n"
        print(f"    {n:>6} {poly:>9.4f} {sent:>9.3f} {trad:>9.4f} {r2:>9.4f} {win:>5}"
              f"        {pin} / {tin}")
    return rows


def section2(points) -> dict[str, object]:
    print("=" * 78)
    print("SECTION 2 -- criterion (a): compare() at MODULE DEFAULTS")
    print("=" * 78)
    print(f"  true_dimension={TRUE_DIMENSION} tolerance={TOLERANCE} k={K} "
          f"max_radius={MAX_RADIUS} samples={SAMPLES}")
    print(f"  n_grid={N_GRID}  theiler_window=0 (default)  max_ball_fraction=1.0 (default)\n")

    print("  --- per-n estimates from both sides (same prefixes compare() uses) ---")
    rows = _per_n_table(points)

    res = compare(points, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE, k=K,
                  n_grid=N_GRID, max_radius=MAX_RADIUS)
    print("\n  --- comparison.compare() ---")
    print(f"    poly_algebraic_min_n:         {res.poly_algebraic_min_n}")
    print(f"    traditional_min_n:            {res.traditional_min_n}")
    print(f"    poly_algebraic_wins:          {res.poly_algebraic_wins}")
    print(f"    compute_savings_fraction:     {res.compute_savings_fraction}")
    print(f"    poly estimate @ n={res.max_n_tested}: "
          f"{res.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"    trad estimate @ n={res.max_n_tested}: "
          f"{res.traditional_estimate_at_max_n:.4f}\n")

    # Sensitivity only: the AR2 saturation guard, reported not adopted.
    res_mbf = compare(points, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE, k=K,
                      n_grid=N_GRID, max_radius=MAX_RADIUS, max_ball_fraction=0.5)
    print(f"  sensitivity, max_ball_fraction=0.5: poly {res_mbf.poly_algebraic_min_n} "
          f"trad {res_mbf.traditional_min_n} savings {res_mbf.compute_savings_fraction}\n")
    return {"result": res, "rows": rows, "result_mbf05": res_mbf}


# =============================================================================
# SECTION 3 -- H1 sensitivity: does the Theiler window move traditional_min_n?
# =============================================================================
def section3(points, time_indices, time_ordered) -> dict[str, object]:
    print("=" * 78)
    print("SECTION 3 -- H1: Theiler window sensitivity (time_indices = perm)")
    print("=" * 78)

    w_auto_full = theiler_window_from_autocorrelation(time_ordered)
    print(f"  'auto' heuristic on the FULL time-ordered orbit (M={M}, stride 1): "
          f"W={w_auto_full}")
    print(f"    (max_window clamp = min(M//10, 250) = {min(M // 10, 250)}; a single-period "
          f"closed orbit\n     has no truly decorrelated lag, so a clamped answer is expected)")
    quarter = M // 4
    print(f"    for scale: quarter period = {quarter} steps, full period = {M} steps\n")

    settings: list[tuple[str, int | str]] = [
        ("0 (default)", 0),
        ("1 (<=1 = pre-H1 path)", 1),
        ("2", 2),
        ("4", 4),
        ("8", 8),
        ("16", 16),
        ("32", 32),
        ("128", 128),
        (f"{w_auto_full} (auto on full orbit)", w_auto_full),
        ("'auto' per prefix", "auto"),
    ]
    out = {}
    for label, w in settings:
        try:
            res = compare(points, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE, k=K,
                          n_grid=N_GRID, max_radius=MAX_RADIUS, theiler_window=w,
                          time_indices=None if w == 0 else time_indices)
            sav = res.compute_savings_fraction
            sav_s = "None" if sav is None else f"{sav:.3f}"
            print(f"  theiler_window={label:<28} poly {str(res.poly_algebraic_min_n):>5}  "
                  f"trad {str(res.traditional_min_n):>5}  "
                  f"wins {str(res.poly_algebraic_wins):>5}  savings {sav_s}")
            out[label] = res
        except ValueError as exc:
            print(f"  theiler_window={label:<28} FAILED: {exc}")
            out[label] = None

    print("\n  --- per-n table under theiler_window='auto' (time_indices=perm) ---")
    _per_n_table(points, theiler_window="auto", time_indices=time_indices)
    print()
    return out


# =============================================================================
# SECTION 4 -- criterion (c) via the H3 function
# =============================================================================
def section4(points) -> dict[str, object]:
    print("=" * 78)
    print("SECTION 4 -- criterion (c): compare_accuracy_at_max_n() (H3)")
    print("=" * 78)
    print(f"  margin={DEFAULT_ACCURACY_MARGIN} (default)  "
          f"max_sentinel_fraction={DEFAULT_MAX_SENTINEL_FRACTION} (default)")
    print("  NOTE: compare_accuracy_at_max_n accepts neither theiler_window nor "
          "max_ball_fraction\n        (documented capability gap); defaults only.\n")

    results = {}
    for n in (256, 1024, 4096):
        acc = compare_accuracy_at_max_n(points, TRUE_DIMENSION, n, k=K, tolerance=TOLERANCE,
                                        max_radius=MAX_RADIUS, samples=SAMPLES)
        print(f"  n={acc.n_evaluated} (poly graph nodes {acc.poly_algebraic_n_nodes})")
        print(f"    poly estimate {acc.poly_algebraic_estimate:.4f}  "
              f"abs error {acc.poly_algebraic_abs_error:.4f}")
        print(f"    trad estimate {acc.traditional_estimate:.4f}  "
              f"abs error {acc.traditional_abs_error:.4f}  (R^2 {acc.traditional_r_squared:.4f})")
        print(f"    sentinel fraction {acc.poly_algebraic_sentinel_fraction:.3f} "
              f"(degenerate {acc.poly_algebraic_degenerate_fraction:.3f} + near-degenerate "
              f"{acc.poly_algebraic_near_degenerate_fraction:.3f})")
        print(f"    more_accurate: {acc.more_accurate}")
        print(f"    poly_algebraic_accuracy_win: {acc.poly_algebraic_accuracy_win}")
        print(f"    traditional_accuracy_win:    {acc.traditional_accuracy_win}")
        print(f"    reason: {acc.verdict_reason}\n")
        results[n] = acc
    return results


# =============================================================================
# SECTION 5 -- is the criterion (a) win a property of the data or the knobs?
# =============================================================================
def section5(points) -> dict[str, object]:
    print("=" * 78)
    print("SECTION 5 -- criterion (a) robustness across k and max_radius (R2-F3 check)")
    print("=" * 78)
    print("  Reference point: docs Sec. 10.10 re-measured this sweep at 17/20 under the")
    print("  ROUND-2 library, the three losing cells being k=10 with max_radius in {5,6,8},")
    print("  which tied at 256=256. Watch those three cells specifically.\n")
    wins = 0
    total = 0
    cells = {}
    print(f"    {'k':>4} {'max_radius':>11} {'poly_n':>8} {'trad_n':>8} {'savings':>9}  win")
    for k in (4, 6, 8, 10):
        for mr in (4, 5, 6, 7, 8):
            res = compare(points, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE, k=k,
                          n_grid=N_GRID, max_radius=mr)
            sav = res.compute_savings_fraction
            total += 1
            win = bool(res.poly_algebraic_wins and sav is not None and sav > 0.10)
            wins += win
            cells[(k, mr)] = (res.poly_algebraic_min_n, res.traditional_min_n, sav, win)
            print(f"    {k:>4} {mr:>11} {str(res.poly_algebraic_min_n):>8} "
                  f"{str(res.traditional_min_n):>8} "
                  f"{'None' if sav is None else f'{sav:>9.3f}'}  {win}")
    print(f"\n  criterion (a) wins: {wins} / {total} (k, max_radius) combinations")
    print(f"  production cell (k={K}, max_radius={MAX_RADIUS}): {cells[(K, MAX_RADIUS)]}\n")
    return {"wins": wins, "total": total, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5")
    args = ap.parse_args()
    want = {s.strip() for s in args.sections.split(",") if s.strip()}

    s1 = section1()
    points, perm, time_ordered = s1["points"], s1["time_indices"], s1["time_ordered"]

    s2 = section2(points) if "2" in want else None
    if "3" in want:
        section3(points, perm, time_ordered)
    if "4" in want:
        section4(points)
    if "5" in want:
        section5(points)

    if s2 is not None:
        res = s2["result"]
        print("=" * 78)
        print("HEADLINE (module defaults, theiler_window=0, max_ball_fraction=1.0)")
        print("=" * 78)
        print(f"  poly_algebraic_min_n {res.poly_algebraic_min_n}  "
              f"traditional_min_n {res.traditional_min_n}  "
              f"savings {res.compute_savings_fraction}")


if __name__ == "__main__":
    main()
