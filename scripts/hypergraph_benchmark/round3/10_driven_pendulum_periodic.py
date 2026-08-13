"""Round 3 remeasurement of problem 10 (driven damped pendulum, mode-locked
period-1) against the current, fully-updated library.

WHY THIS SCRIPT EXISTS
----------------------
Problem 10's round-2 result (poly_algebraic_min_n=64 vs traditional_min_n=256,
compute savings 0.750, 20/20 (k, max_radius) combinations winning) was recorded
BEFORE N8 (near-constant shell gate), H1 (Theiler exclusion window) and H3
(criterion (c), asymptotic accuracy) landed. Two of those three can in
principle move the recorded number:
  * N8 changes which shell sequences the poly side is allowed to call "1-D",
    so it can only move `poly_algebraic_min_n` UP (it is a gate, never a pass).
  * H1 adds a Theiler exclusion window to BOTH estimators, and the brief for
    this remeasurement names it explicitly as the thing that may move
    `traditional_min_n`.
  * H3 adds criterion (c), which did not exist when this problem was scored.
This script re-derives criterion (a) against the unmodified library, measures
the H1 sensitivity as a disclosed sweep rather than an assertion, and adds the
criterion-(c) measurement.

POINT CLOUD: REUSED, NOT REBUILT
--------------------------------
The point cloud is produced by importing
scripts/hypergraph_benchmark/round2/10_driven_pendulum_periodic.py as a module
and calling its own `verify_mode_locking()` and `generate_single_period_cloud()`
-- byte-for-byte the round-2 construction, no re-derivation and no
reimplementation here. Section 1 additionally asserts the resulting cloud is
element-for-element identical to the disk-cached cloud every other round-3
script uses (`n8d_clouds.load_cloud("10")`).

Construction, in one sentence: the driven damped pendulum
thetadot = omega, omegadot = -0.5*omega - sin(theta) + 0.5*cos(0.6 t) is
integrated with RK4 through 200 drive periods of transient, its period-1
mode-locking is re-verified stroboscopically over the next 50 periods (tail
spread gate 1e-6), then exactly ONE further drive period is sampled
time-uniformly at M=4096 steps (round 1's own documented fix for this
problem: sampling several periods of a genuinely period-1 orbit stacks
~exact duplicates at ~1e-9 separation instead of adding distinct points),
and the resulting closed curve is bit-reversal permuted so that `points[:n]`
at every power-of-two n is an evenly-time-spaced sample of the WHOLE closed
orbit rather than a truncated arc.

PARAMETERS: MODULE DEFAULTS, NO DEVIATION
-----------------------------------------
k=6, max_radius=6, samples=40 -- all module defaults, and identical to round 2.
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.4 (finding R2-F3) traced a false
round-2 win to an undisclosed non-default max_radius; there is NO deviation
here to disclose. `max_ball_fraction` is left at its default 1.0 (off) for the
headline numbers, with 0.5 reported alongside as a sensitivity check only.
`theiler_window` is left at its default 0 for the headline numbers, which is
also the setting every recorded round-2 number for this problem was produced
under; section 3 sweeps it.

WHAT H1 MEANS FOR A SINGLE-PERIOD CLOSED ORBIT (read before section 3)
---------------------------------------------------------------------
The Theiler window exists to stop temporally adjacent samples of a *chaotic*
trajectory from being counted as independent spatial neighbours. On this cloud
that argument does not transfer, and section 3 measures the consequence rather
than asserting it: the target here IS a closed 1-D curve traversed exactly once,
so temporally-adjacent pairs are not a redundant sampling artifact -- they ARE
the curve. Excluding them at small r removes precisely the pairs that carry the
1-D scaling. The window is applied with `time_indices=perm` (perm = the
bit-reversal permutation), i.e. real trajectory indices in units of the finest
dt, NOT the storage order of the permuted array.

THE F1 DEGENERACY CAVEAT TRAVELS WITH EVERY NUMBER BELOW
--------------------------------------------------------
Like problem 02, this cloud is a single traversal of a closed curve, so its
k-NN graph is (near-)exactly a ring lattice and the shell sequence is constant,
routing sampled nodes into dimension.py's degenerate branch. Section 2 measures
the sentinel fraction at every n instead of assuming it. Any criterion-(a) win
is therefore a statement about HOW FEW POINTS EACH METHOD NEEDS TO IDENTIFY A
RING, not a claim that the shell-growth estimator is more accurate here.
Section 4 shows what criterion (c) does with that same degeneracy.

RUN
    python scripts/hypergraph_benchmark/round3/10_driven_pendulum_periodic.py
    (sections 1-5, ~5 min; --sections to select)
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

# --- Problem 10's production configuration, copied from the round-2 script ---
TRUE_DIMENSION = 1.0
TOLERANCE = 0.2
N_GRID: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)
K = 6  # module default
MAX_RADIUS = 6  # module default
SAMPLES = 40  # module default
M = N_GRID[-1]


def _import_round2_10():
    """Import the round-2 script unchanged (it is main-guarded)."""
    path = ROUND2 / "10_driven_pendulum_periodic.py"
    spec = importlib.util.spec_from_file_location("round2_p10", path)
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

    m = _import_round2_10()
    print(f"  imported: {ROUND2 / '10_driven_pendulum_periodic.py'}")
    print(f"  round-2 module constants: GAMMA={m.GAMMA} F_DRIVE={m.F_DRIVE} "
          f"OMEGA_DRIVE={m.OMEGA_DRIVE}")
    print(f"                            K={m.K} MAX_RADIUS={m.MAX_RADIUS} N_GRID={m.N_GRID} "
          f"TOLERANCE={m.TOLERANCE}")
    assert (m.K, m.MAX_RADIUS, m.N_GRID, m.TOLERANCE) == (K, MAX_RADIUS, N_GRID, TOLERANCE), (
        "config drift vs round 2"
    )

    v = m.verify_mode_locking()
    print("\n  --- mode-locking gates (round-2 code, re-run) ---")
    print(f"    transient periods: {m.N_TRANSIENT_PERIODS}, check periods: {m.N_CHECK_PERIODS}")
    print(f"    finite state:                {v['finite_ok']}")
    print(f"    theta spread (tail half):    {v['theta_spread']:.3e}")
    print(f"    omega spread (tail half):    {v['omega_spread']:.3e}")
    print(f"    gate: spread < {m.PERIOD1_TOL:.0e}  ->  PERIOD-1 VERIFIED: {v['passed']}")
    if not v["passed"]:
        raise RuntimeError("mode-locking did not verify; refusing to trust the point cloud")

    cloud = m.generate_single_period_cloud(v["theta_final"], v["omega_final"], v["t_final"])
    if not cloud["cloud_closure_passed"]:
        raise RuntimeError("single-period cloud did not close cleanly")
    points = cloud["bitrev_ordered"]
    time_ordered = cloud["time_ordered"]
    print("\n  --- single-period cloud ---")
    print(f"    M = {M} points, dt = {cloud['dt']:.6e}")
    print(f"    closure error |state(t0+T)-state(t0)| = {cloud['closure_error']:.3e} "
          f"(gate < {m.CLOUD_CLOSURE_GATE:.0e}) -> {cloud['cloud_closure_passed']}")
    print(f"    theta range [{cloud['theta_min']:.4f}, {cloud['theta_max']:.4f}], "
          f"omega range [{cloud['omega_min']:.4f}, {cloud['omega_max']:.4f}]")

    perm = m.bit_reversal_permutation(M.bit_length() - 1)
    m.check_bitreversal_prefix_coverage(perm, M)
    assert all(points[i] == time_ordered[perm[i]] for i in range(M)), (
        "perm is not the permutation actually used to build bitrev_ordered"
    )
    print("    bit-reversal prefix-coverage property: VERIFIED (checked, not assumed)")
    print("    time_indices = perm: VERIFIED (points[i] == time_ordered[perm[i]] for all i)")

    # Cross-check against the cloud every other round-3 script uses.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import n8d_clouds  # noqa: PLC0415

    cached = n8d_clouds.load_cloud("10")
    same = len(cached) == len(points) and all(
        tuple(a) == tuple(b) for a, b in zip(cached, points, strict=True)
    )
    print(f"    identical to n8d_clouds.load_cloud('10'): {same}  "
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
    print(f"    for scale: quarter period = {M // 4} steps, full period = {M} steps\n")

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

    # n=64/128 are included deliberately: they are the only point counts on this
    # cloud where the BASELINE is genuinely outside tolerance (trad ~1.51, error
    # ~0.51), i.e. the only place a criterion-(c) win could even be argued for.
    # Showing the criterion refusing there too -- on the F1 degeneracy guard,
    # not on the baseline being accidentally right -- is the point.
    results = {}
    for n in (64, 128, 256, 1024, 4096):
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
    print("  Round 2 recorded 20/20 (k, max_radius) combinations winning for this problem.")
    print("  R2-F3 was an undisclosed max_radius deviation producing a false win elsewhere;")
    print("  this grid is exactly the check that the production cell is not special.\n")
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
