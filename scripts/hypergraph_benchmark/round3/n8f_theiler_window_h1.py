"""H1 (round 3): Theiler-window fairness correction to the traditional baseline.

WHAT THIS SCRIPT IS FOR
-----------------------
`src/socrates/hypergraph/baseline.py` gained an optional Theiler (1986)
exclusion window. This script is the evidence run for that change. It answers
three questions, in order, and refuses to report the third if the first fails:

  GATE 0  Does the default still reproduce the pre-H1 estimator EXACTLY?
          Checked against a verbatim re-implementation of the pre-H1 code
          path inlined below -- not against the shipped function calling
          itself, which would be circular. Bit-for-bit, not approximately.
          Every traditional-method number already recorded in
          docs/MENSURA_BENCHMARK.md was produced with the window off.

  STEP 1  On the actual round-2 trajectory clouds (Lorenz dt=0.005 and
          Rossler, both full un-subsampled time-ordered trajectories -- the
          exact regime Theiler's paper is about), how much closer to the
          literature D2 does the window put the baseline?

  STEP 2  CONTROL. On a cloud with no temporal structure at all (i.i.d.
          uniform cube), does the window leave the answer alone? It must. A
          "correction" that raises the dimension estimate everywhere is not a
          correction, it is a thumb on the scale -- it would just as happily
          inflate a baseline that was already right.

THE DIRECTION OF THIS CHANGE IS THE POINT
-----------------------------------------
Theiler's bias is DOWNWARD: temporally adjacent pairs add a near-constant
offset to C(r), which flattens the small-r end of the log-log curve. Removing
them therefore RAISES the traditional estimate toward the truth and makes the
baseline HARDER for the shell-growth estimator to beat. Under this repo's
standing rule -- a superiority claim is only testable against a real baseline,
never a strawman -- a baseline that got better is the successful outcome of
this change, not a regression in it. STEP 3 states the resulting cost to the
poly-algebraic side explicitly rather than leaving it to be noticed later.

Nothing here is fitted. The window values reported are (a) 0, (b) whatever the
documented `"auto"` heuristic returns on that cloud with no per-cloud tuning,
and (c) a fixed sweep, shown so the reader can see the whole curve rather than
one flattering point on it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from scipy.spatial import cKDTree  # noqa: E402

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    minimum_points_for_target_accuracy,
    theiler_window_from_autocorrelation,
)

# --------------------------------------------------------------------------
# Settings copied from the round-2 scripts, unchanged, so the numbers below
# are comparable to what is already on the record.
# --------------------------------------------------------------------------
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT, LORENZ_T_TRANSIENT = 0.005, 10.0
LORENZ_TRUE_D2 = 2.05  # Grassberger & Procaccia 1983

ROSSLER_A, ROSSLER_B, ROSSLER_C = 0.2, 0.2, 5.7
ROSSLER_DT, ROSSLER_T_TRANSIENT, ROSSLER_STRIDE = 0.01, 100.0, 5
ROSSLER_TRUE_D2 = 2.01

N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
TOLERANCE = 0.5
WINDOW_SWEEP = (0, 5, 10, 20, 50, 100, 200)


# --------------------------------------------------------------------------
# GATE 0 -- verbatim pre-H1 estimator, transcribed from git history.
# --------------------------------------------------------------------------
def pre_h1_correlation_sums(
    points, *, n_radii: int = 20, r_min_frac: float = 0.01, r_max_frac: float = 0.2
):
    """The pre-H1 body of `correlation_dimension`, character-for-character.

    Kept as a separate copy on purpose: comparing the shipped function to
    itself with different arguments proves nothing about whether the recorded
    round-2 numbers still reproduce. This is the thing they were produced by.
    """
    arr = np.asarray(points, dtype=float)
    n = len(arr)
    tree = cKDTree(arr)
    diag = float(np.sqrt(np.sum((arr.max(axis=0) - arr.min(axis=0)) ** 2)))
    radii = np.logspace(math.log10(r_min_frac * diag), math.log10(r_max_frac * diag), n_radii)
    total_pairs = n * (n - 1) / 2
    sums = []
    for r in radii:
        count = tree.count_neighbors(tree, r) - n
        sums.append(count / 2 / total_pairs)
    return tuple(radii), tuple(sums)


# --------------------------------------------------------------------------
# Trajectory generators (identical physics to the round-2 scripts)
# --------------------------------------------------------------------------
def _rk4(deriv, state0, dt, n_steps):
    out = np.empty((n_steps + 1, len(state0)))
    out[0] = state0
    s = np.asarray(state0, dtype=float)
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def lorenz_cloud(n_points: int):
    def deriv(v):
        x, y, z = v
        return np.array(
            [
                LORENZ_SIGMA * (y - x),
                x * (LORENZ_RHO - z) - y,
                x * y - LORENZ_BETA * z,
            ]
        )

    transient_steps = int(LORENZ_T_TRANSIENT / LORENZ_DT)
    traj = _rk4(deriv, [1.0, 1.0, 1.0], LORENZ_DT, transient_steps + n_points + 10)
    post = traj[transient_steps:]
    assert np.all(np.isfinite(post)) and np.max(np.abs(post)) < 100.0, "Lorenz solver diverged"
    return [tuple(float(c) for c in p) for p in post[:n_points]]


def rossler_cloud(n_points: int):
    def deriv(v):
        x, y, z = v
        return np.array([-y - z, x + ROSSLER_A * y, ROSSLER_B + z * (x - ROSSLER_C)])

    transient_steps = int(ROSSLER_T_TRANSIENT / ROSSLER_DT)
    n_steps = transient_steps + n_points * ROSSLER_STRIDE + 10
    traj = _rk4(deriv, [1.0, 1.0, 1.0], ROSSLER_DT, n_steps)
    post = traj[transient_steps::ROSSLER_STRIDE]
    assert np.all(np.isfinite(post)), "Rossler solver diverged"
    return [tuple(float(c) for c in p) for p in post[:n_points]]


def iid_cube_cloud(n_points: int, seed: int = 20260813):
    rng = np.random.default_rng(seed)
    return [tuple(float(c) for c in p) for p in rng.uniform(0, 1, size=(n_points, 3))]


# --------------------------------------------------------------------------


def gate_0_exact_reproduction(clouds) -> bool:
    print("=" * 78)
    print("GATE 0 -- default must reproduce the pre-H1 estimator BIT-FOR-BIT")
    print("=" * 78)
    print("  (compared against an inlined verbatim copy of the pre-H1 code path,")
    print("   not against the shipped function calling itself)")
    all_ok = True
    for name, points in clouds.items():
        _, expected = pre_h1_correlation_sums(points)
        default = correlation_dimension(points)
        w0 = correlation_dimension(points, theiler_window=0)
        w1 = correlation_dimension(points, theiler_window=1)
        ok = (
            default.correlation_sums == expected
            and w0.correlation_sums == expected
            and w1.correlation_sums == expected
            and w0.dimension == default.dimension
            and w1.dimension == default.dimension
        )
        all_ok &= ok
        print(
            f"  {name:<22} n={len(points):>6}  dim={default.dimension:.6f}  "
            f"identical(default, W=0, W=1, pre-H1): {ok}"
        )
    print(f"\n  GATE 0 PASSED: {all_ok}")
    if not all_ok:
        print("  REFUSING to report any further numbers: the default silently moved.")
    return all_ok


def window_sweep(name, points, true_d2):
    print("\n" + "-" * 78)
    print(f"{name}: n={len(points)}, literature D2 = {true_d2}")
    print("-" * 78)
    auto_w = theiler_window_from_autocorrelation(points)
    print(f"  auto window (1/e decorrelation lag, no tuning): W = {auto_w}")
    print(f"  {'W':>6} {'dim':>9} {'|err|':>8} {'R^2':>8} {'pairs_dropped':>14} {'%dropped':>9}")
    total_pairs = len(points) * (len(points) - 1) / 2
    rows = {}
    for w in sorted({*WINDOW_SWEEP, auto_w}):
        est = correlation_dimension(points, theiler_window=w)
        err = abs(est.dimension - true_d2)
        rows[w] = (est.dimension, err, est.r_squared)
        tag = "  <- auto" if w == auto_w else ""
        print(
            f"  {w:>6} {est.dimension:>9.4f} {err:>8.4f} {est.r_squared:>8.4f} "
            f"{est.n_excluded_pairs:>14} {100 * est.n_excluded_pairs / total_pairs:>8.2f}%{tag}"
        )
    return auto_w, rows


def main() -> int:
    n_max = max(n for n in N_GRID if n <= 12800)
    print("Building point clouds (round-2 settings, unchanged) ...")
    lorenz = lorenz_cloud(n_max)
    rossler = rossler_cloud(n_max)
    cube = iid_cube_cloud(n_max)

    clouds = {"lorenz dt=0.005": lorenz, "rossler stride=5": rossler, "iid uniform cube": cube}
    if not gate_0_exact_reproduction(clouds):
        return 1

    print("\n" + "=" * 78)
    print("STEP 1 -- effect on the real trajectory clouds")
    print("=" * 78)

    lorenz_auto, lorenz_rows = window_sweep(
        "LORENZ (dt=0.005, full trajectory)", lorenz, LORENZ_TRUE_D2
    )
    rossler_auto, rossler_rows = window_sweep("ROSSLER (stride=5)", rossler, ROSSLER_TRUE_D2)

    print("\n" + "=" * 78)
    print("STEP 2 -- CONTROL: a cloud with NO temporal structure")
    print("=" * 78)
    print("  If the window moved this one, it would be inflating dimensions rather")
    print("  than removing an autocorrelation artifact. It must leave it alone.")
    cube_auto, cube_rows = window_sweep("I.I.D. UNIFORM CUBE (no trajectory at all)", cube, 3.0)
    cube_drift = abs(cube_rows[cube_auto][0] - cube_rows[0][0])
    control_ok = cube_auto == 1 or cube_drift < 0.02
    print(f"\n  auto window on an i.i.d. cloud: W={cube_auto} (1 == 'no correction needed')")
    print(f"  |dim(auto) - dim(W=0)| = {cube_drift:.6f}")
    print(f"  CONTROL PASSED (window does not invent dimension): {control_ok}")

    print("\n" + "=" * 78)
    print("STEP 3 -- what this costs the poly-algebraic side (stated, not buried)")
    print("=" * 78)
    print("  Points-to-converge for the TRADITIONAL method, window off vs auto.")
    print(f"  tolerance={TOLERANCE}, n_grid={N_GRID}. Lower = a harder baseline to beat.")
    print(f"\n  {'cloud':<22} {'true D2':>8} {'min_n W=0':>11} {'min_n auto':>11} {'verdict':>22}")
    harder_anywhere = False
    for name, pts, true_d2 in (
        ("lorenz", lorenz, LORENZ_TRUE_D2),
        ("rossler", rossler, ROSSLER_TRUE_D2),
    ):
        off = minimum_points_for_target_accuracy(pts, true_d2, TOLERANCE, n_grid=N_GRID)
        auto = minimum_points_for_target_accuracy(
            pts, true_d2, TOLERANCE, n_grid=N_GRID, theiler_window="auto"
        )
        if off is None and auto is not None:
            verdict = "baseline now converges"
        elif off is not None and auto is not None and auto < off:
            verdict = "baseline needs fewer n"
        elif off == auto:
            verdict = "unchanged"
        else:
            verdict = "baseline got no easier"
        harder_anywhere |= verdict in ("baseline now converges", "baseline needs fewer n")
        print(f"  {name:<22} {true_d2:>8.2f} {str(off):>11} {str(auto):>11} {verdict:>22}")

    # A single min_n number hides WHY it moved, and on Lorenz it moves the
    # "wrong" way for an instructive reason. Print the per-n estimates so the
    # reason is visible rather than inferred. The `fixed W` column separates
    # two candidate explanations: is the small-n behaviour caused by the auto
    # heuristic re-picking a window per prefix, or is it intrinsic to removing
    # the temporal pairs at all? If the two columns agree, it is intrinsic.
    print("\n  Per-n detail behind those min_n numbers (the interesting part):")
    for name, pts, true_d2 in (
        ("lorenz", lorenz, LORENZ_TRUE_D2),
        ("rossler", rossler, ROSSLER_TRUE_D2),
    ):
        full_w = theiler_window_from_autocorrelation(pts)
        lo, hi = true_d2 - TOLERANCE, true_d2 + TOLERANCE
        print(f"\n  {name} (in-tolerance band [{lo:.2f}, {hi:.2f}], full-cloud auto W={full_w})")
        print(
            f"  {'n':>7} {'auto W here':>12} {'dim W=0':>9} {'in band':>8} "
            f"{'dim auto':>9} {'in band':>8} {'dim fixed W':>12}"
        )
        for n in N_GRID:
            if n > len(pts):
                break
            sub = pts[:n]
            w_here = theiler_window_from_autocorrelation(sub)
            d0 = correlation_dimension(sub, theiler_window=0).dimension
            da = correlation_dimension(sub, theiler_window="auto").dimension
            df = correlation_dimension(sub, theiler_window=min(full_w, max(1, n // 2))).dimension
            print(
                f"  {n:>7} {w_here:>12} {d0:>9.4f} {str(lo <= d0 <= hi):>8} "
                f"{da:>9.4f} {str(lo <= da <= hi):>8} {df:>12.4f}"
            )
    print("\n  Read this before quoting min_n: an UNCORRECTED estimate can sit inside a")
    print("  wide tolerance band at small n *because* the autocorrelation artifact pins")
    print("  it there, and lose that stability once the artifact is removed. That is a")
    print("  spurious early convergence being taken away, not the baseline getting worse")
    print("  -- the same estimator is simultaneously far more accurate at large n.")

    print("\n" + "=" * 78)
    print("H1 VERDICT")
    print("=" * 78)
    lorenz_gain = lorenz_rows[0][1] - lorenz_rows[lorenz_auto][1]
    rossler_gain = rossler_rows[0][1] - rossler_rows[rossler_auto][1]
    print(
        f"  Lorenz  : |err| {lorenz_rows[0][1]:.4f} -> {lorenz_rows[lorenz_auto][1]:.4f} "
        f"(improvement {lorenz_gain:+.4f}) at auto W={lorenz_auto}"
    )
    print(
        f"  Rossler : |err| {rossler_rows[0][1]:.4f} -> {rossler_rows[rossler_auto][1]:.4f} "
        f"(improvement {rossler_gain:+.4f}) at auto W={rossler_auto}"
    )
    keep = (lorenz_gain > 0 or rossler_gain > 0) and control_ok
    print(f"\n  KEEP H1: {keep}")
    print("  Keep criterion for a FAIRNESS correction is not 'the new estimator won'.")
    print("  It is: (a) the default reproduces the record exactly [GATE 0], (b) the")
    print("  baseline gets measurably MORE accurate on trajectory data, and (c) it")
    print("  gets no free lift where there is no autocorrelation to remove [CONTROL].")
    if harder_anywhere:
        print("\n  NOTE: the baseline now converges sooner on at least one cloud.")
        print("  Any round-2 compute-savings claim on that cloud must be re-measured")
        print("  against the corrected baseline before it is repeated.")
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
