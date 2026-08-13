"""Round 3 remeasurement of benchmark problem 03 (Kepler two-body orbit, e=0.6)
against the CURRENT library state (post N8, H1, H3, and the AutoResearch H2 loop).

WHY THIS SCRIPT EXISTS AND WHAT IT DELIBERATELY DOES DIFFERENTLY
---------------------------------------------------------------
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 10.10 records that the round-2 measurement
of this problem (and of problem 04) was made on a TIME-ORDERED prefix of the
trajectory: `compare()` evaluates `points[:n]`, so with a time-ordered cloud the
n=100 grid point is the first ~1% of a single Kepler period -- a short, smooth,
low-curvature arc near perihelion, not the orbit. That construction was
diagnosed as unsound for exactly this problem, and the corrected construction
(a whole-orbit-covering low-discrepancy prefix: bit-reversal, or a golden-ratio
Weyl index sequence) is what problems 01/02/10 already used.

This script therefore uses whole-orbit-covering prefixes ONLY for its headline
numbers, and keeps the time-ordered cloud alive purely as a NEGATIVE CONTROL
(section 3) so the artifact is visible in the output rather than asserted from
the ledger.

Two independent whole-orbit constructions are measured, not one:

  ARM A  bit-reversal permutation of M = 8192 time-uniform samples of one
         period, on the power-of-two grid (32 ... 8192). This is the exact
         construction Sec. 10.2/10.10's correction used, so its numbers are
         directly comparable to the "poly 64 vs traditional 256, savings 0.75"
         already on the record (measured before H1).

  ARM B  golden-ratio Weyl index sequence over one period, on problem 03's
         PRODUCTION n_grid (100, 200, ... 6400) from the round-2 script. Arm B
         exists because arm A changes two things at once relative to round 2
         (the ordering AND the grid), and a win that only survives on a
         hand-picked grid would be measurement-config gaming. Arm B changes the
         ordering alone.

HYPERPARAMETERS: k=6, max_radius=6 -- the module defaults, and also problem 03's
round-2 production values (scripts/hypergraph_benchmark/round2/03_kepler_orbit.py
K=6, MAX_RADIUS=6). NO deviation. docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.4
(R2-F3) traced a false round-2 win to an undisclosed non-default `max_radius`;
nothing here is swept in search of a better answer. `theiler_window` and
`max_ball_fraction` are reported at their DEFAULTS (0 and 1.0) as the headline,
with the corrected settings shown alongside as robustness rows, never as the
quoted number.

WHAT IS MEASURED
  Criterion (a)  comparison.compare()                -> sections 2, 4
  Criterion (c)  comparison.compare_accuracy_at_max_n() -> section 5 (H3)

Sections are selectable: --sections 1,2,3,4,5 (default: all). Full run is a
few minutes (most of it is building arm B's 200k-step fine trajectory).

MEASURED RESULT (this script, current library: post N8 / H1 / H3 / AR H2 loop)
-----------------------------------------------------------------------------
Criterion (a), at the defaults, is a WIN and reproduces the round-2 correction
exactly:

    arm A (bitrev, pow2 grid)      poly 64  vs trad 256  -> savings 0.7500
    arm B (Weyl, production grid)  poly 100 vs trad 400  -> savings 0.7500

and it is not contingent on a hyperparameter (Sec. 9.4 / R2-F3's failure mode):
savings is 0.875/0.75/0.75/0.50 at k=4/6/8/10 and 0.875/0.875/0.75/0.75/0.75/
0.75 at max_radius=3..8 -- the DEFAULT max_radius=6 sits in the flat 0.75
plateau, so no non-default value is needed and none is used.

Criterion (c) is NOT a win, in either arm, and the reason is stated by the
library rather than by this script: the sentinel fraction is 1.000 (arm A) and
0.975 (arm B) against a 0.05 cap, i.e. finding F1 -- on a smooth closed 1-D
orbit the k-NN graph is a ring lattice and the shell-growth estimate is pinned
at 1.0 by the shell sequence rather than measured from it. Independently, the
correlation sum's own error at max n is 0.0047 (arm A) / 0.0055 (arm B), far
inside the 0.15 tolerance, so condition (c4) ("the baseline fails the same
bar") fails too. Two independent reasons; neither is close.

That F1 caveat also qualifies the criterion-(a) win and is not cancelled by it:
poly reaches tolerance 4x sooner because a ring lattice answers 1.0 structurally
and the orbit really is 1-D, not because a fit measured 1.0 from data. This is
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 10.10's "same win, four times over"; it is
recorded here rather than left for a reader to rediscover.

WHY `theiler_window` STAYS AT ITS DEFAULT 0 HERE (measured, not assumed)
-----------------------------------------------------------------------
Section 4 runs it anyway: `"auto"` resolves to 250 on arm A's 8192-sample single
period (6/25/102/250 at n=64/256/1024/8192) and destroys BOTH estimators -- poly
2.0771, trad 4.2187, and neither side converges anywhere on the grid, so there
is no win in either direction. That is the correct behaviour, not a bug. A
Theiler window exists to drop pairs that are close only because they are
temporally adjacent on a trajectory that revisits a neighbourhood at many
separated times. A single period of a closed orbit visits each neighbourhood
EXACTLY ONCE, so for this cloud temporal proximity and spatial proximity are the
same fact, and excluding a +-250-sample window removes the geometry instead of
an artifact (250 samples is ~0.1 in arc length near aphelion, ~250x the
nearest-neighbour spacing). Reported, not hidden; the headline stays at the
library default.

Modifies nothing under src/ or tests/.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.comparison import (  # noqa: E402
    compare,
    compare_accuracy_at_max_n,
)
from socrates.hypergraph.dimension import (  # noqa: E402
    degenerate_fraction,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

# --- problem 03 constants, copied verbatim from the round-2 script -----------
GM = 1.0
ECCENTRICITY = 0.6
TRUE_DIMENSION = 1.0
TOLERANCE = 0.15
K = 6
MAX_RADIUS = 6
PRODUCTION_N_GRID = (100, 200, 400, 800, 1600, 3200, 6400)

# --- arm A construction constants -------------------------------------------
M = 8192  # time-uniform samples of ONE period; power of two for bit reversal
POW2_N_GRID = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DT_FINE = 5e-4  # round-2 DT, used for the solver-verification run only


# =============================================================================
# Physics + solver verification (unchanged checks from round 2 / solver_ladder)
# =============================================================================
def force(q: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(q))
    return -GM * q / r**3


def exact_period() -> float:
    """Semi-major axis a = 1 by construction below, GM = 1, so T = 2*pi."""
    return 2.0 * math.pi


def run_kepler(n_periods: float, dt: float):
    r_peri = 1.0 - ECCENTRICITY
    v_peri = math.sqrt(GM * (1 + ECCENTRICITY) / r_peri)
    n = int(round(n_periods * exact_period() / dt))
    return leapfrog(force, [r_peri, 0.0], [0.0, v_peri], dt=dt, n_steps=n)


def verify_solver() -> dict[str, object]:
    """Energy drift, angular-momentum drift, and measured radial period against
    the exact Keplerian 2*pi. Gates are round 2's, unchanged. Run on a separate
    6-period throwaway trajectory (a single period yields one perihelion minimum,
    too few to average return-to-return diffs)."""
    run = run_kepler(n_periods=6.0, dt=DT_FINE)
    drift = run.energy_drift(lambda q: -GM / float(np.linalg.norm(q)))
    angular = (
        run.positions[:, 0] * run.velocities[:, 1] - run.positions[:, 1] * run.velocities[:, 0]
    )
    l_drift = float(np.max(np.abs(angular - angular[0])) / abs(angular[0]))

    radius = np.linalg.norm(run.positions, axis=1)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    dt = float(run.times[1] - run.times[0])
    t_minima = []
    for i in minima:
        y0, y1, y2 = radius[i - 1], radius[i], radius[i + 1]
        t_minima.append(run.times[i] + dt * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    measured_period = float(np.mean(np.diff(t_minima))) if len(t_minima) > 2 else float("nan")
    period_err = abs(measured_period - exact_period()) / exact_period()

    passed = drift < 1e-5 and l_drift < 1e-9 and period_err < 1e-5
    return {
        "energy_drift": drift,
        "angular_momentum_drift": l_drift,
        "measured_period": measured_period,
        "period_relative_error": period_err,
        "r_min": float(radius.min()),
        "r_max": float(radius.max()),
        "n_perihelion_returns": len(t_minima),
        "passed": bool(passed),
    }


# =============================================================================
# Point-cloud constructions
# =============================================================================
def bit_reversal_permutation(bits: int) -> list[int]:
    size = 1 << bits
    return [int(format(i, f"0{bits}b")[::-1], 2) for i in range(size)]


def check_bitreversal_prefix_coverage(perm: list[int], size: int) -> None:
    """VERIFY (do not assume) that every power-of-two prefix of the permuted
    index sequence is an exactly evenly-strided subset of one whole period."""
    bits = size.bit_length() - 1
    for b in range(bits + 1):
        n = 1 << b
        prefix = sorted(perm[:n])
        stride = size // n
        expected = list(range(0, size, stride))
        assert prefix == expected, (
            f"bit-reversal prefix coverage FAILED at n={n}: {prefix[:5]}... "
            f"vs expected {expected[:5]}..."
        )


def one_period_samples(m: int) -> np.ndarray:
    """m time-uniform (x, y) samples of exactly one period, endpoint excluded.

    Endpoint exclusion is not cosmetic: t=0 and t=T are the same physical point
    on a closed orbit, and including both is finding F3/N7's near-duplicate
    cluster by construction.
    """
    dt = exact_period() / m
    run = run_kepler(n_periods=1.0, dt=dt)
    return np.asarray(run.positions[:m], dtype=float)


def weyl_indices(n_max: int, n_steps_total: int) -> np.ndarray:
    """Golden-ratio Weyl index sequence; every PREFIX is a near-uniform sample of
    the whole period, so `points[:n]` covers the orbit at every n."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fracs = (np.arange(n_max) * phi) % 1.0
    idx = np.floor(fracs * n_steps_total).astype(int)
    if len(np.unique(idx)) != n_max:
        raise RuntimeError("Weyl index sequence collided; increase the time resolution")
    return idx


def build_arms() -> dict[str, object]:
    """Arm A (bit reversal, M=8192), arm B (Weyl, production grid), and the
    round-2 time-ordered negative control -- all from the same physics."""
    pos = one_period_samples(M)
    time_ordered = [(float(p[0]), float(p[1])) for p in pos]

    perm = bit_reversal_permutation(M.bit_length() - 1)
    check_bitreversal_prefix_coverage(perm, M)
    arm_a = [time_ordered[perm[i]] for i in range(M)]

    # Arm B draws from a finer trajectory so the Weyl indices cannot collide at
    # n_max=6400 and so it is not merely a re-permutation of arm A's 8192 samples.
    n_fine = 200_000
    pos_fine = one_period_samples(n_fine)
    idx_b = weyl_indices(max(PRODUCTION_N_GRID), n_fine - 1)
    arm_b = [(float(pos_fine[i, 0]), float(pos_fine[i, 1])) for i in idx_b]

    return {
        "arm_a": arm_a,
        "arm_a_time_indices": [int(perm[i]) for i in range(M)],
        "arm_b": arm_b,
        "arm_b_time_indices": [int(i) for i in idx_b],
        "time_ordered": time_ordered,
        "closure": float(np.hypot(*(pos[-1] - pos[0]))),
    }


def coverage_report(points: list[tuple[float, float]], n: int) -> str:
    arr = np.asarray(points[:n])
    r = np.linalg.norm(arr, axis=1)
    ang = np.mod(np.arctan2(arr[:, 1], arr[:, 0]), 2 * math.pi)
    ang_sorted = np.sort(ang)
    gaps = np.diff(np.concatenate([ang_sorted, [ang_sorted[0] + 2 * math.pi]]))
    return (
        f"r in [{r.min():.4f}, {r.max():.4f}] of true [0.4000, 1.6000]; "
        f"largest angular gap {np.max(gaps) / (2 * math.pi):.4f} of the orbit"
    )


# =============================================================================
# Reporting helpers
# =============================================================================
def show_compare(label: str, res, expect_note: str = "") -> dict[str, object]:
    savings = res.compute_savings_fraction
    print(f"  [{label}]")
    print(f"      poly_algebraic_min_n : {res.poly_algebraic_min_n}")
    print(f"      traditional_min_n    : {res.traditional_min_n}")
    print(
        f"      poly @ max_n={res.max_n_tested:<5d}  : "
        f"{res.poly_algebraic_estimate_at_max_n:.4f}"
    )
    print(f"      trad @ max_n={res.max_n_tested:<5d}  : {res.traditional_estimate_at_max_n:.4f}")
    print(f"      poly_algebraic_wins  : {res.poly_algebraic_wins}")
    print(
        f"      compute_savings      : "
        f"{'None' if savings is None else format(savings, '.4f')}   {expect_note}"
    )
    return {
        "poly_min_n": res.poly_algebraic_min_n,
        "trad_min_n": res.traditional_min_n,
        "wins": bool(res.poly_algebraic_wins),
        "savings": savings,
    }


def per_n_trace(points, n_grid, *, k=K, max_radius=MAX_RADIUS, time_indices=None,
                theiler_window=0, max_ball_fraction=1.0) -> None:
    """Per-n estimates from BOTH sides, plus the poly side's degeneracy
    diagnostics. compare() returns only the min_n summary; the trace is what
    lets a reader see whether a min_n is a stable crossing or a lucky one."""
    print(
        f"      {'n':>6} | {'poly':>8} {'deg':>6} {'near':>6} | {'trad':>8} {'R^2':>7} | "
        f"{'|dp|':>6} {'|dt|':>6}"
    )
    for n in n_grid:
        if n > len(points) or k >= n:
            break
        sub = points[:n]
        ti = None if time_indices is None else time_indices[:n]
        try:
            hg = knn_hypergraph(
                sub, k=k, dedupe=True, theiler_window=theiler_window, time_indices=ti
            )
            poly = mean_dimension(
                hg, samples=min(n, 40), max_radius=max_radius,
                max_ball_fraction=max_ball_fraction,
            )
            deg = degenerate_fraction(hg, samples=min(n, 40), max_radius=max_radius)
            near = near_degenerate_fraction(hg, samples=min(n, 40), max_radius=max_radius)
        except ValueError as exc:
            print(f"      {n:>6} | {'--':>8} {'':>6} {'':>6} |   ({exc})")
            continue
        trad = correlation_dimension(sub, theiler_window=theiler_window, time_indices=ti)
        print(
            f"      {n:>6} | {poly:8.4f} {deg:6.3f} {near:6.3f} | "
            f"{trad.dimension:8.4f} {trad.r_squared:7.4f} | "
            f"{abs(poly - TRUE_DIMENSION):6.4f} {abs(trad.dimension - TRUE_DIMENSION):6.4f}"
        )


# =============================================================================
# Sections
# =============================================================================
def section_1(state) -> None:
    print("=" * 78)
    print("SECTION 1 -- physics gate and point-cloud construction checks")
    print("=" * 78)
    v = verify_solver()
    for key, val in v.items():
        print(f"  {key}: {val}")
    assert v["passed"], "solver verification gate FAILED -- no measurement is reported"

    arms = state["arms"]
    print(f"\n  one-period closure |x(T) - x(0)| at M={M}: {arms['closure']:.3e}")
    print(f"  bit-reversal prefix-coverage property: verified at every power of two <= {M}")

    print("\n  Prefix coverage of the orbit, per construction (this is the whole point):")
    for n in (64, 100, 256, 400):
        print(f"      arm A (bitrev)  n={n:<5d} {coverage_report(arms['arm_a'], n)}")
        print(f"      arm B (Weyl)    n={n:<5d} {coverage_report(arms['arm_b'], n)}")
        print(f"      time-ordered    n={n:<5d} {coverage_report(arms['time_ordered'], n)}")
    print(
        "\n  => the time-ordered prefix is a perihelion arc, not an orbit; both\n"
        "     whole-orbit constructions cover the full radial and angular range at\n"
        "     every n. This is docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 10.10's finding,\n"
        "     re-derived here rather than quoted."
    )


def section_2(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 2 -- CRITERION (a) headline, at library defaults")
    print(f"             k={K}, max_radius={MAX_RADIUS} (module defaults, no deviation),")
    print("             theiler_window=0, max_ball_fraction=1.0")
    print("=" * 78)
    arms = state["arms"]

    print(f"\n  ARM A: bit-reversal over M={M}, n_grid={POW2_N_GRID}")
    res_a = compare(
        arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_arm_a"] = show_compare("arm A / defaults", res_a, "(round-2 record: 64 vs 256 = 0.75)")
    per_n_trace(arms["arm_a"], POW2_N_GRID)

    print(f"\n  ARM B: Weyl over one period, PRODUCTION n_grid={PRODUCTION_N_GRID}")
    res_b = compare(
        arms["arm_b"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=PRODUCTION_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_arm_b"] = show_compare("arm B / defaults", res_b)
    per_n_trace(arms["arm_b"], PRODUCTION_N_GRID)

    # Reproducibility gate: these are the numbers this script is on the record
    # for. A library change that moves them should fail loudly here rather than
    # be quietly re-quoted.
    recorded = {
        "arm A": (state["a_arm_a"], 64, 256, 0.75),
        "arm B": (state["a_arm_b"], 100, 400, 0.75),
    }
    print("\n  RECORDED-VALUE GATE")
    for label, (got, p, t, s) in recorded.items():
        ok = (got["poly_min_n"], got["trad_min_n"]) == (p, t) and got["savings"] == s
        print(
            f"      {label}: got poly={got['poly_min_n']} trad={got['trad_min_n']} "
            f"savings={got['savings']}; expected {p}/{t}/{s} -> {'MATCH' if ok else 'CHANGED'}"
        )
        assert ok, f"{label} moved off its recorded value -- investigate before quoting"

    print(
        "\n  CAVEAT, stated here because the win does not cancel it: the poly side's\n"
        "  degenerate + near-degenerate fraction is ~1.0 at every n in both arms (see\n"
        "  the 'deg'/'near' columns). This is finding F1 -- a smooth closed 1-D orbit\n"
        "  puts the k-NN graph in a ring-lattice regime where 1.0 is structural. The\n"
        "  savings figure is a true statement about points-to-converge; it is not\n"
        "  evidence that a fit measured the dimension."
    )


def section_3(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 3 -- NEGATIVE CONTROL: the round-2 time-ordered construction")
    print("=" * 78)
    arms = state["arms"]
    print(f"\n  time-ordered prefix, n_grid={POW2_N_GRID}")
    res = compare(
        arms["time_ordered"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_time_ordered_pow2"] = show_compare("time-ordered / pow2 grid", res)

    print(f"\n  time-ordered prefix, PRODUCTION n_grid={PRODUCTION_N_GRID}")
    res2 = compare(
        arms["time_ordered"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=PRODUCTION_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_time_ordered_prod"] = show_compare("time-ordered / production grid", res2)
    print(
        "\n  => reported for transparency ONLY. Neither number is quotable: at these\n"
        "     n the cloud is a sub-1% perihelion arc (section 1), so both estimators\n"
        "     are answering a question about an arc, not about the orbit."
    )


def section_4(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 4 -- CRITERION (a) robustness: the two post-round-2 knobs, and k")
    print("=" * 78)
    arms = state["arms"]

    arr = np.asarray(arms["arm_a"])
    auto_w = theiler_window_from_autocorrelation(
        arr, time_indices=np.asarray(arms["arm_a_time_indices"])
    )
    print(f"\n  theiler_window='auto' resolves to {auto_w} on arm A's full cloud")
    print(
        "  (time_indices are passed explicitly: under a bit-reversal ordering the\n"
        "   array position is NOT the trajectory time, and the default t_i = i\n"
        "   assumption would be silently wrong.)"
    )
    per_n_auto = {
        n: theiler_window_from_autocorrelation(
            arr[:n], time_indices=np.asarray(arms["arm_a_time_indices"][:n])
        )
        for n in (64, 256, 1024, 8192)
    }
    print(f"  per-prefix 'auto' windows: {per_n_auto}")
    print(
        "  EXPECT THIS TO BREAK BOTH SIDES, and see the module docstring for why: on a\n"
        "  single period of a CLOSED orbit each neighbourhood is visited exactly once,\n"
        "  so a Theiler window deletes the geometry rather than a temporal artifact.\n"
        "  Shown because H1 exists and a reader is entitled to the number, not because\n"
        "  it is the setting this problem should be quoted at."
    )
    state["a_auto_window"] = int(auto_w)

    rows = []
    for label, tw, mbf in (
        ("theiler=0,   mbf=1.0 (defaults)", 0, 1.0),
        ("theiler=auto, mbf=1.0", "auto", 1.0),
        ("theiler=0,   mbf=0.5", 0, 0.5),
        ("theiler=auto, mbf=0.5", "auto", 0.5),
    ):
        res = compare(
            arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
            k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
            theiler_window=tw, time_indices=arms["arm_a_time_indices"],
            max_ball_fraction=mbf,
        )
        rows.append((label, res))
        print(f"\n  arm A, {label}")
        show_compare(label, res)
    state["a_knobs"] = {
        lbl: (r.poly_algebraic_min_n, r.traditional_min_n, r.compute_savings_fraction)
        for lbl, r in rows
    }

    print("\n  k sweep at defaults (arm A) -- R2-F3 guard: is the win a property of k?")
    ksweep = {}
    for k in (4, 6, 8, 10):
        res = compare(
            arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
            k=k, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
        )
        sav = res.compute_savings_fraction
        ksweep[k] = (res.poly_algebraic_min_n, res.traditional_min_n, sav)
        print(
            f"      k={k:<3d} poly={res.poly_algebraic_min_n}  trad={res.traditional_min_n}  "
            f"savings={'None' if sav is None else format(sav, '.4f')}"
        )
    state["a_ksweep"] = ksweep

    print("\n  max_radius sweep at defaults (arm A) -- the exact R2-F3 axis:")
    mrsweep = {}
    for mr in (3, 4, 5, 6, 7, 8):
        res = compare(
            arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
            k=K, n_grid=POW2_N_GRID, max_radius=mr,
        )
        sav = res.compute_savings_fraction
        mrsweep[mr] = (res.poly_algebraic_min_n, res.traditional_min_n, sav)
        print(
            f"      max_radius={mr:<3d} poly={res.poly_algebraic_min_n}  "
            f"trad={res.traditional_min_n}  "
            f"savings={'None' if sav is None else format(sav, '.4f')}"
        )
    state["a_mrsweep"] = mrsweep


def section_5(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 5 -- CRITERION (c), H3: accuracy at fixed max n")
    print("=" * 78)
    arms = state["arms"]
    for label, pts, n in (
        ("arm A (bitrev)", arms["arm_a"], M),
        ("arm B (Weyl)", arms["arm_b"], max(PRODUCTION_N_GRID)),
    ):
        res = compare_accuracy_at_max_n(
            pts, true_dimension=TRUE_DIMENSION, max_n=n,
            k=K, tolerance=TOLERANCE, max_radius=MAX_RADIUS,
        )
        print(f"\n  [{label}]  n_evaluated={res.n_evaluated}  nodes={res.poly_algebraic_n_nodes}")
        print(
            f"      poly  {res.poly_algebraic_estimate:.4f}  |err| "
            f"{res.poly_algebraic_abs_error:.4f}"
        )
        print(
            f"      trad  {res.traditional_estimate:.4f}  |err| "
            f"{res.traditional_abs_error:.4f}   R^2={res.traditional_r_squared:.4f}"
        )
        print(
            f"      sentinel fraction {res.poly_algebraic_sentinel_fraction:.4f} "
            f"(degenerate {res.poly_algebraic_degenerate_fraction:.3f} + near-degenerate "
            f"{res.poly_algebraic_near_degenerate_fraction:.3f}); "
            f"cap {res.max_sentinel_fraction:.3f}"
        )
        print(f"      more_accurate            : {res.more_accurate}")
        print(f"      poly_algebraic_accuracy_win: {res.poly_algebraic_accuracy_win}")
        print(f"      traditional_accuracy_win   : {res.traditional_accuracy_win}")
        print(f"      verdict_reason           : {res.verdict_reason}")
        state[f"c_{label.split()[1]}"] = {
            "win": bool(res.poly_algebraic_accuracy_win),
            "poly_err": res.poly_algebraic_abs_error,
            "trad_err": res.traditional_abs_error,
            "sentinel": res.poly_algebraic_sentinel_fraction,
            "n": res.n_evaluated,
            "reason": res.verdict_reason,
        }


def summary(state) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key in ("a_arm_a", "a_arm_b"):
        if key in state:
            d = state[key]
            print(
                f"  criterion (a) {key}: poly={d['poly_min_n']} trad={d['trad_min_n']} "
                f"wins={d['wins']} savings="
                f"{'None' if d['savings'] is None else format(d['savings'], '.4f')}"
            )
    for key in ("c_A", "c_B"):
        if key in state:
            d = state[key]
            print(
                f"  criterion (c) {key}: win={d['win']} poly_err={d['poly_err']:.4f} "
                f"trad_err={d['trad_err']:.4f} sentinel={d['sentinel']:.4f}"
            )


SECTIONS = {1: section_1, 2: section_2, 3: section_3, 4: section_4, 5: section_5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5")
    args = ap.parse_args()
    wanted = [int(s) for s in args.sections.split(",") if s.strip()]

    state: dict[str, object] = {}
    print("building point clouds (one-period Kepler, e=0.6) ...")
    state["arms"] = build_arms()
    print(f"  arm A: {len(state['arms']['arm_a'])} pts | arm B: {len(state['arms']['arm_b'])} pts "
          f"| time-ordered control: {len(state['arms']['time_ordered'])} pts")

    for s in wanted:
        SECTIONS[s](state)
    summary(state)


if __name__ == "__main__":
    main()
