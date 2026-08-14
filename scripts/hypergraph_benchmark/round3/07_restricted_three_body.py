"""Round 3 remeasurement of benchmark problem 07 (CR3BP planar Lyapunov orbit
near L1, with a genuine close approach to the Moon) against the CURRENT library
state (post N8, H1, H3, and the AutoResearch H2 loop).

WHAT THE RECORD SAYS GOING IN
-----------------------------
docs/MENSURA_BENCHMARK.md Sec. 10.2 scores problem 07 as "no": a genuine
criterion-(a) TIE, and its Sec. 10.2 correction block states the tie explicitly
survives the whole-orbit construction that turned problems 03 and 04 from ties
into wins ("CR3BP (07) re-measured the same way stays a genuine tie (128 vs
128), so this construction does not mechanically manufacture wins everywhere it
is applied"). Criterion (b) on this problem was not merely null but INVERTED --
the traditional method was more accurate under the close-approach density
variation (R2-F1, magnitude corrected from 4.5x to ~2.9x in Sec. 10.10) -- and
criterion (b) is retired (Sec. 10.8, carry-forward item 2). It is therefore NOT
re-attempted here, per this round's dispatch.

This script re-derives criterion (a) and measures criterion (c) -- which has
never been measured on this problem, because H3
(`comparison.compare_accuracy_at_max_n`) postdates round 2 entirely.

CONSTRUCTION (whole-orbit prefixes, and why)
--------------------------------------------
`compare()` evaluates `points[:n]`. Round 2 fed it a TIME-ORDERED cloud, so
n=100 at dt=1e-4 is the first 0.01 time units of a ~2.8-unit period -- 0.4% of
the orbit, a short smooth arc far from the Moon. The round-2 script itself said
so in its own output. Sec. 10.10 diagnosed exactly this construction as unsound
for closed orbits, so the headline numbers here use whole-orbit-covering
prefixes, and the time-ordered cloud is kept alive only as a NEGATIVE CONTROL
(section 3), so the artifact is visible in the output rather than quoted.

Two independent whole-orbit constructions, not one:

  ARM A  bit-reversal permutation of M = 8192 evenly-strided samples of exactly
         one period, on the power-of-two grid (32 ... 8192). This is the
         construction Sec. 10.2's correction used for its "128 vs 128" tie, so
         its numbers are directly comparable to that record (measured pre-H1).

  ARM B  golden-ratio Weyl index sequence over the same one period, on problem
         07's PRODUCTION n_grid (100 ... 12800) from the round-2 script. Arm A
         changes two things at once relative to round 2 (the ordering AND the
         grid); arm B changes the ordering alone. A verdict that needed a
         hand-picked grid would be measurement-config gaming.

Both arms are drawn from ONE fine trajectory of exactly one period (131072 RK4
steps, dt = T/131072 ~ 2.1e-5, five times finer than round 2's dt=1e-4), so the
two arms differ in index sequence only and share identical physics. The
endpoint is excluded: t=0 and t=T are the same physical point on a closed orbit,
and including both is finding F3/N7's near-duplicate cluster by construction.

HYPERPARAMETERS: k=6, max_radius=6 -- the module defaults, and also problem 07's
round-2 production values (scripts/hypergraph_benchmark/round2/
07_restricted_three_body.py K=6, MAX_RADIUS=6). NO deviation. Sec. 9.4 (R2-F3)
traced a false round-2 win to an undisclosed non-default `max_radius`; both
knobs are swept in section 4 as a guard, never to pick a headline. tolerance=0.2
and true_dimension=1.0 are problem 07's production values, unchanged.
`theiler_window` and `max_ball_fraction` are reported at their library DEFAULTS
(0 and 1.0) as the headline, with the corrected settings shown alongside as
robustness rows.

WHAT IS MEASURED
  Criterion (a)  comparison.compare()                   -> sections 2, 3, 4, 6
  Criterion (c)  comparison.compare_accuracy_at_max_n() -> section 5 (H3)
  Criterion (b)  NOT measured -- retired, Sec. 10.8 item 2.
  Provenance     which library change moved the number  -> section 7

Sections are selectable: --sections 1,...,7 (default: all). The orbit
determination and the fine trajectory are cached to the scratchpad, so the
first run pays ~3 minutes for the shooting and later runs do not. Section 4 is
~4 minutes and section 6 ~4 minutes; everything else is seconds.

MEASURED RESULT (this script, current library: post N8 / H1 / H3 / AR H2 loop)
-----------------------------------------------------------------------------
CRITERION (a) IS NOW A WIN, and the recorded tie is no longer reproduced:

    arm A (bitrev, pow2 grid)      poly  64 vs trad 128  -> savings 0.5000
    arm B (Weyl, production grid)  poly 100 vs trad 200  -> savings 0.5000
    time-ordered CONTROL (prod)    poly 100 vs trad 100  -> 0.0 (reproduces the
                                   round-2 tie on the round-2 construction)

It is not the R2-F3 failure mode. `max_radius` 3..8 gives savings 0.5000 in all
six (perfectly flat -- the default is not a special value here); k=4/6/8/10
gives 0.75/0.50/0.50/0.00, so the DEFAULT k=6 is not the best cell; the six
orbit-phase rotations of section 6b all give 64 vs 128 (6 of 6); and tolerance
0.10/0.15/0.20/0.25/0.30 gives arm-A savings 0.75/0.75/0.50/0.50/0.00, i.e. the
win is if anything SUPPRESSED by the production tolerance rather than
manufactured by it (it only dies when the tolerance is LOOSENED to 0.30, which
lets the baseline in at n=64 too).

PROVENANCE (section 7): the flip is attributable to N8 and to nothing else.
Under the exact pre-N8 acceptance rule (`is_well_fit(0.9,
near_constant_consensus=False)` -- R^2 only, which is what R2-F4's cancellation
made near-constant sequences fail) the identical sampled nodes on the identical
cloud give poly min_n = 128, i.e. the recorded 128-vs-128 tie, reproduced. Under
the shipped rule it is 64. At n=64 the pre-N8 rule accepts 0 of 40 sampled nodes
and the shipped rule accepts 33 of 40, of which the near-constant branch supplies
the difference (near_degenerate fraction 0.825).

That is also the caveat, and it is not cancelled by the win: the newly accepted
nodes are near-constant shell sequences, so this win is MORE dependent on the F1
ring-lattice regime than the tie it replaces. It is a true statement about
points-to-converge; it is not evidence that a fit measured the dimension.

CRITERION (c) IS NOT A WIN, in either arm, for two independent reasons, and the
library states both rather than this script:
  * sentinel fraction 1.000 (arm A, n=8192) and 0.925 (arm B, n=12800) against
    the 0.05 cap -- condition (c2), finding F1;
  * the baseline's own error at max n is 0.0014 (arm A) and 0.0023 (arm B), far
    inside the 0.2 tolerance, so condition (c4) ("the baseline fails the same
    bar") fails too. Neither is close.
Poly's errors there are 0.0000 (arm A) and 0.0217 (arm B); on this problem both
methods are simply correct at large n, which is the honest reason (c) is empty.

H1 ON THIS PROBLEM (section 4, measured not assumed): `theiler_window="auto"`
resolves to 250 on arm A's full cloud (12/51/204/250 at n=128/512/2048/8192) and
DESTROYS the poly side -- 2.1371 at n=8192, no convergence anywhere on the grid,
so `poly_algebraic_min_n` becomes None and there is no win in either direction.
This is correct behaviour, not a defect: a single period of a closed orbit visits
each neighbourhood exactly once, so temporal proximity IS spatial proximity here
and a +-250-sample window deletes geometry instead of an artifact. The headline
therefore stays at the library default 0, reported not hidden.
`max_ball_fraction=0.5` changes nothing at all (64 vs 128 either way).

Modifies nothing under src/ or tests/.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
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
    _sampled_nodes,  # private on purpose: section 7 must sample the SAME nodes
    degenerate_fraction,
    local_dimension,
    mean_dimension,
    near_constant_consensus,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import (  # noqa: E402
    DuplicatePointsError,
    knn_hypergraph,
)

# --- problem 07 constants, taken from the round-2 script ---------------------
TRUE_DIMENSION = 1.0
TOLERANCE = 0.2
K = 6
MAX_RADIUS = 6
PRODUCTION_N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)

# --- arm A / fine-trajectory construction constants --------------------------
M = 8192  # evenly-strided samples of ONE period; power of two for bit reversal
POW2_N_GRID = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
N_FINE = 131072  # RK4 steps over exactly one period (dt ~ 2.1e-5)

CLOSURE_POS_TOL = 1e-8
CLOSURE_VEL_TOL = 1e-8

CACHE_DIR = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b"
    "/scratchpad/round3_p07"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Physics: imported from the ROUND-2 script, not re-typed
# =============================================================================
# The CR3BP right-hand side, the RK4 stepper, the L1 root-find, the linear
# eigenmode guess and the single-shooting differential correction are the
# round-2 module's own functions, loaded here by path (the round-2 script is
# main-guarded, so importing it runs nothing). Re-typing them would introduce a
# transcription risk for no benefit, and this way the orbit measured in round 3
# is provably the same object round 2 measured -- only the ORDERING of the
# sampled cloud differs, which is the single change this remeasurement makes.
# This is the same mechanism scripts/hypergraph_benchmark/round3/n8d_clouds.py
# uses for all ten clouds.
def _round2_module():
    path = ROOT / "scripts" / "hypergraph_benchmark" / "round2" / "07_restricted_three_body.py"
    spec = importlib.util.spec_from_file_location("round2_p07", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R2 = _round2_module()
MU = R2.MU
AX_AMPLITUDE = R2.AX_AMPLITUDE
DT_CLOUD = R2.DT_CLOUD  # round-2's time-ordered cloud step, used by the control


def solve_orbit() -> dict[str, object]:
    """Round 2's orbit determination, verbatim, cached to disk.

    Returns the corrected initial state, the period, and the closure /
    Jacobi-conservation diagnostics that gate every number below.
    """
    cache = CACHE_DIR / "orbit.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)

    x_l1 = R2.find_l1(MU)
    x0 = x_l1 + AX_AMPLITUDE
    vy0_guess = R2.linear_vy0_guess(x_l1, AX_AMPLITUDE)
    vy0_coarse = R2.find_periodic_orbit(x0, vy0_guess, dt=1e-3, t_max=2.2, span_frac=0.4)
    vy0_fine = R2.find_periodic_orbit(x0, vy0_coarse, dt=1e-4, t_max=2.2, span_frac=0.05)
    t_half, cross_state = R2.half_period_crossing(
        np.array([x0, 0.0, 0.0, vy0_fine]), dt=1e-5, t_max=2.2
    )
    period = 2.0 * t_half
    state0 = np.array([x0, 0.0, 0.0, vy0_fine])

    # Closure gate: integrate exactly one confirmed period and require BOTH
    # position and velocity to return, plus Jacobi conservation. Round 2's gate,
    # unchanged; a small shooting residual is not accepted as proof of closure.
    dt_verify = 1e-5
    n_steps_verify = int(round(period / dt_verify))
    dt_actual = period / n_steps_verify
    states_one_period = R2.integrate(state0, dt_actual, n_steps_verify)
    final_state = states_one_period[-1]
    pos_err = float(np.linalg.norm(final_state[:2] - state0[:2]))
    vel_err = float(np.linalg.norm(final_state[2:] - state0[2:]))
    cj0 = R2.jacobi_constant(state0)
    cj_all = np.array([R2.jacobi_constant(s) for s in states_one_period[::200]])
    cj_drift = float(np.max(np.abs(cj_all - cj0)) / abs(cj0))

    r2_full = R2.moon_distance(states_one_period)
    speeds = np.hypot(states_one_period[:, 2], states_one_period[:, 3])

    out = {
        "x_l1": float(x_l1),
        "x0": float(x0),
        "vy0": float(vy0_fine),
        "period": float(period),
        "state0": state0,
        "vx_at_half_period": float(cross_state[2]),
        "pos_err": pos_err,
        "vel_err": vel_err,
        "jacobi_drift": cj_drift,
        "passed": bool(
            pos_err <= CLOSURE_POS_TOL and vel_err <= CLOSURE_VEL_TOL and cj_drift < 1e-8
        ),
        "moon_dist_min": float(r2_full.min()),
        "moon_dist_max": float(r2_full.max()),
        "speed_min": float(speeds.min()),
        "speed_max": float(speeds.max()),
    }
    with cache.open("wb") as fh:
        pickle.dump(out, fh)
    return out


def fine_one_period(orbit: dict[str, object]) -> np.ndarray:
    """N_FINE evenly-spaced-in-time states over exactly one period, endpoint
    EXCLUDED (t=0 and t=T are the same point on a closed orbit)."""
    cache = CACHE_DIR / f"fine_{N_FINE}.npy"
    if cache.exists():
        return np.load(cache)
    period = float(orbit["period"])
    dt = period / N_FINE
    states = R2.integrate(np.asarray(orbit["state0"]), dt, N_FINE)[:N_FINE]
    np.save(cache, states)
    return states


def time_ordered_cloud(orbit: dict[str, object]) -> list[tuple[float, float]]:
    """Round 2's production cloud for this problem, reproduced exactly:
    dt=1e-4 over 1.02 periods, tail trimmed by 20 steps. Used ONLY as the
    negative control in section 3."""
    cache = CACHE_DIR / "time_ordered.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    period = float(orbit["period"])
    n_steps = int(round(1.02 * period / DT_CLOUD))
    states = R2.integrate(np.asarray(orbit["state0"]), DT_CLOUD, n_steps)
    pos = states[:-20, :2]
    pts = [(float(p[0]), float(p[1])) for p in pos]
    with cache.open("wb") as fh:
        pickle.dump(pts, fh)
    return pts


# =============================================================================
# Point-cloud orderings
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


def weyl_indices(n_max: int, n_steps_total: int) -> np.ndarray:
    """Golden-ratio Weyl index sequence; every PREFIX is a near-uniform sample of
    the whole period, so `points[:n]` covers the orbit at every n."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fracs = (np.arange(n_max) * phi) % 1.0
    idx = np.floor(fracs * n_steps_total).astype(int)
    if len(np.unique(idx)) != n_max:
        raise RuntimeError("Weyl index sequence collided; increase the time resolution")
    return idx


def rotated_arm_a(states: np.ndarray, perm: list[int], rot: int):
    """Arm A's bit-reversal cloud with the sample phase rotated by `rot`.

    A deterministic orbit has no seed, so the closest analogue of a redraw is
    the arbitrary choice of which point on the closed loop is index 0. Rotating
    by `rot` makes the n-point prefix a DIFFERENT evenly-strided set of n points
    on the same orbit (still whole-orbit-covering), which is exactly the
    perturbation h3_win_robustness.py uses reseeding for on stochastic clouds.
    """
    stride = N_FINE // M
    base_idx = [(rot + i) * stride % N_FINE for i in range(M)]
    base = [(float(states[i, 0]), float(states[i, 1])) for i in base_idx]
    pts = [base[perm[i]] for i in range(M)]
    tidx = [int(base_idx[perm[i]]) for i in range(M)]
    return pts, tidx


def build_arms(orbit: dict[str, object]) -> dict[str, object]:
    states = fine_one_period(orbit)
    stride = N_FINE // M
    base_idx = list(range(0, N_FINE, stride))  # M evenly-strided samples of one period
    base = [(float(states[i, 0]), float(states[i, 1])) for i in base_idx]

    perm = bit_reversal_permutation(M.bit_length() - 1)
    check_bitreversal_prefix_coverage(perm, M)
    arm_a = [base[perm[i]] for i in range(M)]
    arm_a_time_indices = [int(base_idx[perm[i]]) for i in range(M)]

    idx_b = weyl_indices(max(PRODUCTION_N_GRID), N_FINE - 1)
    arm_b = [(float(states[i, 0]), float(states[i, 1])) for i in idx_b]
    arm_b_time_indices = [int(i) for i in idx_b]

    closure = float(
        np.hypot(states[-1, 0] - states[0, 0], states[-1, 1] - states[0, 1])
    )
    return {
        "arm_a": arm_a,
        "arm_a_time_indices": arm_a_time_indices,
        "arm_b": arm_b,
        "arm_b_time_indices": arm_b_time_indices,
        "time_ordered": time_ordered_cloud(orbit),
        "sample_gap": closure,
        "states": states,
        "perm": perm,
    }


# =============================================================================
# Reporting helpers
# =============================================================================
def coverage_report(points, n: int, orbit: dict[str, object]) -> str:
    """How much of the ORBIT a prefix of length n actually covers.

    Angle is measured about the prefix-independent orbit centroid (the full
    M-sample centroid would be circular reasoning), i.e. about the centre of the
    closed Lyapunov loop, so 'largest angular gap' is a real coverage statistic.
    """
    arr = np.asarray(points[:n])
    centre = np.asarray(orbit["centre"])
    rel = arr - centre
    ang = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * math.pi)
    ang_sorted = np.sort(ang)
    gaps = np.diff(np.concatenate([ang_sorted, [ang_sorted[0] + 2 * math.pi]]))
    dmoon = np.hypot(arr[:, 0] - 1 + MU, arr[:, 1])
    return (
        f"x in [{arr[:, 0].min():.4f}, {arr[:, 0].max():.4f}], "
        f"y in [{arr[:, 1].min():+.4f}, {arr[:, 1].max():+.4f}]; "
        f"d(Moon) in [{dmoon.min():.5f}, {dmoon.max():.5f}]; "
        f"largest angular gap {np.max(gaps) / (2 * math.pi):.4f} of the orbit"
    )


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
    """Per-n estimates from BOTH sides plus the poly side's degeneracy
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
    orbit = state["orbit"]
    print(f"  mu                      : {MU}")
    print(f"  x_L1                    : {orbit['x_l1']:.10f}   Moon at x = {1 - MU:.10f}")
    print(f"  ax_amplitude            : {AX_AMPLITUDE}  (round-2 value, close lunar approach)")
    print(f"  x0, vy0 (corrected)     : {orbit['x0']:.10f}, {orbit['vy0']:.12f}")
    print(f"  period T                : {orbit['period']:.10f}")
    print(f"  vx at half period       : {orbit['vx_at_half_period']:.3e}  (want ~0)")
    print(f"  |pos err| after 1 period: {orbit['pos_err']:.3e}  (tol {CLOSURE_POS_TOL:.0e})")
    print(f"  |vel err| after 1 period: {orbit['vel_err']:.3e}  (tol {CLOSURE_VEL_TOL:.0e})")
    print(f"  Jacobi max rel drift    : {orbit['jacobi_drift']:.3e}  (tol 1e-8)")
    print(f"  SOLVER GATE PASSED      : {orbit['passed']}")
    assert orbit["passed"], "solver verification gate FAILED -- no measurement is reported"

    print(
        f"\n  close approach: min d(Moon) over one period = {orbit['moon_dist_min']:.5f} "
        f"(Moon radius ~0.0045 in these units)"
    )
    print(
        f"  speed range [{orbit['speed_min']:.5f}, {orbit['speed_max']:.5f}] = "
        f"{orbit['speed_max'] / orbit['speed_min']:.3f}x -- the density variation that "
        f"criterion (b) probed"
    )
    print(
        "  criterion (b) is RETIRED (Sec. 10.8 item 2) and is not measured here; on this\n"
        "  problem it was not merely null but inverted (R2-F1), which is on the record."
    )

    arms = state["arms"]
    print(
        f"\n  fine trajectory: {N_FINE} RK4 steps over exactly one period "
        f"(dt = {orbit['period'] / N_FINE:.3e})"
    )
    print(
        f"  gap between the last sample and t=0 (one sampling step, NOT a closure\n"
        f"  error -- the endpoint is deliberately excluded): {arms['sample_gap']:.3e}"
    )
    print(f"  bit-reversal prefix-coverage property: verified at every power of two <= {M}")

    print("\n  DUPLICATE CHECK (dedupe=False must not raise -- finding F3/N7):")
    for label in ("arm_a", "arm_b", "time_ordered"):
        try:
            knn_hypergraph(arms[label][: min(len(arms[label]), 4096)], k=K, dedupe=False)
            print(f"      {label:<13s}: no near-duplicate cluster")
        except DuplicatePointsError as exc:
            print(f"      {label:<13s}: DUPLICATES -- {exc}")

    print("\n  Prefix coverage of the orbit, per construction (this is the whole point):")
    for n in (100, 128, 400, 512):
        print(f"      arm A (bitrev)  n={n:<5d} {coverage_report(arms['arm_a'], n, orbit)}")
        print(f"      arm B (Weyl)    n={n:<5d} {coverage_report(arms['arm_b'], n, orbit)}")
        print(f"      time-ordered    n={n:<5d} {coverage_report(arms['time_ordered'], n, orbit)}")
    print(
        "\n  => the time-ordered prefix is a short arc near x0, nowhere near the Moon;\n"
        "     both whole-orbit constructions cover the full loop at every n. This is\n"
        "     docs/MENSURA_BENCHMARK.md Sec. 10.10's finding, re-derived rather\n"
        "     than quoted."
    )


def section_2(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 2 -- CRITERION (a) headline, at library defaults")
    print(f"             k={K}, max_radius={MAX_RADIUS} (module defaults, no deviation),")
    print(f"             tolerance={TOLERANCE}, theiler_window=0, max_ball_fraction=1.0")
    print("=" * 78)
    arms = state["arms"]

    print(f"\n  ARM A: bit-reversal over M={M}, n_grid={POW2_N_GRID}")
    res_a = compare(
        arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_arm_a"] = show_compare("arm A / defaults", res_a, "(round-2 record: 128 vs 128, tie)")
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
    # be quietly re-quoted. Note the round-2 record for arm A was 128 vs 128;
    # section 7 exhibits which library change accounts for the difference.
    recorded = {
        "arm A": (state["a_arm_a"], 64, 128, 0.5),
        "arm B": (state["a_arm_b"], 100, 200, 0.5),
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
        "\n  CAVEAT (finding F1), stated whatever the verdict: the poly side's\n"
        "  degenerate + near-degenerate fraction is ~1.0 at every n (the 'deg'/'near'\n"
        "  columns). A smooth closed 1-D orbit puts the k-NN graph in a ring-lattice\n"
        "  regime where the answer 1.0 is structural, not fitted. Section 7 shows the\n"
        "  win above is carried by exactly those near-constant nodes, so it inherits\n"
        "  this caveat in full."
    )


def section_3(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 3 -- NEGATIVE CONTROL: the round-2 time-ordered construction")
    print("=" * 78)
    arms = state["arms"]
    print(f"\n  time-ordered prefix, PRODUCTION n_grid={PRODUCTION_N_GRID}")
    res = compare(
        arms["time_ordered"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=PRODUCTION_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_time_ordered_prod"] = show_compare("time-ordered / production grid", res)
    per_n_trace(arms["time_ordered"], PRODUCTION_N_GRID)

    print(f"\n  time-ordered prefix, pow2 n_grid={POW2_N_GRID}")
    res2 = compare(
        arms["time_ordered"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
    )
    state["a_time_ordered_pow2"] = show_compare("time-ordered / pow2 grid", res2)
    print(
        "\n  => reported for transparency ONLY; neither number is quotable. At these n\n"
        "     the cloud is a sub-1% arc of the orbit (section 1), so both estimators are\n"
        "     answering a question about an arc, not about the orbit."
    )


def section_4(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 4 -- CRITERION (a) robustness: H1/H2 knobs, k, and max_radius")
    print("=" * 78)
    arms = state["arms"]

    arr = np.asarray(arms["arm_a"])
    auto_w = theiler_window_from_autocorrelation(
        arr, time_indices=np.asarray(arms["arm_a_time_indices"])
    )
    print(f"\n  theiler_window='auto' resolves to {auto_w} on arm A's full cloud")
    print(
        "  (time_indices are passed explicitly: under a bit-reversal ordering the array\n"
        "   position is NOT the trajectory time, and the default t_i = i assumption\n"
        "   would be silently wrong.)"
    )
    per_n_auto = {
        n: int(
            theiler_window_from_autocorrelation(
                arr[:n], time_indices=np.asarray(arms["arm_a_time_indices"][:n])
            )
        )
        for n in (128, 512, 2048, 8192)
    }
    print(f"  per-prefix 'auto' windows: {per_n_auto}")
    print(
        "  EXPECT THIS TO HURT BOTH SIDES: a single period of a CLOSED orbit visits each\n"
        "  neighbourhood exactly once, so temporal proximity IS spatial proximity here and\n"
        "  a Theiler window deletes geometry rather than an artifact. Shown because H1\n"
        "  exists and a reader is entitled to the number, not because it is the setting\n"
        "  this problem should be quoted at."
    )
    state["a_auto_window"] = int(auto_w)

    rows = {}
    for label, tw, mbf in (
        ("theiler=0,    mbf=1.0 (defaults)", 0, 1.0),
        ("theiler=auto, mbf=1.0", "auto", 1.0),
        ("theiler=0,    mbf=0.5", 0, 0.5),
        ("theiler=auto, mbf=0.5", "auto", 0.5),
    ):
        try:
            res = compare(
                arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
                k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
                theiler_window=tw, time_indices=arms["arm_a_time_indices"],
                max_ball_fraction=mbf,
            )
        except ValueError as exc:
            print(f"\n  arm A, {label}: ValueError -- {exc}")
            rows[label] = (None, None, None)
            continue
        print(f"\n  arm A, {label}")
        show_compare(label, res)
        rows[label] = (
            res.poly_algebraic_min_n, res.traditional_min_n, res.compute_savings_fraction
        )
    state["a_knobs"] = rows

    print("\n  k sweep at defaults (arm A) -- R2-F3 guard: is the verdict a property of k?")
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
    print(
        "\n  The DEFAULT max_radius=6 is the value quoted in section 2; the sweep exists\n"
        "  only to show whether the verdict is a property of the data or of the knob."
    )


def section_5(state) -> None:
    print("\n" + "=" * 78)
    print("SECTION 5 -- CRITERION (c), H3: accuracy at fixed max n")
    print(f"             tolerance={TOLERANCE}, margin/sentinel cap at library defaults")
    print("=" * 78)
    print(
        "  NOTE (capability gap, on the record): compare_accuracy_at_max_n accepts\n"
        "  neither theiler_window nor max_ball_fraction, so criterion (c) can only be\n"
        "  measured at the pre-H1/H2 defaults. Stated, not worked around."
    )
    arms = state["arms"]
    for key, label, pts, n in (
        ("c_A", "arm A (bitrev)", arms["arm_a"], M),
        ("c_B", "arm B (Weyl)", arms["arm_b"], max(PRODUCTION_N_GRID)),
        ("c_T", "time-ordered (control)", arms["time_ordered"], max(PRODUCTION_N_GRID)),
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
        print(f"      more_accurate              : {res.more_accurate}")
        print(f"      poly_algebraic_accuracy_win: {res.poly_algebraic_accuracy_win}")
        print(f"      traditional_accuracy_win   : {res.traditional_accuracy_win}")
        print(f"      verdict_reason             : {res.verdict_reason}")
        state[key] = {
            "win": bool(res.poly_algebraic_accuracy_win),
            "poly_err": float(res.poly_algebraic_abs_error),
            "trad_err": float(res.traditional_abs_error),
            "sentinel": float(res.poly_algebraic_sentinel_fraction),
            "n": int(res.n_evaluated),
            "reason": res.verdict_reason,
        }


def summary(state) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, label in (
        ("a_arm_a", "arm A (bitrev, pow2 grid)"),
        ("a_arm_b", "arm B (Weyl, production grid)"),
        ("a_time_ordered_prod", "time-ordered CONTROL (not quotable)"),
        ("a_time_ordered_pow2", "time-ordered CONTROL pow2 (not quotable)"),
    ):
        if key in state:
            d = state[key]
            print(
                f"  criterion (a) {label:<38s}: poly={d['poly_min_n']} trad={d['trad_min_n']} "
                f"wins={d['wins']} savings="
                f"{'None' if d['savings'] is None else format(d['savings'], '.4f')}"
            )
    for key, label in (
        ("c_A", "arm A"), ("c_B", "arm B"), ("c_T", "time-ordered CONTROL")
    ):
        if key in state:
            d = state[key]
            print(
                f"  criterion (c) {label:<38s}: win={d['win']} poly_err={d['poly_err']:.4f} "
                f"trad_err={d['trad_err']:.4f} sentinel={d['sentinel']:.4f}"
            )


def section_6(state) -> None:
    """The two axes on which a one-grid-step win is most likely to be an
    artifact: the tolerance that defines "converged", and the arbitrary phase
    of the sample set. Neither is a hyperparameter of the ESTIMATOR, which is
    why they are not covered by section 4's R2-F3 sweep."""
    print("\n" + "=" * 78)
    print("SECTION 6 -- adversarial robustness of a ONE-GRID-STEP win")
    print("=" * 78)
    arms = state["arms"]

    print(
        "\n  (6a) TOLERANCE SENSITIVITY. tolerance=0.2 is problem 07's round-2 production\n"
        "       value and is NOT chosen here; the sweep exists so a reader can see how\n"
        "       much of the verdict rides on it. Both min_n values move with it, since\n"
        "       both are 'first n within tolerance and stably so'."
    )
    tol_rows = {}
    for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
        res_a = compare(
            arms["arm_a"], true_dimension=TRUE_DIMENSION, tolerance=tol,
            k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
        )
        res_b = compare(
            arms["arm_b"], true_dimension=TRUE_DIMENSION, tolerance=tol,
            k=K, n_grid=PRODUCTION_N_GRID, max_radius=MAX_RADIUS,
        )
        tol_rows[tol] = {
            "A": (res_a.poly_algebraic_min_n, res_a.traditional_min_n,
                  res_a.compute_savings_fraction),
            "B": (res_b.poly_algebraic_min_n, res_b.traditional_min_n,
                  res_b.compute_savings_fraction),
        }
        mark = "  <- PRODUCTION" if tol == TOLERANCE else ""
        sav_a = res_a.compute_savings_fraction
        sav_b = res_b.compute_savings_fraction
        txt_a = "None" if sav_a is None else format(sav_a, ".4f")
        txt_b = "None" if sav_b is None else format(sav_b, ".4f")
        print(
            f"      tol={tol:.2f}  armA poly={res_a.poly_algebraic_min_n} "
            f"trad={res_a.traditional_min_n} sav={txt_a}"
            f"   |   armB poly={res_b.poly_algebraic_min_n} "
            f"trad={res_b.traditional_min_n} sav={txt_b}{mark}"
        )
    state["a_tolsweep"] = tol_rows

    print(
        "\n  (6b) ORBIT-PHASE ROTATION (the deterministic analogue of a reseed): the same\n"
        "       orbit, the same construction, a different arbitrary choice of which point\n"
        "       is index 0. Each rotation gives a genuinely different n-point prefix."
    )
    rot_rows = {}
    for rot in (0, 13, 41, 77, 333, 1234):
        pts, _ = rotated_arm_a(arms["states"], arms["perm"], rot)
        res = compare(
            pts, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
            k=K, n_grid=POW2_N_GRID, max_radius=MAX_RADIUS,
        )
        sav = res.compute_savings_fraction
        rot_rows[rot] = (res.poly_algebraic_min_n, res.traditional_min_n, sav,
                         bool(res.poly_algebraic_wins))
        print(
            f"      rot={rot:<5d} poly={res.poly_algebraic_min_n}  "
            f"trad={res.traditional_min_n}  "
            f"savings={'None' if sav is None else format(sav, '.4f')}  "
            f"win={res.poly_algebraic_wins}"
        )
    state["a_rotsweep"] = rot_rows
    wins = sum(1 for v in rot_rows.values() if v[3])
    print(f"      -> criterion (a) win in {wins} of {len(rot_rows)} phase rotations")


def _stable_min_n(rows: list[tuple[int, float]], tolerance: float) -> int | None:
    """comparison.poly_algebraic_minimum_points' acceptance rule, applied to a
    trace this script computed itself. Same rule, re-typed only because the
    library function does not expose a way to swap the node-acceptance test."""
    for i, (_, dim) in enumerate(rows):
        if not math.isfinite(dim) or abs(dim - TRUE_DIMENSION) > tolerance:
            continue
        if all(
            math.isfinite(d) and abs(d - TRUE_DIMENSION) <= tolerance for _, d in rows[i:]
        ):
            return rows[i][0]
    return None


def section_7(state) -> None:
    """WHERE THE CHANGE CAME FROM. The record (Sec. 10.2 correction) has this
    problem as a 128-vs-128 tie under this same whole-orbit construction. It is
    now 64 vs 128. Something has to account for that, and "the library was
    updated" is an assertion until the specific mechanism is exhibited.

    The traditional side is untouched by N8/H1/H3, so the change must be on the
    poly side. `DimensionEstimate.is_well_fit(threshold, near_constant_consensus=
    False)` is EXACTLY the pre-N8 acceptance rule -- accept iff R^2 >= threshold,
    which the exactly-constant (degenerate) sentinel passes at 1.0 and a merely
    NEAR-constant sequence fails through the R2-F4 cancellation. Running both
    rules over the identical sampled node set at each n therefore isolates N8's
    contribution with nothing else changed.
    """
    print("\n" + "=" * 78)
    print("SECTION 7 -- provenance: which library change moved this problem")
    print("=" * 78)
    arms = state["arms"]
    print(
        f"\n  arm A, k={K}, max_radius={MAX_RADIUS}, identical sampled nodes per row.\n"
        f"  'pre-N8' = accept iff R^2 >= 0.9 (near-constant branch OFF);\n"
        f"  'current' = the shipped rule (near-constant branch, graph consensus)."
    )
    print(
        f"      {'n':>6} | {'pre-N8':>8} {'acc':>5} | {'current':>8} {'acc':>5} | "
        f"{'consensus':>9} {'near':>6}"
    )
    pre_rows, cur_rows = [], []
    for n in POW2_N_GRID:
        if n > len(arms["arm_a"]) or n <= K:
            break
        hg = knn_hypergraph(arms["arm_a"][:n], k=K, dedupe=True)
        nodes = _sampled_nodes(hg, min(n, 40))
        est = [local_dimension(hg, u, max_radius=MAX_RADIUS) for u in nodes]
        consensus = near_constant_consensus(est, threshold=0.9)
        cur = [e.dimension for e in est if e.is_well_fit(0.9, near_constant_consensus=consensus)]
        pre = [e.dimension for e in est if e.is_well_fit(0.9, near_constant_consensus=False)]
        near = sum(1 for e in est if e.near_degenerate) / len(est)
        cur_v = sum(cur) / len(cur) if cur else float("nan")
        pre_v = sum(pre) / len(pre) if pre else float("nan")
        pre_rows.append((n, pre_v))
        cur_rows.append((n, cur_v))
        print(
            f"      {n:>6} | {pre_v:8.4f} {len(pre):>5d} | {cur_v:8.4f} {len(cur):>5d} | "
            f"{str(consensus):>9} {near:6.3f}"
        )

    pre_min = _stable_min_n(pre_rows, TOLERANCE)
    cur_min = _stable_min_n(cur_rows, TOLERANCE)
    print(f"\n      poly min_n under the pre-N8 acceptance rule : {pre_min}")
    print(f"      poly min_n under the current rule           : {cur_min}")
    trad_note = state.get("a_arm_a", {}).get("trad_min_n", "run section 2")
    print(f"      traditional min_n (unaffected by N8)        : {trad_note}")
    state["provenance"] = {"pre_n8_min_n": pre_min, "current_min_n": cur_min}
    print(
        "\n  => the tie-to-win flip is attributable to N8 (the R2-F4 fix: near-constant\n"
        "     shell sequences are no longer silently discarded), not to the point-cloud\n"
        "     construction, not to a hyperparameter, and not to the baseline changing.\n"
        "     Note what that means for the CAVEAT: the extra accepted nodes at small n\n"
        "     are near-constant ones, so the win is MORE dependent on the F1 ring-lattice\n"
        "     regime than the old tie was, not less."
    )


SECTIONS = {
    1: section_1, 2: section_2, 3: section_3, 4: section_4, 5: section_5,
    6: section_6, 7: section_7,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    wanted = [int(s) for s in args.sections.split(",") if s.strip()]

    orbit = solve_orbit()
    arms = build_arms(orbit)
    # Orbit centre for the coverage statistic: centroid of the full one-period
    # sample, computed once and shared, so a prefix's angular coverage is
    # measured against the orbit rather than against itself.
    base = np.asarray(arms["arm_a"])
    orbit["centre"] = (float(base[:, 0].mean()), float(base[:, 1].mean()))
    state = {"orbit": orbit, "arms": arms}

    for s in wanted:
        SECTIONS[s](state)
    summary(state)


if __name__ == "__main__":
    main()
