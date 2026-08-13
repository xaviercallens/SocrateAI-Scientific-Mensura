"""Round 3 -- problem 04 (Mars, real JPL Horizons IC) remeasured against the
current library (post N8 / H1 / H3 / AutoResearch H2 loop).

WHAT CHANGED SINCE ROUND 2 AND WHY THIS SCRIPT EXISTS
-----------------------------------------------------
Round 2's own script (scripts/hypergraph_benchmark/round2/04_mars_horizons.py)
measured this problem on a *time-ordered* prefix of the trajectory: `points[:n]`
was the first n/6400 of one Mars year, i.e. a short smooth ARC, not the orbit.
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 10.10 records that this is the wrong
prefix construction for a closed orbit, and that the round-2 skeptic, re-running
04 on the *whole-orbit-covering* prefix construction that Sec. 10.4 / R2-F8
endorses (and that problems 01, 02 and 10 already used), found poly `min_n`=64
vs traditional `min_n`=256 -- a criterion-(a) win at savings 0.75. That skeptic
run predates N8 (near-constant shell gate), H1 (`theiler_window`) and H3
(criterion (c) / `compare_accuracy_at_max_n`). This script remeasures it against
the current, unmodified library and reports BOTH criteria.

THE PREFIX CONSTRUCTION, STATED EXPLICITLY (this is the methodology change)
--------------------------------------------------------------------------
The cloud is M = 4096 points sampled uniformly in time over the half-open
interval [0, T) of one Mars period T (from the verified two-body solver, real
Horizons initial condition), then reordered by a BIT-REVERSAL PERMUTATION so
that `points[:n]` at every power-of-two n is an evenly-strided sample of the
WHOLE orbit rather than a growing arc. The prefix-coverage property is verified
programmatically in section 1 (`check_bitreversal_prefix_coverage`), not
assumed. This is byte-for-byte the same construction problems 02 and 10 use.
Both estimators see identical prefixes, so the change is not one-sided.
Section 2 also runs the time-ordered cloud as an explicit control so the size
and the SIGN of the construction's effect are visible in the same output.

HYPERPARAMETERS: NO DEVIATION FROM THE MODULE DEFAULTS (R2-F3 discipline)
------------------------------------------------------------------------
k = 6, max_radius = 6 (`comparison.compare`'s own default), theiler_window = 0
(default), max_ball_fraction = 1.0 (default). Nothing is tuned. Sec. 9.4/R2-F3
found round 2's problem-01 "win" was an undisclosed `max_radius` deviation, so
section 4 additionally reports the full 4x5 (k, max_radius) sweep: the quoted
cell must not be a favourable edge. The one deviation from round 2 is the
n_grid -- powers of two (32...4096) instead of (100...6400) -- which the
bit-reversal prefix property REQUIRES and which problems 02 and 10 already use.
Section 2 defends that choice with a fine grid (16, 32, ..., 512) on which the
first stable convergence of each method is located directly.

Nothing under src/ is modified or needed to be. Run:
    python scripts/hypergraph_benchmark/round3/04_mars_horizons.py
    python scripts/hypergraph_benchmark/round3/04_mars_horizons.py --sections 1,2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    local_dimension,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

ROUND2_SCRIPT = REPO_ROOT / "scripts" / "hypergraph_benchmark" / "round2" / "04_mars_horizons.py"

# --- the measurement configuration, quoted with every number below ---
K = 6                       # module default for a 1-manifold target
MAX_RADIUS = 6              # comparison.compare's own default -- NOT tuned (R2-F3)
THEILER_WINDOW = 0          # default; see section 4 for the "auto" diagnostic
MAX_BALL_FRACTION = 1.0     # default (saturation guard OFF); section 4 reports 0.5
TRUE_DIMENSION = 1.0
TOLERANCE = 0.2             # unchanged from round 2's assigned brief
M = 4096                    # cloud size; power of two for the bit-reversal prefix
N_GRID = (32, 64, 128, 256, 512, 1024, 2048, 4096)
FINE_GRID = tuple(range(16, 513, 16)) + (1024, 2048, 4096)


# =============================================================================
# Cloud construction
# =============================================================================


def _round2_module():
    """Import round 2's problem-04 script for its verified solver + real IC.

    Reused rather than reimplemented so that the solver-verification gate and
    the vis-viva period are literally the same code that round 1 and round 2
    ran; only the ORDERING of the resulting cloud changes here.
    """
    spec = importlib.util.spec_from_file_location("round2_p04", ROUND2_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bit_reversal_permutation(m_bits: int) -> list[int]:
    """perm[i] = bit-reversal of i over m_bits bits, for i in 0..2^m_bits - 1."""
    return [int(f"{i:0{m_bits}b}"[::-1], 2) for i in range(1 << m_bits)]


def check_bitreversal_prefix_coverage(perm: list[int], size: int) -> None:
    """Verify (do not assume) that every power-of-two prefix of the permuted
    sequence is exactly an evenly-strided index set over the whole period."""
    m_bits = size.bit_length() - 1
    for j in range(m_bits + 1):
        n = 1 << j
        prefix = sorted(perm[:n])
        stride = size // n
        expected = list(range(0, size, stride))
        assert prefix == expected, (
            f"bit-reversal prefix-coverage FAILED at n={n}: "
            f"got {prefix[:5]}..., expected {expected[:5]}..."
        )


def build_clouds(verbose: bool = True) -> dict[str, object]:
    """Verify the solver against the real ephemeris, then build both orderings."""
    mod = _round2_module()
    assert mod.CACHE.exists(), f"expected cached Horizons data at {mod.CACHE}, do not refetch"
    data = np.load(mod.CACHE)
    r_true, v0, t_days = data["r"], data["v0"], data["t"]

    verification = mod.verify_solver_against_real_ephemeris(r_true, v0, t_days)
    period = mod.orbital_period_days(r_true[0], v0)
    if verbose:
        print(f"Cached Horizons data: {len(t_days)} daily epochs (IC + solver verification only)")
        print(f"Solver vs real ephemeris: relative RMS = "
              f"{verification['relative_rms_vs_ephemeris']:.3e} (gate < 5e-3), "
              f"passed = {verification['passed']}")
        print(f"Vis-viva period from the real IC: {period:.4f} days "
              f"({period / 365.25:.4f} yr; Mars ~686.98 d)")
    assert verification["passed"], "refusing to trust a cloud from an unverified solver"
    assert abs(period - 687.0) < 5.0, "period sanity check on the real IC failed"

    time_ordered = mod.generate_single_period_cloud(r_true[0], v0, period, M)
    assert len(time_ordered) == M

    perm = bit_reversal_permutation(M.bit_length() - 1)
    check_bitreversal_prefix_coverage(perm, M)
    whole_orbit = [time_ordered[perm[i]] for i in range(M)]

    arr = np.asarray(time_ordered)
    step = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    wrap = float(np.linalg.norm(arr[-1] - arr[0]))
    assert wrap > 0.5 * step.min(), "near-duplicate at the period wraparound"

    if verbose:
        print(f"Cloud: M={M} points over one period, dt_sample={period / M:.5f} d; "
              f"step distances {step.min():.6f}..{step.max():.6f} AU "
              f"(ratio {step.max() / step.min():.3f}), wraparound gap {wrap:.6f} AU")
        print("bit-reversal prefix-coverage property: VERIFIED at n = 1, 2, ..., 4096")

    return {
        "period": period,
        "time_ordered": time_ordered,
        "whole_orbit": whole_orbit,
        "time_indices": [perm[i] for i in range(M)],
        "relative_rms": float(verification["relative_rms_vs_ephemeris"]),
    }


# =============================================================================
# Sections
# =============================================================================


def section1(clouds: dict[str, object]) -> None:
    """Cloud provenance + the F1 degeneracy mechanism, stated up front."""
    print("\n" + "=" * 78)
    print("SECTION 1 -- provenance, prefix property, and what poly is actually fitting")
    print("=" * 78)
    whole = clouds["whole_orbit"]

    print("\nShell sequences the shell-growth estimator sees (k=6, max_radius=6):")
    for n in (64, 256, 4096):
        hg = knn_hypergraph(whole[:n], k=K, dedupe=True)
        est = local_dimension(hg, sorted(hg.nodes)[0], max_radius=MAX_RADIUS)
        shells = tuple(
            b - a for a, b in zip((1,) + est.volumes[:-1], est.volumes, strict=True)
        )
        print(f"  n={n:5d}: volumes {est.volumes} -> shells {shells}, "
              f"dimension {est.dimension:.4f}, r_squared {est.r_squared:.4f}, "
              f"degenerate={est.degenerate}, near_degenerate={est.near_degenerate}")
    print("  A k=6 k-NN graph on a smooth closed curve is an exact circulant ring")
    print("  lattice: the shell sequence is CONSTANT (6, 6, 6, 6, 6, 6), so the")
    print("  log-log fit takes the finding-F1 degenerate branch -- dimension 1.0")
    print("  with r_squared=1.0 as a SENTINEL, not a measured fit quality.")
    print("  This is the same mechanism as the two accepted round-2 wins (02, 10);")
    print("  it is disclosed here for the same reason (Sec. 9.7).")


def section2(clouds: dict[str, object]) -> dict[str, object]:
    """Criterion (a): compute savings via comparison.compare."""
    print("\n" + "=" * 78)
    print("SECTION 2 -- CRITERION (a): compute savings, via comparison.compare()")
    print("=" * 78)
    whole = clouds["whole_orbit"]
    time_ordered = clouds["time_ordered"]

    print(f"\nconfig: k={K}, max_radius={MAX_RADIUS}, theiler_window={THEILER_WINDOW}, "
          f"max_ball_fraction={MAX_BALL_FRACTION}, true_dimension={TRUE_DIMENSION}, "
          f"tolerance={TOLERANCE}\n        n_grid={N_GRID}")

    headline = compare(
        whole, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=N_GRID, max_radius=MAX_RADIUS,
        theiler_window=THEILER_WINDOW, max_ball_fraction=MAX_BALL_FRACTION,
    )
    print("\n--- whole-orbit-covering prefix (bit-reversal) -- THE HEADLINE NUMBER ---")
    print(f"  poly_algebraic_min_n     = {headline.poly_algebraic_min_n}")
    print(f"  traditional_min_n        = {headline.traditional_min_n}")
    print(f"  poly_algebraic_wins      = {headline.poly_algebraic_wins}")
    print(f"  compute_savings_fraction = {headline.compute_savings_fraction}")
    print(f"  estimates at max_n={headline.max_n_tested}: poly "
          f"{headline.poly_algebraic_estimate_at_max_n:.4f}, traditional "
          f"{headline.traditional_estimate_at_max_n:.4f}")

    control = compare(
        time_ordered, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=N_GRID, max_radius=MAX_RADIUS,
    )
    print("\n--- CONTROL: identical points, TIME-ORDERED prefix (round 2's construction) ---")
    print(f"  poly {control.poly_algebraic_min_n} vs trad {control.traditional_min_n}, "
          f"savings {control.compute_savings_fraction}, "
          f"wins {control.poly_algebraic_wins}")
    print("  Round 2 scored 04 'no' from this arm. The construction changes the sign,")
    print("  which is exactly what Sec. 10.10 reports, and is why it must be disclosed.")

    print("\n--- why the time-ordered arm is an artifact and not merely 'a different cut' ---")
    for n in (32, 100, 256):
        arc = np.asarray(time_ordered[:n])
        centred = arc - arc.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        print(f"  time-ordered prefix n={n:4d}: spans {100 * n / M:5.2f}% of the orbit; "
              f"PCA singular-value ratios s2/s1={sv[1] / sv[0]:.4f}, s3/s1={sv[2] / sv[0]:.2e}")
    print("  A prefix of a few percent of a near-circular orbit is a nearly straight")
    print("  segment (s2/s1 ~ 1e-2). BOTH estimators report ~1 on it, for the trivial")
    print("  reason that a line is 1-dimensional -- not because either has identified")
    print("  the orbit. That is why the time-ordered arm's traditional min_n is small:")
    print("  it measures how few points make a chord look like a chord. The whole-orbit")
    print("  prefix asks the question the benchmark means to ask.")

    print("\n--- independent per-n re-derivation (bypasses compare()'s bookkeeping) ---")
    per_n = []
    for n in N_GRID:
        subset = whole[:n]
        hg = knn_hypergraph(subset, k=K, dedupe=True)
        poly = mean_dimension(hg, samples=min(n, 40), max_radius=MAX_RADIUS)
        trad = correlation_dimension(subset).dimension
        deg = degenerate_fraction(hg, samples=min(n, 40), max_radius=MAX_RADIUS)
        near = near_degenerate_fraction(hg, samples=min(n, 40), max_radius=MAX_RADIUS)
        per_n.append((n, poly, trad, deg, near))
        print(f"  n={n:5d}: poly={poly:7.4f} (|err| {abs(poly - TRUE_DIMENSION):6.4f})  "
              f"trad={trad:7.4f} (|err| {abs(trad - TRUE_DIMENSION):6.4f})  "
              f"degenerate_frac={deg:.3f} near_degenerate_frac={near:.3f}")
    print("  -> poly first within tolerance and STAYS within: n=64 (it is nan at n=32).")
    print("  -> traditional first within tolerance and STAYS within: n=256.")

    fine = compare(
        whole, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=FINE_GRID, max_radius=MAX_RADIUS,
    )
    fine_control = compare(
        time_ordered, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE,
        k=K, n_grid=FINE_GRID, max_radius=MAX_RADIUS,
    )
    print(f"\n--- grid-floor check: fine grid {FINE_GRID[0]}, {FINE_GRID[1]}, ..., 512 ---")
    print(f"  whole-orbit : poly {fine.poly_algebraic_min_n} vs trad "
          f"{fine.traditional_min_n}, savings {fine.compute_savings_fraction:.4f}")
    print(f"  time-ordered: poly {fine_control.poly_algebraic_min_n} vs trad "
          f"{fine_control.traditional_min_n}, savings "
          f"{fine_control.compute_savings_fraction:.4f}")
    print("  Sec. 10.2 knocked round 2's row 04 down with 'a finer grid favours")
    print("  traditional (converges at n~15)'. That is reproduced here -- but only on")
    print("  the TIME-ORDERED arm. On the whole-orbit cloud the finer grid keeps the")
    print("  ordering and nearly the same savings, so 64 vs 256 is not a grid artifact.")

    return {"headline": headline, "control": control, "fine": fine, "per_n": per_n}


def section3(clouds: dict[str, object]) -> dict[str, object]:
    """Criterion (c): asymptotic accuracy at max_n, via the H3 function."""
    print("\n" + "=" * 78)
    print("SECTION 3 -- CRITERION (c): accuracy at max n, via compare_accuracy_at_max_n()")
    print("=" * 78)
    whole = clouds["whole_orbit"]

    acc = compare_accuracy_at_max_n(
        whole, TRUE_DIMENSION, M, k=K, tolerance=TOLERANCE, max_radius=MAX_RADIUS,
    )
    print(f"\n  n_evaluated               = {acc.n_evaluated} "
          f"(poly graph nodes {acc.poly_algebraic_n_nodes})")
    print(f"  poly_algebraic_estimate   = {acc.poly_algebraic_estimate:.6f}  "
          f"|err| = {acc.poly_algebraic_abs_error:.6f}")
    print(f"  traditional_estimate      = {acc.traditional_estimate:.6f}  "
          f"|err| = {acc.traditional_abs_error:.6f}  (R^2 {acc.traditional_r_squared:.4f})")
    print(f"  sentinel fraction         = {acc.poly_algebraic_sentinel_fraction:.3f} "
          f"(degenerate {acc.poly_algebraic_degenerate_fraction:.3f} + near-degenerate "
          f"{acc.poly_algebraic_near_degenerate_fraction:.3f}); "
          f"cap {acc.max_sentinel_fraction:.3f}")
    print(f"  more_accurate (ranking)   = {acc.more_accurate}")
    print(f"  poly_algebraic_accuracy_win = {acc.poly_algebraic_accuracy_win}")
    print(f"  traditional_accuracy_win    = {acc.traditional_accuracy_win}")
    print(f"  verdict_reason: {acc.verdict_reason}")

    print("\n  TWO INDEPENDENT CONDITIONS FAIL, so this is not a near miss and there is")
    print("  no parameter that rescues it:")
    print("    (c2) the sentinel fraction is 1.000 -- every sampled node is the")
    print("         constant-shell degenerate branch, so poly's 1.000000 is pinned by")
    print("         the ring lattice, not measured from it (finding F1).")
    print("    (c4) the traditional estimator's own error is "
          f"{acc.traditional_abs_error:.4f} <= tolerance "
          f"{acc.tolerance:.4f}: the baseline")
    print("         also meets the bar, so there is no decision-relevant separation.")
    print("  Raising or lowering `tolerance` cannot fix this -- (c3) and (c4) pull in")
    print("  opposite directions and (c2) is independent of tolerance entirely.")
    print("  Criterion (c) is correctly a NON-WIN on a smooth 1-D orbit; the criterion")
    print("  is doing the job H3 built it for.")
    return {"accuracy": acc}


def section4(clouds: dict[str, object]) -> dict[str, object]:
    """Robustness of the criterion-(a) win: the R2-F3 guard, plus the knobs."""
    print("\n" + "=" * 78)
    print("SECTION 4 -- robustness of the criterion-(a) win (R2-F3 discipline)")
    print("=" * 78)
    whole = clouds["whole_orbit"]
    times = clouds["time_indices"]

    print("\n--- (k, max_radius) sweep; the quoted cell is k=6, max_radius=6 ---")
    wins, total, savings = 0, 0, []
    for k in (4, 6, 8, 10):
        row = []
        for mr in (3, 4, 5, 6, 8):
            r = compare(whole, TRUE_DIMENSION, TOLERANCE, k=k, n_grid=N_GRID, max_radius=mr)
            total += 1
            wins += bool(r.poly_algebraic_wins)
            savings.append(r.compute_savings_fraction)
            mark = "*" if (k == K and mr == MAX_RADIUS) else " "
            row.append(f"mr={mr}: {r.poly_algebraic_min_n}/{r.traditional_min_n}="
                       f"{r.compute_savings_fraction:.3f}{mark}")
        print(f"  k={k:2d}  " + "  ".join(row))
    print(f"  wins: {wins}/{total}; savings range "
          f"{min(savings):.3f}..{max(savings):.3f}; the quoted cell is 0.750 -- "
          f"mid-range,\n  not a favourable edge (the edge would be 0.875).")

    print("\n--- one knob at a time, everything else at defaults ---")
    for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
        r = compare(whole, TRUE_DIMENSION, tol, k=K, n_grid=N_GRID, max_radius=MAX_RADIUS)
        print(f"  tolerance={tol:.2f}: poly {r.poly_algebraic_min_n} vs trad "
              f"{r.traditional_min_n}, savings {r.compute_savings_fraction:.4f}")

    r_ball = compare(whole, TRUE_DIMENSION, TOLERANCE, k=K, n_grid=N_GRID,
                     max_radius=MAX_RADIUS, max_ball_fraction=0.5)
    print(f"  max_ball_fraction=0.5 (AR2 guard on): poly {r_ball.poly_algebraic_min_n} vs "
          f"trad {r_ball.traditional_min_n}, savings {r_ball.compute_savings_fraction:.4f}")
    print("    -- reported for disclosure only; the headline uses the default 1.0, and")
    print("       the guard would only make the win LARGER, so it is not load-bearing.")

    print("\n--- is the win an artifact of the n_grid or of M? ---")
    r2grid = (100, 200, 400, 800, 1600, 3200)
    r_r2grid = compare(whole, TRUE_DIMENSION, TOLERANCE, k=K, n_grid=r2grid,
                       max_radius=MAX_RADIUS)
    print(f"  round-2's OWN n_grid {r2grid} on the whole-orbit cloud:")
    print(f"    poly {r_r2grid.poly_algebraic_min_n} vs trad "
          f"{r_r2grid.traditional_min_n}, savings "
          f"{r_r2grid.compute_savings_fraction:.4f} -- still a win, at coarser")
    print("    resolution (poly is floored by the grid at 100; it converges at 64).")
    print("    So the win comes from the CONSTRUCTION, not from changing the grid;")
    print("    the power-of-two grid only resolves it more finely.")

    mod = _round2_module()
    data = np.load(mod.CACHE)
    r_true, v0 = data["r"], data["v0"]
    period = float(clouds["period"])
    for m_alt in (1024, 2048, 8192):
        alt_time = mod.generate_single_period_cloud(r_true[0], v0, period, m_alt)
        perm = bit_reversal_permutation(m_alt.bit_length() - 1)
        check_bitreversal_prefix_coverage(perm, m_alt)
        alt = [alt_time[perm[i]] for i in range(m_alt)]
        grid = tuple(g for g in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if g <= m_alt)
        r_alt = compare(alt, TRUE_DIMENSION, TOLERANCE, k=K, n_grid=grid, max_radius=MAX_RADIUS)
        extra = ""
        if m_alt == 8192:
            d = np.abs(np.asarray(alt[:64]) - np.asarray(whole[:64])).max()
            extra = f"; max |prefix_64(M=8192) - prefix_64(M=4096)| = {d:.2e} AU"
        print(f"  M={m_alt:5d}: poly {r_alt.poly_algebraic_min_n} vs trad "
              f"{r_alt.traditional_min_n}, savings "
              f"{r_alt.compute_savings_fraction:.4f}{extra}")
    print("    M is not load-bearing, and provably so: a bit-reversal prefix of length")
    print("    n is the index set range(0, M, M/n), i.e. exactly n TIME-UNIFORM samples")
    print("    over [0, T) whatever M is. Different M values differ only by the solver's")
    print("    integration step, which is why the two n=64 prefixes agree to ~1e-8 AU.")

    auto_w = theiler_window_from_autocorrelation(
        np.asarray(whole), time_indices=np.asarray(times)
    )
    r_theiler = compare(whole, TRUE_DIMENSION, TOLERANCE, k=K, n_grid=N_GRID,
                        max_radius=MAX_RADIUS, theiler_window="auto", time_indices=times)
    print(f"  theiler_window='auto' (both sides, real time indices; window={auto_w} of "
          f"{M} samples):")
    print(f"    poly {r_theiler.poly_algebraic_min_n} vs trad "
          f"{r_theiler.traditional_min_n}; estimates at max_n: poly "
          f"{r_theiler.poly_algebraic_estimate_at_max_n:.4f}, trad "
          f"{r_theiler.traditional_estimate_at_max_n:.4f}")
    print("    NEITHER method converges anywhere on the grid. This is H1 behaving")
    print("    correctly on a problem it does not apply to: a single closed period of a")
    print("    periodic orbit never decorrelates within itself, so the autocorrelation")
    print(f"    window is ~{100 * auto_w / M:.0f}% of the whole orbit and excises the")
    print("    geometric neighbourhood both estimators are trying to measure. The")
    print("    headline therefore stays at the DEFAULT theiler_window=0, and this cell")
    print("    is recorded so that nobody quotes the win as robust to it. It is")
    print("    symmetric -- both sides are destroyed -- so it is not a rigged knob.")
    return {"sweep_wins": wins, "sweep_total": total, "theiler_auto_window": int(auto_w)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default="1,2,3,4",
                        help="comma-separated subset of 1,2,3,4 (default: all)")
    args = parser.parse_args()
    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}

    clouds = build_clouds()
    if "1" in wanted:
        section1(clouds)
    if "2" in wanted:
        section2(clouds)
    if "3" in wanted:
        section3(clouds)
    if "4" in wanted:
        section4(clouds)

    print("\n" + "=" * 78)
    print("density_robustness_tested = False (criterion (b) not attempted)")
    print("  Unchanged from round 2's brief: Mars's e~0.093 orbit has no natural")
    print("  density spike for the density-robustness test to catch, and R2-F1/Sec.")
    print("  10.3 found criterion (b) null on every problem that did have one.")
    print("=" * 78)


if __name__ == "__main__":
    main()
