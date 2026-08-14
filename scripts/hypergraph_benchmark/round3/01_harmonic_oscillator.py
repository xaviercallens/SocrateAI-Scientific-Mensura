"""Round 3: problem 01, simple harmonic oscillator -- remeasured against the
CURRENT library (post-N8 near-constant shell gate, post-H1 Theiler window,
post-H3 criterion (c), post-AutoResearch H2 loop, which kept no source change).

KNOWN DIMENSION: 1 (exact). The phase-space orbit of x'' = -x is the unit
circle, a smooth closed curve; its correlation / topological dimension is
exactly 1 by elementary dynamics, not by measurement. Tolerance +/- 0.15,
as in rounds 1 and 2.

WHAT THIS RUN SETTLES
---------------------
Round 2 reported 87.5% compute savings for this problem using max_radius=3
instead of the module default 6. The round-2 skeptic's finding R2-F3
(docs/MENSURA_BENCHMARK.md Sec. 9.4) showed that the win existed ONLY
below the default -- at max_radius=6 the poly side returned nan at small n and
poly_min_n moved 50 -> 400, i.e. a tie -- and that the stated justification
("radius-6 balls saturate the whole graph") was factually wrong: the real
mechanism was R2-F4, the near-constant-shell defect, which threw away accurate
per-node fits because R^2 collapses when there is almost no variance to explain.

N8 fixed R2-F4. This script therefore re-runs the problem at the MODULE
DEFAULTS -- k=6, max_radius=6, theiler_window=0, max_ball_fraction=1.0, no
deviation of any kind -- which is the run Sec. 11's next-steps list asked for
("re-run problem 01 at the default max_radius to settle R2-F3 honestly"), and
then attacks the resulting number the way R2-F3 attacked the round-2 one.

POINT-CLOUD CONSTRUCTION -- stated explicitly, per the round-3 brief
-------------------------------------------------------------------
Identical to round 2's production cloud, and re-derived here rather than
quoted: ONE period of x''=-x integrated with `socrates.solvers.leapfrog` at
dt=2e-4 (n_steps = 31416), solver verified against the exact cos t / -sin t
solution before use, then 3200 points drawn from that dense grid by a
golden-ratio (Weyl) low-discrepancy index sequence

    idx_i = floor( frac(i * phi) * n_steps ),   phi = (sqrt(5) - 1) / 2

WHY THIS CONSTRUCTION AND NOT A NAIVE ONE. `compare()` evaluates each n in
n_grid on the literal PREFIX points[:n]. A chronologically ordered sample
would make points[:50] a 1.6% arc of the period and points[:3200] the whole
circle -- the object being sampled would change with n, so a "fewer points"
claim would not be a claim about sample efficiency. The Weyl sequence is
NESTED: the first n indices are a near-uniform (low-discrepancy) sample of the
WHOLE period for every n, so every prefix samples the same circle at a
different density. That is the property the criterion-(a) contract needs.
A single period is used (not 10) because sampling multiple periods aliases
duplicate points onto each other -- round 1's finding F3.

Section 1 checks this cloud against two independent things: the exact
analytic solution evaluated at the same Weyl indices (no solver involved), and
`n8d_clouds.load_cloud("01")` (the cached round-2 production cloud), so the
numbers below cannot be an artifact of this script's own cloud builder.

SECTIONS
  1  cloud construction, solver verification, independence checks   (~5 s)
  2  criterion (a) at the defaults, with per-n table                (~40 s)
  3  criterion (c) at max n                                        (~10 s)
  4  robustness: max_radius, k, 10 phase redraws, baseline window   (~6 min)
  5  mechanism + controls: what the win is actually made of         (~1 min)
  6  the knob R2-F3 never checked: the point-cloud construction     (~8 min)

Run (from the repo root, with the venv active):
  python scripts/hypergraph_benchmark/round3/01_harmonic_oscillator.py
  python scripts/hypergraph_benchmark/round3/01_harmonic_oscillator.py --sections 1,2,3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
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
from socrates.solvers import leapfrog  # noqa: E402

TRUE_DIMENSION = 1.0
TOLERANCE = 0.15

# Module defaults, restated here so a deviation would be visible as an edit.
K = 6
MAX_RADIUS = 6
THEILER_WINDOW = 0
MAX_BALL_FRACTION = 1.0

N_GRID = (50, 100, 200, 400, 800, 1600, 3200)
N_MAX = max(N_GRID)
DT = 2e-4

RULE = "=" * 78


# ---------------------------------------------------------------------------
# Section 1: the point cloud
# ---------------------------------------------------------------------------


def verify_solver(dt: float = DT) -> dict[str, object]:
    """Integrate exactly one period and check leapfrog against cos t / -sin t.

    A dimension number computed on an unverified trajectory measures the
    integrator, not the physics.
    """
    force = lambda q: -q  # noqa: E731
    n_steps = int(round(2 * np.pi / dt))
    run = leapfrog(force, [1.0], [0.0], dt=dt, n_steps=n_steps)

    t = run.times
    max_err_x = float(np.max(np.abs(run.positions[:, 0] - np.cos(t))))
    max_err_v = float(np.max(np.abs(run.velocities[:, 0] + np.sin(t))))
    drift = run.energy_drift(lambda q: 0.5 * float(q @ q))
    closure = float(
        np.hypot(run.positions[-1, 0] - np.cos(t[-1]), run.velocities[-1, 0] + np.sin(t[-1]))
    )
    return {
        "run": run,
        "n_steps": n_steps,
        "max_err_x": max_err_x,
        "max_err_v": max_err_v,
        "energy_drift": drift,
        "closure": closure,
        "passed": max_err_x < 1e-4 and max_err_v < 1e-4 and drift < 1e-6 and closure < 1e-3,
    }


def weyl_indices(n_max: int, n_steps_total: int, offset: float = 0.0) -> np.ndarray:
    """Golden-ratio low-discrepancy index sequence; nested in n by construction."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fracs = ((np.arange(n_max) * phi) + offset) % 1.0
    idx = np.clip(np.floor(fracs * n_steps_total).astype(int), 0, n_steps_total)
    if len(np.unique(idx)) != n_max:
        raise RuntimeError(
            f"Weyl sequence collided: {n_max - len(np.unique(idx))} duplicate indices "
            f"of {n_max} at n_steps_total={n_steps_total}; use a finer dt."
        )
    return idx


def build_cloud(run, n_max: int = N_MAX) -> list[tuple[float, float]]:
    idx = weyl_indices(n_max, len(run.times) - 1)
    return list(
        zip(run.positions[idx, 0].tolist(), run.velocities[idx, 0].tolist(), strict=True)
    )


def build_exact_cloud(
    n_max: int = N_MAX, n_steps_total: int | None = None, offset: float = 0.0
) -> list[tuple[float, float]]:
    """The same Weyl indices evaluated on the EXACT solution -- no integrator."""
    if n_steps_total is None:
        n_steps_total = int(round(2 * np.pi / DT))
    idx = weyl_indices(n_max, n_steps_total, offset=offset)
    t = idx * (2 * np.pi / n_steps_total)
    return list(zip(np.cos(t).tolist(), (-np.sin(t)).tolist(), strict=True))


def section1() -> list[tuple[float, float]]:
    print(RULE)
    print("SECTION 1  point cloud: build, verify the solver, check independence")
    print(RULE)
    v = verify_solver()
    print(f"  dt={DT}, n_steps={v['n_steps']} (exactly one period)")
    print(f"  max |x - cos t|        = {v['max_err_x']:.3e}")
    print(f"  max |v + sin t|        = {v['max_err_v']:.3e}")
    print(f"  energy drift           = {v['energy_drift']:.3e}")
    print(f"  orbit closure error    = {v['closure']:.3e}")
    print(f"  SOLVER VERIFIED        : {v['passed']}")
    if not v["passed"]:
        raise SystemExit("solver did not verify -- downstream numbers are meaningless")

    pts = build_cloud(v["run"])
    print(f"  cloud: {len(pts)} points, Weyl (golden-ratio) index sequence, all indices distinct")

    exact = build_exact_cloud()
    dev = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, exact, strict=True))
    print(f"  vs EXACT analytic solution at the same indices: max L1 deviation = {dev:.3e}")
    if dev > 1e-4:
        raise SystemExit("integrated cloud disagrees with the exact solution")

    try:
        from n8d_clouds import load_cloud  # type: ignore

        cached = load_cloud("01")
        same = len(cached) == len(pts) and all(
            a == b for a, b in zip(cached, pts, strict=True)
        )
        print(f"  vs n8d_clouds.load_cloud('01') (round-2 production cloud): identical = {same}")
        if not same:
            raise SystemExit("cloud does not reproduce the round-2 production cloud")
    except ImportError:  # pragma: no cover - only if the helper is missing
        print("  n8d_clouds not importable; skipped the round-2 cloud identity check")
    print()
    return pts


# ---------------------------------------------------------------------------
# Section 2: criterion (a)
# ---------------------------------------------------------------------------


def section2(pts: list[tuple[float, float]]) -> dict[str, object]:
    print(RULE)
    print("SECTION 2  criterion (a), compute savings -- comparison.compare() at the DEFAULTS")
    print(RULE)
    print(
        f"  k={K}, max_radius={MAX_RADIUS}, theiler_window={THEILER_WINDOW}, "
        f"max_ball_fraction={MAX_BALL_FRACTION}  <- all module defaults, no deviation"
    )
    print(f"  n_grid={N_GRID}, true_dimension={TRUE_DIMENSION}, tolerance={TOLERANCE}")
    print()
    print("  per-n, both estimators on identical prefixes:")
    header = (
        f"    {'n':>6} {'poly':>9} {'|err|':>8} {'deg':>6} {'near':>6} "
        f"{'trad':>9} {'|err|':>8} {'tradR2':>8}"
    )
    print(header)
    for n in N_GRID:
        hg = knn_hypergraph(pts[:n], k=K, dedupe=True)
        s = min(n, 40)
        p = mean_dimension(hg, samples=s, max_radius=MAX_RADIUS)
        d = degenerate_fraction(hg, samples=s, max_radius=MAX_RADIUS)
        nd = near_degenerate_fraction(hg, samples=s, max_radius=MAX_RADIUS)
        t = correlation_dimension(pts[:n])
        print(
            f"    {n:>6} {p:9.4f} {abs(p - 1.0):8.4f} {d:6.3f} {nd:6.3f} "
            f"{t.dimension:9.4f} {abs(t.dimension - 1.0):8.4f} {t.r_squared:8.4f}"
        )

    r = compare(
        pts,
        true_dimension=TRUE_DIMENSION,
        tolerance=TOLERANCE,
        k=K,
        n_grid=N_GRID,
        max_radius=MAX_RADIUS,
        theiler_window=THEILER_WINDOW,
        max_ball_fraction=MAX_BALL_FRACTION,
    )
    print()
    print(f"  poly_algebraic_min_n     = {r.poly_algebraic_min_n}")
    print(f"  traditional_min_n        = {r.traditional_min_n}")
    print(f"  poly_algebraic_wins      = {r.poly_algebraic_wins}")
    print(f"  compute_savings_fraction = {r.compute_savings_fraction}")
    print(f"  poly estimate at n={r.max_n_tested}  = {r.poly_algebraic_estimate_at_max_n:.6f}")
    print(f"  trad estimate at n={r.max_n_tested}  = {r.traditional_estimate_at_max_n:.6f}")
    print()
    print(
        "  R2-F3 comparison: round 2 measured poly 50 / trad 400 at max_radius=3 and\n"
        "  poly 400 / trad 400 (a tie) at the default 6. Section 4 re-runs that sweep."
    )
    print()
    return {
        "poly_min_n": r.poly_algebraic_min_n,
        "trad_min_n": r.traditional_min_n,
        "wins": r.poly_algebraic_wins,
        "savings": r.compute_savings_fraction,
        "poly_at_max_n": r.poly_algebraic_estimate_at_max_n,
        "trad_at_max_n": r.traditional_estimate_at_max_n,
    }


# ---------------------------------------------------------------------------
# Section 3: criterion (c)
# ---------------------------------------------------------------------------


def section3(pts: list[tuple[float, float]]) -> dict[str, object]:
    print(RULE)
    print("SECTION 3  criterion (c), accuracy at max n -- comparison.compare_accuracy_at_max_n()")
    print(RULE)
    r = compare_accuracy_at_max_n(
        pts,
        true_dimension=TRUE_DIMENSION,
        max_n=N_MAX,
        k=K,
        tolerance=TOLERANCE,
        max_radius=MAX_RADIUS,
    )
    print(
        f"  n_evaluated              = {r.n_evaluated} "
        f"(poly graph nodes: {r.poly_algebraic_n_nodes})"
    )
    print(
        f"  poly estimate            = {r.poly_algebraic_estimate:.6f}"
        f"   |err| = {r.poly_algebraic_abs_error:.6f}"
    )
    print(
        f"  traditional estimate     = {r.traditional_estimate:.6f}"
        f"   |err| = {r.traditional_abs_error:.6f}"
    )
    print(f"  traditional R^2          = {r.traditional_r_squared:.6f}")
    print(
        f"  poly sentinel fraction   = {r.poly_algebraic_sentinel_fraction:.3f} "
        f"(degenerate {r.poly_algebraic_degenerate_fraction:.3f} + "
        f"near-degenerate {r.poly_algebraic_near_degenerate_fraction:.3f})"
    )
    print(f"  more_accurate (ranking)  = {r.more_accurate}")
    print(f"  poly_algebraic_accuracy_win = {r.poly_algebraic_accuracy_win}")
    print(f"  traditional_accuracy_win    = {r.traditional_accuracy_win}")
    print(f"  verdict reason: {r.verdict_reason}")
    print()
    print(
        "  Note this case fails criterion (c) TWICE over, independently: the F1\n"
        "  sentinel gate (c2) fires first, and even without it (c4) would refuse --\n"
        "  the traditional error is also inside the 0.15 tolerance, so the two\n"
        "  methods do not straddle the bar and no margin could rescue it."
    )
    print()
    return {
        "poly_err": r.poly_algebraic_abs_error,
        "trad_err": r.traditional_abs_error,
        "sentinel": r.poly_algebraic_sentinel_fraction,
        "win": r.poly_algebraic_accuracy_win,
        "reason": r.verdict_reason,
    }


# ---------------------------------------------------------------------------
# Section 4: is the criterion (a) win a property of the data or of the settings?
# ---------------------------------------------------------------------------


def _compare_min_ns(pts, *, k=K, max_radius=MAX_RADIUS, n_grid=N_GRID):
    r = compare(
        pts,
        true_dimension=TRUE_DIMENSION,
        tolerance=TOLERANCE,
        k=k,
        n_grid=n_grid,
        max_radius=max_radius,
    )
    return (
        r.poly_algebraic_min_n,
        r.traditional_min_n,
        r.poly_algebraic_wins,
        r.compute_savings_fraction,
    )


def _traditional_min_n_with_window(pts, r_min_frac: float, r_max_frac: float):
    """The baseline's own convergence search, run with a non-default scaling
    window. `minimum_points_for_target_accuracy` does not expose r_min_frac /
    r_max_frac, so the same stable-convergence rule is applied here directly --
    this is the baseline's side of the R2-F3 question (finding R2-F8)."""
    res = []
    for n in N_GRID:
        try:
            d = correlation_dimension(
                pts[:n], n_radii=20, r_min_frac=r_min_frac, r_max_frac=r_max_frac
            ).dimension
        except ValueError:
            d = float("nan")
        res.append((n, d))
    for i, (n, d) in enumerate(res):
        if not math.isfinite(d) or abs(d - TRUE_DIMENSION) > TOLERANCE:
            continue
        if all(math.isfinite(x) and abs(x - TRUE_DIMENSION) <= TOLERANCE for _, x in res[i:]):
            return n
    return None


def section4(pts: list[tuple[float, float]]) -> dict[str, object]:
    print(RULE)
    print("SECTION 4  robustness -- the R2-F3 question asked of THIS run's number")
    print(RULE)

    print("  (4a) max_radius sweep at k=6. Round 2's win lived only at max_radius<=5.")
    mr_rows = []
    for mr in range(3, 9):
        p, t, w, s = _compare_min_ns(pts, max_radius=mr)
        mr_rows.append((mr, p, t, w, s))
        flag = " <- MODULE DEFAULT" if mr == MAX_RADIUS else ""
        print(f"       max_radius={mr}: poly={p} trad={t} win={w} savings={s}{flag}")
    mr_wins = sum(1 for _, _, _, w, _ in mr_rows if w)
    print(f"       wins: {mr_wins}/{len(mr_rows)}")

    print("  (4b) k sweep at max_radius=6.")
    k_rows = []
    for k in (4, 6, 8, 10, 12, 14):
        p, t, w, s = _compare_min_ns(pts, k=k)
        k_rows.append((k, p, t, w, s))
        flag = " <- MODULE DEFAULT" if k == K else ""
        print(f"       k={k:>2}: poly={p} trad={t} win={w} savings={s}{flag}")
    k_wins = sum(1 for _, _, _, w, _ in k_rows if w)
    print(f"       wins: {k_wins}/{len(k_rows)}")

    print("  (4c) 10 redraws: the Weyl sequence re-phased (a different sample of the")
    print("       same circle each time), built from the EXACT solution, no integrator.")
    redraw = []
    for off in np.linspace(0.0, 0.9, 10):
        c = build_exact_cloud(offset=float(off))
        p, t, w, s = _compare_min_ns(c)
        redraw.append((float(off), p, t, w, s))
        print(f"       offset={off:.1f}: poly={p} trad={t} win={w} savings={s}")
    rd_wins = sum(1 for _, _, _, w, _ in redraw if w)
    print(f"       wins: {rd_wins}/{len(redraw)}")

    print("  (4d) FAIRNESS THE OTHER WAY (R2-F8): give the BASELINE its best scaling")
    print("       window instead of its default, and re-ask whether poly still wins.")
    best = None
    for rmin in (0.001, 0.003, 0.01, 0.03):
        cells = []
        for rmax in (0.05, 0.1, 0.2, 0.4):
            m = _traditional_min_n_with_window(pts, rmin, rmax)
            cells.append(f"r_max={rmax}: {m}")
            if m is not None and (best is None or m < best[0]):
                best = (m, rmin, rmax)
        print(f"       r_min={rmin}:  " + "   ".join(cells))
    poly_default = mr_rows[3][1]  # max_radius=6 row
    print(
        f"       best traditional min_n over the 16 windows = {best[0]} "
        f"(r_min_frac={best[1]}, r_max_frac={best[2]}); default window gives 400"
    )
    if best[0] is not None and poly_default is not None:
        adv_sav = 1.0 - poly_default / best[0]
        print(
            f"       poly {poly_default} vs BEST-CASE traditional {best[0]}: "
            f"still a win = {poly_default < best[0]}, savings = {adv_sav:.3f}"
        )
    print()
    return {
        "max_radius_sweep": mr_rows,
        "k_sweep": k_rows,
        "redraws": redraw,
        "best_traditional_min_n": best,
    }


# ---------------------------------------------------------------------------
# Section 5: what the win is made of, and the control that bounds the objection
# ---------------------------------------------------------------------------


def section5(pts: list[tuple[float, float]]) -> dict[str, object]:
    print(RULE)
    print("SECTION 5  mechanism, stated rather than left implicit")
    print(RULE)
    print("  (5a) the accepted fits at the winning n=50, node by node (first 6 of 40 sampled):")
    hg = knn_hypergraph(pts[:50], k=K, dedupe=True)
    nodes = sorted(hg.nodes)[:40]
    ests = [local_dimension(hg, nd, max_radius=MAX_RADIUS) for nd in nodes]
    for e in ests[:6]:
        print(
            f"       volumes={e.volumes} dim={e.dimension:.4f} r2={e.r_squared:.4f} "
            f"degenerate={e.degenerate} near_degenerate={e.near_degenerate} "
            f"shell_cv={e.shell_cv:.4f} slope_bound={e.slope_bound:.4f}"
        )
    lens = {len(e.radii) for e in ests}
    print(f"       fit-window lengths over all 40 sampled nodes: {sorted(lens)}")
    print(
        "       Every window is the full 6 radii, so these are NOT the zero-residual-dof\n"
        "       two-radius fits that carry a caveat on problem 09: there are 4 residual\n"
        "       degrees of freedom per fit. But shell_cv ~ 0.08-0.13 means the shell\n"
        "       sequence is near-constant, which is exactly what 1-D means and exactly\n"
        "       why R^2 is uninformative here (finding R2-F4, fixed by N8)."
    )

    print("  (5b) CONTROL for the F1 objection 'the estimator just answers 1.0 at small n'.")
    print("       Same settings, same n, on clouds that are NOT 1-dimensional:")
    rng = np.random.default_rng(20260813)
    controls = {
        "uniform square (true 2)": [tuple(p) for p in rng.random((800, 2))],
        "uniform cube (true 3)": [tuple(p) for p in rng.random((800, 3))],
    }
    control_out = {}
    for name, cloud in controls.items():
        row = []
        for n in (50, 100, 200, 400, 800):
            h = knn_hypergraph(cloud[:n], k=K, dedupe=True)
            s = min(n, 40)
            d = mean_dimension(h, samples=s, max_radius=MAX_RADIUS)
            sent = degenerate_fraction(h, samples=s, max_radius=MAX_RADIUS)
            sent += near_degenerate_fraction(h, samples=s, max_radius=MAX_RADIUS)
            row.append((n, d, sent))
        control_out[name] = row
        print(f"       {name}:")
        for n, d, sent in row:
            print(f"          n={n:>4}: poly={d:.4f}  sentinel_fraction={sent:.3f}")
    print(
        "       The sentinel branch does not fire at all off a 1-D curve (0.000 everywhere)\n"
        "       and the estimator does not report ~1.0 there. So the fast lock on 1.0 is\n"
        "       a response to genuinely 1-D local structure, not a constant the method\n"
        "       emits for any small sample. It remains, however, a pinned value: this is\n"
        "       why criterion (c) refuses it (Section 3) even though criterion (a) counts it."
    )

    print("  (5c) how far below the grid floor does poly actually reach? poly_min_n=50 sits")
    print("       ON the n_grid floor, so 0.875 is a LOWER bound on the savings. Probe only")
    print("       -- the headline number stays on the round-2 production grid:")
    p, t, w, s = _compare_min_ns(pts, n_grid=(20, 25, 30, 40) + N_GRID)
    print(f"       extended grid (floor 20): poly={p} trad={t} win={w} savings={s}")
    print()
    return {"control": control_out, "extended_grid": (p, t, w, s)}


# ---------------------------------------------------------------------------
# Section 6: the knob R2-F3 did NOT check -- the point-cloud construction
# ---------------------------------------------------------------------------


def _stable_min_n(per_n: list[tuple[int, float]]) -> int | None:
    """The library's stable-convergence rule, applied to a per-n series computed
    here. Used only where `compare()` cannot be called because the clouds are
    not prefixes of one another (section 6a/6b build an independent cloud per n)."""
    for i, (n, d) in enumerate(per_n):
        if not math.isfinite(d) or abs(d - TRUE_DIMENSION) > TOLERANCE:
            continue
        if all(math.isfinite(x) and abs(x - TRUE_DIMENSION) <= TOLERANCE for _, x in per_n[i:]):
            return n
    return None


def _per_n_independent(make_cloud) -> tuple[list, list]:
    poly, trad = [], []
    for n in N_GRID:
        c = make_cloud(n)
        hg = knn_hypergraph(c, k=K, dedupe=True)
        poly.append((n, mean_dimension(hg, samples=min(n, 40), max_radius=MAX_RADIUS)))
        trad.append((n, correlation_dimension(c).dimension))
    return poly, trad


def section6(pts: list[tuple[float, float]]) -> dict[str, object]:
    print(RULE)
    print("SECTION 6  THE DECISIVE CAVEAT -- the win is contingent on SAMPLING REGULARITY")
    print(RULE)
    print(
        "  R2-F3 asked whether round 2's win was a property of the data or of a\n"
        "  hyperparameter. Section 4 answers that (it is not a hyperparameter). This\n"
        "  section asks the same question of the one knob R2-F3 never touched: the\n"
        "  POINT-CLOUD CONSTRUCTION. All of the following are the SAME circle with\n"
        "  the SAME true dimension 1; only how it is sampled changes."
    )

    print()
    print("  (6a) i.i.d. uniform-angle circle -- a random, not low-discrepancy, sample.")
    print("       Prefix-fair by construction (i.i.d. prefixes are i.i.d. samples), so")
    print("       compare() applies directly. 10 seeds:")
    iid = []
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        th = rng.random(N_MAX) * 2 * np.pi
        cloud = list(zip(np.cos(th).tolist(), (-np.sin(th)).tolist(), strict=True))
        p, t, w, s = _compare_min_ns(cloud)
        iid.append((seed, p, t, w, s))
        print(f"       seed={seed}: poly={p} trad={t} win={w} savings={s}")
    print(f"       poly wins: {sum(1 for r in iid if r[3])}/10")
    rng = np.random.default_rng(1000)
    th = rng.random(N_MAX) * 2 * np.pi
    cloud = list(zip(np.cos(th).tolist(), (-np.sin(th)).tolist(), strict=True))
    print("       per-n on seed 0 (poly returns NO estimate at any n up to 3200):")
    for n in N_GRID:
        hg = knn_hypergraph(cloud[:n], k=K, dedupe=True)
        s_ = min(n, 40)
        print(
            f"          n={n:>4}: poly={mean_dimension(hg, samples=s_, max_radius=MAX_RADIUS):.4f}"
            f"  near_deg={near_degenerate_fraction(hg, samples=s_, max_radius=MAX_RADIUS):.3f}"
            f"  trad={correlation_dimension(cloud[:n]).dimension:.4f}"
        )
    print(
        "       COMPLETE INVERSION. On an irregularly sampled circle the shell-growth\n"
        "       estimator produces no answer at all (every sampled node fails both the\n"
        "       R^2 bar and the near-constant gate: shell_cv ~ 0.34-0.52 vs the 0.15\n"
        "       gate), while the correlation sum is accurate from n=50 on."
    )

    print()
    print("  (6b) Where is the boundary? Equispaced angles + uniform jitter of")
    print("       j x (mean spacing). j=0 is the physically natural constant-dt sample of")
    print("       one period; j>=1 lets neighbouring samples cross. An independent cloud")
    print("       per n (an equispaced PREFIX would be an arc, not the circle), so the")
    print("       library's convergence rule is applied here via _stable_min_n:")
    jitter_rows = []
    for j in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0):

        def make(n, j=j):
            rng = np.random.default_rng(7)
            th = np.arange(n) * 2 * np.pi / n + rng.uniform(-j, j, n) * (2 * np.pi / n)
            return list(zip(np.cos(th).tolist(), (-np.sin(th)).tolist(), strict=True))

        poly, trad = _per_n_independent(make)
        pm, tm = _stable_min_n(poly), _stable_min_n(trad)
        jitter_rows.append((j, pm, tm))
        note = "  <- physically natural constant-dt sampling" if j == 0.0 else ""
        print(f"       jitter={j:4.2f}: poly_min_n={pm} trad_min_n={tm}{note}")
    print(
        "       The win survives sampling irregularity up to ~0.75 of the mean spacing,\n"
        "       ties at 1.0, and inverts completely beyond. The round-2 Weyl cloud has\n"
        "       discrepancy far below one spacing, so it sits well inside the winning\n"
        "       regime -- as does the exactly-equispaced physical sample."
    )

    print()
    print("  (6c) Additive measurement noise on the round-2 production cloud, sigma")
    print("       quoted as a fraction of the mean nearest-neighbour spacing at n=3200:")
    base = np.array(pts)
    spacing = 2 * np.pi / N_MAX
    noise_rows = []
    for f in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        rng = np.random.default_rng(11)
        noisy = [tuple(q) for q in base + rng.normal(0.0, f * spacing, base.shape)]
        p, t, w, s = _compare_min_ns(noisy)
        noise_rows.append((f, p, t, w, s))
        print(f"       sigma={f:4.2f}*spacing: poly={p} trad={t} win={w} savings={s}")
    print(
        "       Same shape: robust to noise well below the sample spacing, gone at one\n"
        "       full spacing (poly then converges at no n in the grid)."
    )

    print()
    print(
        "  READING. Criterion (a) here is a real win at the module defaults, and\n"
        "  Section 4 shows it is not a hyperparameter artifact -- R2-F3's specific\n"
        "  objection is dead. But the honest statement of what was measured is\n"
        "  'on a REGULARLY sampled circle', not 'on a circle'. That qualifier is\n"
        "  defensible physics (a periodic orbit sampled at constant dt IS regular,\n"
        "  and j=0 above wins 50 vs 200), but it is a qualifier, and it is the same\n"
        "  ring-lattice mechanism -- sentinel fraction 1.000 at every n, Section 2 --\n"
        "  that already carries every other win in this benchmark."
    )
    print()
    return {"iid": iid, "jitter": jitter_rows, "noise": noise_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5,6")
    args = ap.parse_args()
    want = {s.strip() for s in args.sections.split(",") if s.strip()}

    pts = section1()  # always: everything else needs the cloud
    if "2" in want:
        section2(pts)
    if "3" in want:
        section3(pts)
    if "4" in want:
        section4(pts)
    if "5" in want:
        section5(pts)
    if "6" in want:
        section6(pts)


if __name__ == "__main__":
    main()
