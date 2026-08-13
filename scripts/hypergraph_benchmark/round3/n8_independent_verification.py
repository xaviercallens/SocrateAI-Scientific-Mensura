"""Independent skeptic verification of N8 / R2-F4 (round 3).

Written from scratch by the verification agent -- it does NOT import or run
`n8_near_constant_shell_gate.py`. Everything the repair agent claimed is
re-derived here:

  A. The documented anchor case (problem 01, n=200, k=6, max_radius=6) rebuilt
     from the leapfrog integrator, with BFS shells computed by a hand-written
     BFS and dimension/R^2 computed by numpy least-squares -- not by the
     repo's `_log_log_fit`.
  B. The slope_bound formula checked as a *bound*, by brute-force / random
     search over arrangements inside the observed band.
  C. Generalization: near-constant-but-not-exactly-constant shell sequences the
     repair agent did NOT test.
  D. Adversarial: sequences with real, large-magnitude curvature that must
     still be rejected, plus a search for the worst dimension error the gate
     will admit.
  E. N1 regression: exactly-constant sequences still take N1's branch.

Run: python scripts/hypergraph_benchmark/round3/n8_independent_verification.py
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph import knn_hypergraph, local_dimension, mean_dimension  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_SLOPE_BOUND,
    _log_log_fit,
)
from socrates.solvers import leapfrog  # noqa: E402

# ---------------------------------------------------------------- independent maths


def independent_fit(shells) -> tuple[float, float]:
    """dimension and R^2 from numpy, with no reference to the repo's fit code."""
    x = np.log(np.arange(1, len(shells) + 1, dtype=float))
    y = np.log(np.asarray(shells, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(slope) + 1.0, 1.0 - ss_res / ss_tot


def independent_slope_bound(shells) -> float:
    """The claimed bound, recomputed independently."""
    x = np.log(np.arange(1, len(shells) + 1, dtype=float))
    y = np.log(np.asarray(shells, dtype=float))
    w = float(y.max() - y.min())
    xc = x - x.mean()
    return w * float(np.abs(xc).sum()) / (2.0 * float((xc**2).sum()))


def independent_cv(shells) -> float:
    a = np.asarray(shells, dtype=float)
    return float(a.std()) / float(a.mean())


def bfs_shells(adj, source, max_radius):
    """Hand-written BFS shell sequence -- independent of local_dimension."""
    seen = {source}
    frontier = {source}
    shells = []
    for _ in range(max_radius):
        nxt = set()
        for u in frontier:
            nxt |= adj.get(u, set()) - seen
        seen |= nxt
        shells.append(len(nxt))
        if not nxt:
            break
        frontier = nxt
    return shells


# ---------------------------------------------------------------- A. anchor case


def part_a() -> None:
    print("=" * 78)
    print("A. ANCHOR CASE REBUILT FROM SCRATCH (problem 01, n=200, k=6, max_radius=6)")
    print("=" * 78)

    dt = 2e-4
    force = lambda q: -q  # noqa: E731
    n_steps = int(round(2 * np.pi / dt))
    run = leapfrog(force, [1.0], [0.0], dt=dt, n_steps=n_steps)
    t = run.times
    err = max(
        float(np.max(np.abs(run.positions[:, 0] - np.cos(t)))),
        float(np.max(np.abs(run.velocities[:, 0] + np.sin(t)))),
    )
    print(f"  solver check: max |x-cos(t)|, |v+sin(t)| = {err:.3e}  (must be < 1e-4)")
    assert err < 1e-4

    n_steps_total = len(run.times) - 1
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    idx = np.clip(
        np.floor(((np.arange(200) * phi) % 1.0) * n_steps_total).astype(int), 0, n_steps_total
    )
    pts = list(zip(run.positions[idx, 0].tolist(), run.velocities[idx, 0].tolist(), strict=True))
    print(f"  point cloud: {len(pts)} points, {len(set(idx.tolist()))} distinct indices")

    hg = knn_hypergraph(pts, k=6, dedupe=True)
    adj = hg.adjacency()
    print(f"  hypergraph: {len(hg.nodes)} nodes")

    # Find the documented anchor node using MY OWN BFS.
    target = [6, 5, 5, 6, 6, 6]
    hits = [u for u in sorted(hg.nodes) if bfs_shells(adj, u, 6) == target]
    print(f"  nodes with shells (6,5,5,6,6,6) via hand-written BFS: {len(hits)}")
    assert hits, "ANCHOR CASE NOT REPRODUCED -- the documented node does not exist"
    u = hits[0]
    vol = list(itertools.accumulate(target, initial=1))[1:]
    print(f"  node {u}: shells={target} -> volumes={tuple(vol)}")
    print("  doc says volumes (7,12,17,23,29,35)")
    assert tuple(vol) == (7, 12, 17, 23, 29, 35)

    dim_np, r2_np = independent_fit(target)
    print(f"  NUMPY least-squares : dimension={dim_np:.10f}  R^2={r2_np:.10f}")
    print("  doc says            : dimension=1.0333        R^2=0.0550")
    assert abs(dim_np - 1.0333) < 5e-5 and abs(r2_np - 0.0550) < 5e-5

    est = local_dimension(hg, u, max_radius=6)
    print(f"  repo local_dimension: dimension={est.dimension:.10f}  R^2={est.r_squared:.10f}")
    print(f"    volumes={est.volumes}")
    print(f"    degenerate={est.degenerate} near_degenerate={est.near_degenerate}")
    print(f"    shell_cv={est.shell_cv:.6f} slope_bound={est.slope_bound:.6f}")
    print(
        f"    |dim-1|={abs(est.dimension - 1.0):.6f} <= slope_bound: "
        f"{abs(est.dimension - 1.0) <= est.slope_bound}"
    )
    print(f"    is_well_fit(0.9)={est.is_well_fit(0.9)}  (pre-fix would be {est.r_squared >= 0.9})")
    print(f"    is_genuinely_well_fit(0.9)={est.is_genuinely_well_fit(0.9)}")
    assert abs(est.dimension - dim_np) < 1e-12, "repo fit disagrees with numpy"
    assert abs(est.r_squared - r2_np) < 1e-12, "repo R^2 disagrees with numpy"
    assert est.is_well_fit(0.9) and est.near_degenerate and not est.degenerate

    ests = [local_dimension(hg, v, max_radius=6) for v in sorted(hg.nodes)]
    kept = [e for e in ests if e.is_well_fit(0.9)]
    pre = [e for e in ests if e.r_squared >= 0.9]
    print()
    print(
        f"  WHOLE GRAPH: pre-fix kept {len(pre)}/{len(ests)}, post-fix kept {len(kept)}/{len(ests)}"
    )
    print(f"  mean_dimension(max_radius=6) = {mean_dimension(hg, max_radius=6):.6f}  (truth 1)")
    if kept:
        worst = max(abs(e.dimension - 1.0) for e in kept)
        print(f"  worst |dim-1| among kept = {worst:.6f}  (problem tolerance +/-0.15)")
    print()


# ---------------------------------------------------------------- B. bound is a bound


def part_b() -> None:
    print("=" * 78)
    print("B. IS slope_bound ACTUALLY A BOUND? brute force inside the observed band")
    print("=" * 78)
    rng = random.Random(20260813)
    worst_ratio = 0.0
    worst_case = None
    n_checked = 0
    for _ in range(20000):
        n = rng.randint(3, 8)
        lo = rng.uniform(1.0, 40.0)
        hi = lo * rng.uniform(1.0001, 4.0)
        shells = [rng.uniform(lo, hi) for _ in range(n)]
        shells[rng.randrange(n)] = lo
        shells[rng.randrange(n)] = hi
        f = _log_log_fit([float(r) for r in range(1, n + 1)], shells)
        if f.degenerate:
            continue
        n_checked += 1
        ratio = abs(f.slope) / f.slope_bound if f.slope_bound > 0 else 0.0
        if ratio > worst_ratio:
            worst_ratio, worst_case = ratio, (n, tuple(round(s, 4) for s in shells))
    print(f"  {n_checked} random sequences; max |slope| / slope_bound = {worst_ratio:.6f}")
    print(f"  worst case: {worst_case}")
    print(f"  BOUND HOLDS (ratio <= 1): {worst_ratio <= 1.0}")
    assert worst_ratio <= 1.0, "slope_bound is NOT a bound"

    # Also: the extremal arrangement should SATURATE the bound (i.e. it is tight,
    # not vacuously large). Put each y at whichever band end matches sign(x-xbar).
    for n in (4, 5, 6, 7):
        xs = [float(r) for r in range(1, n + 1)]
        lx = [math.log(x) for x in xs]
        mx = sum(lx) / n
        lo, hi = 4.0, 8.0
        ys = [hi if x > mx else lo for x in lx]
        f = _log_log_fit(xs, ys)
        print(
            f"  n={n} extremal arrangement: |slope|={abs(f.slope):.6f} "
            f"bound={f.slope_bound:.6f} tight={abs(abs(f.slope) - f.slope_bound) < 1e-9}"
        )
    print()


# ---------------------------------------------------------------- C. generalization


NEW_NEAR_CONSTANT = [
    # (shells, why this is new / what it probes)
    ((7, 7, 7, 7, 7, 8), "off-by-one at the LAST radius, base 7 (agent used base 6)"),
    ((12, 13, 12, 12, 13, 12), "two-valued alternation at base 12, amplitude 1"),
    ((6, 6, 6, 6, 6, 5), "off-by-one DOWN at the last radius"),
    ((10, 10, 11, 10, 10, 10), "single bump in the middle, base 10"),
    ((5, 5, 5, 5), "SHORT sequence (n=4), exactly constant -> N1 branch"),
    ((5, 5, 6, 5), "SHORT sequence (n=4), near constant"),
    ((20, 21, 20, 21, 20, 21), "high base, perfectly alternating, tiny relative amplitude"),
    ((6, 6, 6, 6, 6, 6, 6, 7), "LONG sequence (n=8), one off-by-one at the end"),
    ((100, 101, 99, 100, 102, 98), "large base, +/-2 noise: CV tiny"),
    ((3, 3, 3, 4, 3, 3), "small base 3 -- highest CV a +/-1 wobble can give"),
]

MUST_REJECT = [
    ((1, 10, 100, 1000, 10000, 100000), "exponential blow-up: huge dynamic range"),
    ((2, 4, 2, 40, 2, 4), "one enormous outlier spike"),
    ((50, 40, 30, 20, 10, 1), "strong monotone DECAY (real negative slope)"),
    ((1, 1, 1, 1, 1, 100), "flat then a 100x jump at the end"),
    ((6, 60, 6, 60, 6, 60), "10x-amplitude oscillation"),
    ((5, 5, 5, 30, 30, 30), "step function, 6x amplitude"),
    ((100, 1, 100, 1, 100, 1), "100x oscillation at high base"),
    ((1, 3, 9, 27, 81, 243), "geometric 3^r -- not a power law in r at all"),
]


def part_c_d() -> None:
    print("=" * 78)
    print("C. GENERALIZATION -- near-constant sequences the repair agent did NOT test")
    print("=" * 78)
    print(f"  thresholds: CV <= {NEAR_CONSTANT_CV}, bound <= {NEAR_CONSTANT_SLOPE_BOUND}")
    print(
        f"  {'shells':<28} {'dim':>8} {'R^2':>8} {'CV':>7} {'bound':>7} "
        f"{'deg':>6} {'near':>6} {'kept':>5}"
    )
    for shells, why in NEW_NEAR_CONSTANT:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        f = _log_log_fit(xs, [float(s) for s in shells])
        dim_np, r2_np = (float("nan"), float("nan"))
        if len(set(shells)) > 1:
            dim_np, r2_np = independent_fit(shells)
            assert abs((f.slope + 1) - dim_np) < 1e-9, shells
            assert abs(f.r_squared - r2_np) < 1e-9, shells
            assert abs(independent_cv(shells) - f.shell_cv) < 1e-12, shells
            assert abs(independent_slope_bound(shells) - f.slope_bound) < 1e-12, shells
        kept = f.r_squared >= 0.9 or f.near_degenerate
        print(
            f"  {str(shells):<28} {f.slope + 1:8.4f} {f.r_squared:8.4f} {f.shell_cv:7.4f} "
            f"{f.slope_bound:7.4f} {str(f.degenerate):>6} {str(f.near_degenerate):>6} "
            f"{'yes' if kept else 'NO':>5}   <- {why}"
        )
        # The guarantee must hold on every one of these.
        assert abs(f.slope) <= f.slope_bound + 1e-12, f"bound violated on {shells}"
        if f.near_degenerate:
            assert abs(f.slope) <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12, shells

    print()
    print("=" * 78)
    print("D. ADVERSARIAL -- real, large-magnitude curvature MUST still be rejected")
    print("=" * 78)
    print(f"  {'shells':<32} {'dim':>9} {'R^2':>8} {'CV':>8} {'bound':>8} {'near':>6} {'kept':>5}")
    for shells, why in MUST_REJECT:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        f = _log_log_fit(xs, [float(s) for s in shells])
        pre_kept = f.r_squared >= 0.9  # what the code did BEFORE N8
        kept = pre_kept or f.near_degenerate
        note = "" if kept == pre_kept else "  *** N8 CHANGED THIS ***"
        print(
            f"  {str(shells):<32} {f.slope + 1:9.4f} {f.r_squared:8.4f} {f.shell_cv:8.4f} "
            f"{f.slope_bound:8.4f} {str(f.near_degenerate):>6} {'yes' if kept else 'NO':>5}"
            f"   <- {why}{note}"
        )
        # The load-bearing claim: N8 must not excuse any of these, and must not
        # change the verdict on any of them. (One of them, the pure geometric
        # 10^r, happens to pass the PRE-EXISTING R^2 branch at R^2=0.94; that is
        # an old property of the estimator, not something N8 introduced, and the
        # `kept == pre_kept` assertion is what pins N8's responsibility.)
        assert f.near_degenerate is False, f"GATE TOO WIDE: {shells} ({why})"
        assert kept == pre_kept, f"N8 CHANGED A HIGH-CURVATURE VERDICT: {shells} ({why})"

    print()
    print("  -- does the gate accept EVERYTHING? exhaustive scan of small integer shells --")
    total = near = 0
    worst_err = 0.0
    worst_shells = None
    for shells in itertools.product(range(1, 13), repeat=4):
        xs = [1.0, 2.0, 3.0, 4.0]
        f = _log_log_fit(xs, [float(s) for s in shells])
        total += 1
        if f.near_degenerate:
            near += 1
            err = abs(f.slope)
            if err > worst_err:
                worst_err, worst_shells = err, shells
    print(
        f"  4-long integer shell sequences in [1,12]^4: {near}/{total} "
        f"({100 * near / total:.2f}%) flagged near-degenerate"
    )
    print(f"  worst |dimension - 1| admitted by the gate: {worst_err:.6f} on {worst_shells}")
    print(f"  (contract: must be <= NEAR_CONSTANT_SLOPE_BOUND = {NEAR_CONSTANT_SLOPE_BOUND})")
    assert worst_err <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12
    assert near / total < 0.25, "the gate accepts most of the space -- it is not a gate"

    total6 = near6 = 0
    worst6, worst6_shells = 0.0, None
    for shells in itertools.product(range(1, 9), repeat=6):
        f = _log_log_fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [float(s) for s in shells])
        total6 += 1
        if f.near_degenerate:
            near6 += 1
            if abs(f.slope) > worst6:
                worst6, worst6_shells = abs(f.slope), shells
    print(
        f"  6-long integer shell sequences in [1,8]^6: {near6}/{total6} "
        f"({100 * near6 / total6:.2f}%) flagged near-degenerate"
    )
    print(f"  worst |dimension - 1| admitted: {worst6:.6f} on {worst6_shells}")
    assert worst6 <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12
    print()


# ---------------------------------------------------------------- E. N1 regression


def part_e() -> None:
    print("=" * 78)
    print("E. N1 REGRESSION -- exactly-constant sequences still take N1's branch")
    print("=" * 78)
    ok = True
    for shell in (1.0, 2.0, 3.0, 6.0, 7.0, 12.0, 100.0, 1e6):
        for n in range(2, 12):
            xs = [float(x) for x in range(1, n + 1)]
            f = _log_log_fit(xs, [shell] * n)
            good = (
                abs(f.slope) < 1e-9
                and abs(f.r_squared - 1.0) < 1e-9
                and f.degenerate is True
                and f.near_degenerate is False
            )
            ok = ok and good
            if not good:
                print(
                    f"  FAIL shell={shell} n={n}: slope={f.slope} r2={f.r_squared} "
                    f"deg={f.degenerate} near={f.near_degenerate}"
                )
    print(
        f"  all 80 (shell, n) constant combos: degenerate=True, near_degenerate=False, "
        f"R^2=1.0 -> {ok}"
    )
    assert ok

    # And the real N1 artifact: the 400-point circle (exact ring lattice).
    pts = [(math.cos(2 * math.pi * i / 400), math.sin(2 * math.pi * i / 400)) for i in range(400)]
    hg = knn_hypergraph(pts, k=6, dedupe=True)
    est = local_dimension(hg, sorted(hg.nodes)[0], max_radius=6)
    print(
        f"  400-pt circle k=6: dim={est.dimension:.6f} R^2={est.r_squared:.6f} "
        f"degenerate={est.degenerate} near_degenerate={est.near_degenerate} "
        f"genuinely_well_fit={est.is_genuinely_well_fit()}"
    )
    assert est.degenerate is True and est.near_degenerate is False
    assert est.is_genuinely_well_fit() is False

    # Positive control: a genuine 2D grid must be untouched by N8.
    grid = [(float(i), float(j)) for i in range(30) for j in range(30)]
    hgg = knn_hypergraph(grid, k=8, dedupe=True)
    e2 = local_dimension(hgg, (14 * 30 + 14), max_radius=6)
    print(
        f"  30x30 grid k=8 centre node: dim={e2.dimension:.4f} R^2={e2.r_squared:.4f} "
        f"near_degenerate={e2.near_degenerate} genuinely_well_fit={e2.is_genuinely_well_fit()}"
    )
    assert e2.near_degenerate is False, "a genuine 2D grid must NOT be excused as near-constant"
    print()


if __name__ == "__main__":
    part_a()
    part_b()
    part_c_d()
    part_e()
    print("=" * 78)
    print("ALL INDEPENDENT CHECKS PASSED")
    print("=" * 78)
