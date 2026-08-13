"""Round-3 (N8d) step 4: verify the shipped consensus rule through PRODUCTION code.

Everything here calls `socrates.hypergraph.comparison.poly_algebraic_minimum_points`
and `socrates.hypergraph.dimension.mean_dimension` as they now stand on disk --
no re-implementation of the acceptance rule -- and compares against the pre-N8
baseline (R^2 alone) computed on the same estimates.

Checks:
  1. the documented R2-F4 anchor is preserved bit-for-bit;
  2. round-2 problem 06's regression is fixed (the culprit node is refused);
  3. round-2 problem 05 is unchanged;
  4. the FULL round-2 sweep: poly_algebraic_min_n for problems 01-10 at their
     production settings, shipped vs pre-N8 baseline;
  5. N1 (exactly-constant) is untouched.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8d_clouds import ANCHOR_SHELLS, ANCHOR_VOLUMES, SPECS, anchor_cloud, load_cloud  # noqa: E402
from n8d_policy import R2_THRESHOLD, measure_grid  # noqa: E402

from socrates.hypergraph.comparison import poly_algebraic_minimum_points  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CONSENSUS_FRACTION,
    NEAR_CONSTANT_CONSENSUS_TOLERANCE,
    _log_log_fit,
    local_dimension,
    mean_dimension,
    near_constant_consensus,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def baseline_min_n(measurements, true_dim, tolerance) -> int | None:
    """pre-N8 poly_algebraic_min_n: R^2 alone, same stable-convergence rule."""
    dims = []
    for m in measurements:
        kept = [e.dimension for e in m.estimates if e.r_squared >= R2_THRESHOLD]
        dims.append((m.n, sum(kept) / len(kept) if kept else float("nan")))
    for i, (n, dim) in enumerate(dims):
        if not math.isfinite(dim) or abs(dim - true_dim) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dim) <= tolerance for _, d in dims[i:]):
            return n
    return None


def main() -> int:
    print("=" * 100)
    print("N8d STEP 4: verification of the shipped graph-level consensus rule")
    print(f"    NEAR_CONSTANT_CONSENSUS_TOLERANCE = {NEAR_CONSTANT_CONSENSUS_TOLERANCE}")
    print(f"    NEAR_CONSTANT_CONSENSUS_FRACTION  = {NEAR_CONSTANT_CONSENSUS_FRACTION}")
    print("=" * 100)

    # ---------------------------------------------------------------- 1 anchor
    print("\n[1] DOCUMENTED R2-F4 ANCHOR: problem 01 cloud, n=200, k=6, max_radius=6")
    pts = anchor_cloud()
    hg = knn_hypergraph(pts, k=6, dedupe=True)
    ests = [local_dimension(hg, node, max_radius=6) for node in sorted(hg.nodes)]
    hits = [e for e in ests if e.volumes == ANCHOR_VOLUMES]
    check(
        "anchor shell sequence present", len(hits) >= 1,
        f"{len(hits)} node(s) with {ANCHOR_SHELLS}",
    )
    anchor = hits[0]
    print(f"      volumes={anchor.volumes} dim={anchor.dimension:.10f} R^2={anchor.r_squared:.10f}")
    print(f"      shell_cv={anchor.shell_cv:.6f} slope_bound={anchor.slope_bound:.6f} "
          f"near_degenerate={anchor.near_degenerate}")
    check("anchor still near_degenerate", anchor.near_degenerate is True)
    check("anchor dimension unchanged", abs(anchor.dimension - 1.0333275633) < 1e-9,
          f"{anchor.dimension:.10f} (doc 1.0333)")
    check("anchor R^2 unchanged", abs(anchor.r_squared - 0.0550202079) < 1e-9,
          f"{anchor.r_squared:.10f} (doc 0.0550)")

    consensus = near_constant_consensus(ests, threshold=0.9)
    n_conf = sum(1 for e in ests if e.r_squared >= 0.9)
    frac = sum(1 for e in ests if e.near_degenerate) / len(ests)
    check("anchor graph reaches consensus", consensus is True,
          f"confident nodes={n_conf} (fallback branch), near_degenerate_fraction={frac:.4f} "
          f">= {NEAR_CONSTANT_CONSENSUS_FRACTION}")
    kept = [e for e in ests if e.is_well_fit(0.9, near_constant_consensus=consensus)]
    base = [e for e in ests if e.r_squared >= 0.9]
    mean = mean_dimension(hg, max_radius=6)
    check(
        "anchor kept-node count unchanged (196/200)", len(kept) == 196,
        f"{len(kept)}/{len(ests)}",
    )
    check(
        "anchor mean_dimension unchanged", abs(mean - 0.994266) < 1e-5,
        f"{mean:.6f} (truth 1.0)",
    )
    check("R^2 alone would still keep nothing", len(base) == 0, f"{len(base)}/200")
    worst = max(abs(e.dimension - 1.0) for e in kept)
    check(
        "worst kept |dim - 1| inside problem 01's +/-0.15", worst < 0.15, f"{worst:.6f}"
    )

    # ------------------------------------------------------------- 2 problem 06
    print("\n[2] ROUND-2 PROBLEM 06 REGRESSION (2D Brownian, k=10, max_radius=6, truth 2.0)")
    culprit_shells = (13, 17, 13, 13, 17, 15)
    fit = _log_log_fit([1.0, 2, 3, 4, 5, 6], [float(s) for s in culprit_shells])
    print(f"      culprit {culprit_shells}: dim={fit.slope + 1:.4f} R^2={fit.r_squared:.4f} "
          f"cv={fit.shell_cv:.4f} bound={fit.slope_bound:.4f} "
          f"near_degenerate={fit.near_degenerate}")
    check("culprit is still a near-constant CANDIDATE (node-local gate unchanged)",
          fit.near_degenerate is True)

    spec06 = SPECS["06"]
    pts06 = load_cloud("06")
    rows06 = measure_grid(pts06, spec06)
    row400 = next(m for m in rows06 if m.n == 400)
    cons400 = near_constant_consensus(list(row400.estimates), threshold=0.9)
    conf400 = [e for e in row400.estimates if e.r_squared >= 0.9]
    cm = sum(e.dimension for e in conf400) / len(conf400)
    check("problem 06 n=400 graph REFUSES consensus", cons400 is False,
          f"{len(conf400)} confident nodes, mean {cm:.4f}, |mean-1|={abs(cm - 1):.4f} "
          f"> {NEAR_CONSTANT_CONSENSUS_TOLERANCE}")
    nd400 = [e for e in row400.estimates if e.near_degenerate]
    print(f"      near-constant candidates at n=400: {len(nd400)} "
          f"{[e.volumes for e in nd400]} dim={[round(e.dimension, 4) for e in nd400]}")
    hg400 = knn_hypergraph(pts06[:400], k=spec06.k, dedupe=True)
    m400 = mean_dimension(hg400, samples=40, max_radius=spec06.max_radius)
    check("problem 06 n=400 mean_dimension back to the R^2-only value",
          abs(m400 - 1.8434) < 1e-3, f"{m400:.4f} (was 1.6864 under the node-local gate)")
    check("problem 06 n=400 now inside its +/-0.3 tolerance", abs(m400 - 2.0) <= 0.3,
          f"|{m400:.4f} - 2.0| = {abs(m400 - 2.0):.4f}")

    # ------------------------------------------------------------- 3+4 full sweep
    print("\n[3+4] FULL ROUND-2 SWEEP: poly_algebraic_min_n at production settings")
    print("      (shipped = production poly_algebraic_minimum_points; baseline = R^2 alone)")
    print(f"\n      {'prob':>4} {'true':>5} {'tol':>5} {'k':>3} {'R':>2} "
          f"{'pre_N8':>8} {'shipped':>8}  verdict")
    changed = []
    for problem, spec in SPECS.items():
        points = load_cloud(problem)
        rows = measure_grid(points, spec)
        pre = baseline_min_n(rows, spec.true_dimension, spec.tolerance)
        shipped = poly_algebraic_minimum_points(
            points,
            spec.true_dimension,
            spec.tolerance,
            k=spec.k,
            n_grid=spec.n_grid,
            max_radius=spec.max_radius,
            samples=spec.samples,
        )
        same = pre == shipped
        if not same:
            changed.append((problem, pre, shipped))
        print(f"      {problem:>4} {spec.true_dimension:>5} {spec.tolerance:>5} {spec.k:>3} "
              f"{spec.max_radius:>2} {str(pre):>8} {str(shipped):>8}  "
              f"{'unchanged' if same else 'CHANGED'}")
    check("no round-2 poly_algebraic_min_n moves away from its pre-N8 value",
          not changed, f"changed: {changed}" if changed else "all 10 identical")

    print("\n      problem 05 detail (must be unchanged through all three rounds):")
    spec05 = SPECS["05"]
    m05 = poly_algebraic_minimum_points(
        load_cloud("05"), spec05.true_dimension, spec05.tolerance, k=spec05.k,
        n_grid=spec05.n_grid, max_radius=spec05.max_radius, samples=40)
    check("problem 05 poly_algebraic_min_n == 400", m05 == 400, str(m05))

    # ------------------------------------------------------------------ 5 N1
    print("\n[5] N1 (EXACTLY-CONSTANT) REGRESSION")
    ok = True
    for shell in range(1, 21):
        for length in range(2, 12):
            f = _log_log_fit(
                [float(r) for r in range(1, length + 1)], [float(shell)] * length
            )
            if not (f.degenerate and not f.near_degenerate and f.r_squared == 1.0):
                ok = False
    check("200 exactly-constant combos: degenerate=True, near_degenerate=False, R^2=1.0", ok)

    ring = [
        (math.cos(2 * math.pi * i / 400), math.sin(2 * math.pi * i / 400)) for i in range(400)
    ]
    hgr = knn_hypergraph(ring, k=6, dedupe=True)
    e0 = local_dimension(hgr, sorted(hgr.nodes)[0], max_radius=6)
    check("400-pt exact circle still takes N1's branch",
          e0.degenerate and not e0.near_degenerate and e0.r_squared == 1.0,
          f"dim={e0.dimension:.6f} R^2={e0.r_squared:.6f}")
    check(
        "exact circle mean_dimension still 1.0",
        abs(mean_dimension(hgr, max_radius=6) - 1.0) < 1e-9,
    )
    check(
        "exact circle near_degenerate_fraction still 0",
        near_degenerate_fraction(hgr, max_radius=6) == 0.0,
    )

    grid = [(float(i), float(j)) for i in range(30) for j in range(30)]
    hgg = knn_hypergraph(grid, k=8, dedupe=True)
    centre = sorted(hgg.nodes)[len(hgg.nodes) // 2]
    eg = local_dimension(hgg, centre, max_radius=6)
    check("30x30 grid centre still measures ~2 and is not excused",
          abs(eg.dimension - 2.0) < 0.15 and not eg.near_degenerate,
          f"dim={eg.dimension:.4f} R^2={eg.r_squared:.4f}")

    check("near_constant_consensus([]) is False", near_constant_consensus([]) is False)

    print("\n" + "=" * 100)
    if FAILURES:
        print(f"FAILED CHECKS ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL N8d VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
