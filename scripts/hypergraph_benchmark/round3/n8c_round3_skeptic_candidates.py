"""Round-3 independent skeptic, part 6: measured candidate discriminators for
the residual defect (NOT applied -- this is evidence for the repair loop).

The defect this round found: round-2 problem 06's recorded
`poly_algebraic_min_n` moves 400 -> 800, because at n=400 only 4 of 40 sampled
nodes pass R^2, and one near_degenerate node with shells (13,17,13,13,17,15)
and dimension 1.0587 (truth 2) gets equal weight in the mean, dragging it from
1.8434 (in tolerance) to 1.6864 (out). The fit window is length 6, so
NEAR_CONSTANT_MIN_FIT_LENGTH cannot reach it.

Two candidates are measured against every must-accept case on record:

  (A) tighten NEAR_CONSTANT_CV. The culprit sits at CV=0.1224; the worst
      must-accept on record ((7,5,5,6,6,6), docs 9.5) sits at 0.1178. So this
      works, but the margin is ~2%, i.e. knife-edge, and is reported as such.

  (B) a graph-level consensus condition: only honour the near-constant branch
      where near-constant nodes are the DOMINANT mode of the graph
      (`near_degenerate_fraction` high), which is what "this whole structure
      is a low-dynamic-range 1D object" actually looks like. Problem 01 reads
      0.98; the damaged Brownian graphs read 0.025-0.06.

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8c_round3_skeptic_anchor import build_problem01_points  # noqa: E402
from n8c_round3_skeptic_collateral import (  # noqa: E402
    brownian2d,
    brownian_round2_cloud,
    torus_lissajous_2d,
    torus_problem05_asbuilt,
)

from socrates.hypergraph.dimension import (  # noqa: E402
    _log_log_fit,
    local_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

# Every sequence the record REQUIRES the gate to accept.
MUST_ACCEPT = [
    ((6, 5, 5, 6, 6, 6), "docs 10.3 ANCHOR"),
    ((6, 6, 6, 6, 6, 7), "docs 9.5"),
    ((6, 5, 6, 6, 6, 6), "docs 9.5"),
    ((5, 6, 6, 6, 6, 6), "docs 9.5"),
    ((7, 5, 5, 6, 6, 6), "docs 9.5 -- worst CV on record"),
]
MUST_REJECT = [
    ((4, 8, 4, 8, 4, 8), "docs 9.5 negative control"),
    ((1, 1, 1, 1, 50, 50), "step"),
    ((1, 2, 3, 60, 61, 62), "saturating"),
    ((3, 30, 3, 30, 3, 30), "large oscillation"),
]
CULPRIT = ((13, 17, 13, 13, 17, 15), "round-2 problem 06 n=400, dim 1.0587 vs truth 2")


def cv_of(shells):
    fit = _log_log_fit([float(r) for r in range(1, len(shells) + 1)], [float(s) for s in shells])
    return fit.shell_cv, fit.slope_bound, fit.slope + 1.0


def main() -> int:
    print("=" * 92)
    print("PART 6A: CANDIDATE (A) -- tighten NEAR_CONSTANT_CV")
    print("=" * 92)
    print(f"{'shells':<28} {'CV':>8} {'bound':>8} {'dim':>8}   role")
    worst_accept = 0.0
    for shells, why in MUST_ACCEPT:
        cv, b, d = cv_of(shells)
        worst_accept = max(worst_accept, cv)
        print(f"{str(shells):<28} {cv:8.4f} {b:8.4f} {d:8.4f}   MUST ACCEPT ({why})")
    cv_c, b_c, d_c = cv_of(CULPRIT[0])
    print(f"{str(CULPRIT[0]):<28} {cv_c:8.4f} {b_c:8.4f} {d_c:8.4f}   MUST REJECT ({CULPRIT[1]})")
    for shells, why in MUST_REJECT:
        cv, b, d = cv_of(shells)
        print(f"{str(shells):<28} {cv:8.4f} {b:8.4f} {d:8.4f}   must reject ({why})")

    print(f"\n  worst must-accept CV = {worst_accept:.4f}")
    print(f"  culprit CV           = {cv_c:.4f}")
    gap = cv_c - worst_accept
    print(
        f"  usable window for NEAR_CONSTANT_CV = ({worst_accept:.4f}, {cv_c:.4f}]"
        f"  width {gap:.4f}  ({gap / worst_accept:.1%} of the lower edge)"
    )
    print("  VERDICT ON (A): it separates the cases, but the margin is ~2%. That is")
    print("  knife-edge -- exactly the kind of threshold the standing rules distrust.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("PART 6B: CANDIDATE (B) -- graph-level near_degenerate_fraction consensus")
    print("=" * 92)
    print(f"{'graph':<50} {'k':>2} {'R':>2} {'near_frac':>10} {'R2only_mean':>12}")

    def row(label, pts, k, mr):
        hg = knn_hypergraph(pts, k=k, dedupe=True)
        frac = near_degenerate_fraction(hg, samples=200, max_radius=mr)
        nodes = sorted(hg.nodes)
        stp = max(1, len(nodes) // 200)
        ests = [local_dimension(hg, u, max_radius=mr) for u in nodes[::stp][:200]]
        base = [e for e in ests if e.r_squared >= 0.9]
        mb = sum(e.dimension for e in base) / len(base) if base else float("nan")
        print(f"{label:<50} {k:2d} {mr:2d} {frac:10.4f} {mb:12.4f}")
        return frac

    p01 = build_problem01_points(200)
    f_anchor = row("problem 01 ring (MUST keep the branch)", p01, 6, 6)

    r2c = brownian_round2_cloud()
    f_bad = []
    for n in (400, 3200, 6400):
        f_bad.append(row(f"round-2 problem 06 production n={n} (MUST NOT)", r2c[:n], 10, 6))
    f_bad.append(row("Brownian n=1500 seed=101 (MUST NOT)", brownian2d(1500, 101), 8, 5))
    f_bad.append(row("Brownian n=1500 seed=109 (MUST NOT)", brownian2d(1500, 109), 6, 6))
    f_bad.append(row("torus Lissajous n=2000 (MUST NOT)", torus_lissajous_2d(2000), 6, 5))
    f_bad.append(row("torus problem05 as built (MUST NOT)", torus_problem05_asbuilt(), 10, 5))

    print(f"\n  problem 01 (branch must fire)      near_degenerate_fraction = {f_anchor:.4f}")
    print(f"  damaged graphs (must not fire)     max                      = {max(f_bad):.4f}")
    sep = f_anchor / max(max(f_bad), 1e-9)
    print(f"  separation ratio                   = {sep:.1f}x")
    print(
        "  VERDICT ON (B): a graph-level threshold anywhere in "
        f"({max(f_bad):.3f}, {f_anchor:.3f}) separates every case measured here,"
    )
    print("  a window ~2 orders of magnitude wider than (A)'s. It is a larger design")
    print("  change (node-local -> graph-level) and would need its own calibration, but")
    print("  it matches the physical claim the branch actually rests on: 'this whole")
    print("  structure is a low-dynamic-range ~1D object', not 'this one window looks flat'.")
    print("\n  NEITHER CANDIDATE IS APPLIED HERE -- both are behavioural design decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
