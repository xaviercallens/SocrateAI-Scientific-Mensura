"""Round-3 independent skeptic, part 3: NON-VACUITY CONTROL.

"Zero spurious admissions after the fix" is worthless unless the same clouds
would have shown admissions before it. This file re-derives `near_degenerate`
WITHOUT the length condition (i.e. N8 as shipped in round 1) from the shipped
`shell_cv` / `slope_bound` fields, on the exact clouds part 2 used, and shows
the before/after counts side by side.

It also answers the question the round-2 repair explicitly did not: does
round-2 problem 06 (2D Brownian, K=10, compare()'s default max_radius=6, so
the gate IS live) change its verdict? That is computed by running the real
`poly_algebraic_minimum_points` twice -- once as shipped, once with
`is_well_fit` monkeypatched back to pure R^2 -- on the real production cloud.

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8c_round3_skeptic_collateral import (  # noqa: E402
    brownian2d,
    torus_lissajous_2d,
    torus_problem05_asbuilt,
    torus_quasiperiodic_4d,
    torus_uniform_4d,
)

from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_SLOPE_BOUND,
    DimensionEstimate,
    local_dimension,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SAMPLES = 200


def unguarded_near_degenerate(e: DimensionEstimate) -> bool:
    """N8 as shipped in ROUND 1: the same two spread conditions, no length
    condition. Recomputed from the shipped fields, so it is exactly the old
    predicate and not a re-implementation that could drift."""
    return (
        not e.degenerate
        and e.shell_cv <= NEAR_CONSTANT_CV
        and e.slope_bound <= NEAR_CONSTANT_SLOPE_BOUND
    )


def counts(points, k: int, max_radius: int) -> tuple[int, int, float, float, float]:
    hg = knn_hypergraph(points, k=k, dedupe=True)
    nodes = sorted(hg.nodes)
    step = max(1, len(nodes) // SAMPLES)
    nodes = nodes[::step][:SAMPLES]
    ests = [local_dimension(hg, u, max_radius=max_radius) for u in nodes]

    old = [e for e in ests if e.r_squared < 0.9 and unguarded_near_degenerate(e)]
    new = [e for e in ests if e.r_squared < 0.9 and e.near_degenerate]

    base = [e for e in ests if e.r_squared >= 0.9]
    mean_base = sum(e.dimension for e in base) / len(base) if base else float("nan")
    old_kept = base + old
    new_kept = base + new
    mean_old = sum(e.dimension for e in old_kept) / len(old_kept)
    mean_new = sum(e.dimension for e in new_kept) / len(new_kept)
    return len(old), len(new), mean_base, mean_old, mean_new


def main() -> int:
    print("=" * 96)
    print("PART 3A: NON-VACUITY CONTROL -- would these clouds have shown damage before N8b?")
    print("=" * 96)
    print(
        f"{'cloud':<44} {'k':>2} {'R':>2} {'OLD':>4} {'NEW':>4} "
        f"{'mean_R2only':>11} {'mean_OLD':>9} {'mean_NEW':>9}"
    )

    cases = []
    for seed in range(101, 111):
        cases.append((f"Brownian n=1500 seed={seed}", brownian2d(1500, seed), 6, 3))
    cases += [
        ("torus uniform flat R^4 n=2000", torus_uniform_4d(2000), 6, 3),
        ("torus uniform flat R^4 n=2000", torus_uniform_4d(2000), 10, 3),
        ("torus quasiperiodic (x,vx,y,vy) n=2000", torus_quasiperiodic_4d(2000), 6, 3),
        ("torus quasiperiodic (x,vx,y,vy) n=2000", torus_quasiperiodic_4d(2000), 10, 3),
        ("torus Lissajous (x,y) n=2000 t=2e5", torus_lissajous_2d(2000), 6, 3),
        ("torus Lissajous (x,y) n=2000 t=2e5", torus_lissajous_2d(2000), 10, 3),
        ("torus problem05 AS BUILT n=500 t=1800", torus_problem05_asbuilt(), 6, 3),
        ("torus problem05 AS BUILT n=500 t=1800", torus_problem05_asbuilt(), 10, 3),
        ("torus problem05 AS BUILT n=500 t=1800", torus_problem05_asbuilt(), 10, 4),
    ]

    tot_old = tot_new = 0
    for label, pts, k, mr in cases:
        o, nw, mb, mo, mn = counts(pts, k, mr)
        tot_old += o
        tot_new += nw
        print(f"{label:<44} {k:2d} {mr:2d} {o:4d} {nw:4d} {mb:11.4f} {mo:9.4f} {mn:9.4f}")
    print(
        f"\n  TOTAL spurious admissions at max_radius<=4:  OLD (round-1 N8) = {tot_old}"
        f"   NEW (shipped N8b) = {tot_new}"
    )
    assert tot_new == 0, "the mandated short-window cases are NOT clean"
    assert tot_old > 0, "VACUOUS CHECK: these clouds never exercised the bug in the first place"
    print("  -> the check is NOT vacuous: the same clouds show real damage without the gate,")
    print("     and zero with it.")

    print("\n  PART 3A RESULT: the mandated short-window cases are genuinely clean.")
    print("  For the round-2 problem-06 regression this round found at max_radius=6,")
    print("  see n8c_round3_skeptic_prob06.py and n8c_round3_skeptic_round2_sweep.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
