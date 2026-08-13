"""Round-3 independent skeptic, part 3B (isolated): round-2 problem 06.

Round-2 problem 06 (2D Brownian) runs `compare()`, whose default
`max_radius` is 6. Fit windows there are length 6 >= NEAR_CONSTANT_MIN_FIT_LENGTH,
so the near-constant branch IS live -- contrary to the round-2 repair's
reasoning, which checked only round-2 problem 01 (max_radius=3) on the grounds
that the length gate "can only remove acceptances". That is true relative to
round 1's N8, but N8 as a whole ADDS acceptances relative to the pre-N8 tree,
and this is a length-6 caller.

This file measures poly_algebraic_min_n three ways on the real production
cloud -- pre-N8 (pure R^2), N8 round-1 (no length gate), N8b (as shipped) --
and prints the per-n table that drives the difference.

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_prob06.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8c_round3_skeptic_collateral import brownian_round2_cloud  # noqa: E402

from socrates.hypergraph import comparison  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_SLOPE_BOUND,
    DimensionEstimate,
    local_dimension,
    mean_dimension,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600)
K = 10
TRUE_DIM = 2.0
TOL = 0.3


@contextmanager
def well_fit_mode(mode: str):
    """Swap DimensionEstimate.is_well_fit for one of the three historical
    behaviours, so all three are measured by the SAME production code path."""
    # N8d gave is_well_fit a keyword-only `near_constant_consensus` argument
    # (the graph-level decision `mean_dimension` now makes once per graph).
    # The historical stand-ins below must accept and ignore it, or they cannot
    # be called through the same production path any more.
    original = DimensionEstimate.is_well_fit
    if mode == "pre-N8":

        def f(self, threshold=0.95, **_kwargs):
            return self.r_squared >= threshold
    elif mode == "N8-round1":

        def f(self, threshold=0.95, **_kwargs):
            return self.r_squared >= threshold or (
                not self.degenerate
                and self.shell_cv <= NEAR_CONSTANT_CV
                and self.slope_bound <= NEAR_CONSTANT_SLOPE_BOUND
            )
    elif mode == "N8b-shipped":

        def f(self, threshold=0.95, **_kwargs):
            return self.r_squared >= threshold or self.near_degenerate
    elif mode == "N8d-shipped":
        f = original
    else:
        raise ValueError(mode)
    DimensionEstimate.is_well_fit = f
    try:
        yield
    finally:
        DimensionEstimate.is_well_fit = original


def main() -> int:
    points = brownian_round2_cloud()
    print(
        f"round-2 problem 06 production cloud: {len(points)} points, "
        f"K={K}, true_dim={TRUE_DIM}, tolerance={TOL}"
    )

    print("\n=== poly_algebraic_min_n, same production call, four behaviours ===")
    results = {}
    for mode in ("pre-N8", "N8-round1", "N8b-shipped", "N8d-shipped"):
        with well_fit_mode(mode):
            results[mode] = comparison.poly_algebraic_minimum_points(
                points, TRUE_DIM, TOL, k=K, n_grid=N_GRID
            )
        print(f"  {mode:<12} poly_algebraic_min_n = {results[mode]}")

    print("\n=== per-n mean_dimension(samples=40, max_radius=6) ===")
    print(
        f"{'n':>7} {'pre-N8':>9} {'N8-rd1':>9} {'N8b':>9} {'N8d':>9}   "
        "in-tolerance? (|d-2|<0.3)"
    )
    for n in N_GRID:
        if n > len(points):
            break
        hg = knn_hypergraph(points[:n], k=K, dedupe=True)
        vals = {}
        for mode in ("pre-N8", "N8-round1", "N8b-shipped", "N8d-shipped"):
            with well_fit_mode(mode):
                vals[mode] = mean_dimension(hg, samples=40, max_radius=6)
        ok = {m: abs(v - TRUE_DIM) < TOL for m, v in vals.items()}
        marks = " ".join(
            f"{m}={'IN' if ok[m] else 'OUT'}"
            for m in ("pre-N8", "N8-round1", "N8b-shipped", "N8d-shipped")
        )
        print(
            f"{n:7d} {vals['pre-N8']:9.4f} {vals['N8-round1']:9.4f} "
            f"{vals['N8b-shipped']:9.4f} {vals['N8d-shipped']:9.4f}   {marks}"
        )

    print("\n=== the culprit node(s) at the n where the verdict flips ===")
    for n in (200, 400, 800):
        hg = knn_hypergraph(points[:n], k=K, dedupe=True)
        nodes = sorted(hg.nodes)
        step = max(1, len(nodes) // 40)
        nodes = nodes[::step][:40]
        ests = [local_dimension(hg, u, max_radius=6) for u in nodes]
        adm = [e for e in ests if e.near_degenerate and e.r_squared < 0.9]
        base = [e for e in ests if e.r_squared >= 0.9]
        mb = sum(e.dimension for e in base) / len(base)
        print(
            f"  n={n}: sampled {len(ests)}, R^2-only kept {len(base)} (mean {mb:.4f}), "
            f"near_degenerate admitted {len(adm)}"
        )
        for e in adm:
            shells = tuple(
                e.volumes[i] - (e.volumes[i - 1] if i else 1) for i in range(len(e.volumes))
            )
            print(
                f"      volumes={e.volumes} shells={shells} dim={e.dimension:.4f} "
                f"R^2={e.r_squared:.4f} cv={e.shell_cv:.4f} bound={e.slope_bound:.4f} "
                f"|dim-2|={abs(e.dimension - 2):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
